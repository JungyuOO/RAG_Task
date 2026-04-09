from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.config import Settings
from app.rag.bge_embedding_server import EmbeddingPayloadTooLargeError
from app.rag.chunking_markdown import StructuredMarkdownChunker
from app.rag.ingestion_pdf import DocumentIngestor
from app.rag.types import Document
from app.rag.utils import extracted_markdown_path, stable_hash
from app.rag.version_manager import VersionManager
from app.storage import CacheRepository, IndexRepository

logger = logging.getLogger("rag.startup")


class IndexingService:
    def __init__(
        self,
        settings: Settings,
        ingestor: DocumentIngestor,
        structured_chunker: StructuredMarkdownChunker,
        embedder: Any,
        index_repository: IndexRepository,
        embedding_cache_repository: CacheRepository,
    ) -> None:
        self.settings = settings
        self.ingestor = ingestor
        self.structured_chunker = structured_chunker
        self.embedder = embedder
        self.index_repository = index_repository
        self.embedding_cache_repository = embedding_cache_repository
        self.version_manager = VersionManager()

    def _document_group_for_source_path(self, source_path: Path) -> str:
        customer_dirs = [
            (self.settings.rag_source_dir / "generated").resolve(),
            (self.settings.rag_source_dir / "generated_pdf").resolve(),
            (self.settings.rag_source_dir / "chat_uploads").resolve(),
        ]
        try:
            resolved = source_path.resolve()
        except Exception:
            resolved = source_path
        for customer_dir in customer_dirs:
            if resolved == customer_dir or customer_dir in resolved.parents:
                return "customer_generated"
        return "official_ocp"

    def _doc_type_for_source_path(self, source_path: Path) -> str:
        if self._document_group_for_source_path(source_path) == "customer_generated":
            return "operation_manual"
        return "official"

    def _encode_chunk(self, text: str) -> list[float]:
        return self.embedder.encode_passage(text)

    def _embed_batch_size(self) -> int:
        return max(1, int(self.settings.embedding_batch_size))

    def _embed_parallel_workers(self) -> int:
        return max(1, int(self.settings.embedding_parallel_workers))

    def _embed_batch_char_limit(self) -> int:
        if self.settings.embedding_backend != "tei":
            return 0
        return max(0, int(self.settings.embedding_batch_char_limit))

    @staticmethod
    def _split_embed_batches(
        chunks: list[Any],
        miss_indices: list[int],
        batch_size: int,
        batch_char_limit: int,
    ) -> list[tuple[int, list[int], list[str]]]:
        batches: list[tuple[int, list[int], list[str]]] = []
        batch_idx: list[int] = []
        batch_texts: list[str] = []
        batch_chars = 0

        for miss_idx in miss_indices:
            text = chunks[miss_idx].text
            text_chars = len(text)
            would_exceed_count = len(batch_idx) >= batch_size
            would_exceed_chars = bool(
                batch_char_limit and batch_idx and batch_chars + text_chars > batch_char_limit
            )

            if would_exceed_count or would_exceed_chars:
                batches.append((len(batches), batch_idx, batch_texts))
                batch_idx = []
                batch_texts = []
                batch_chars = 0

            batch_idx.append(miss_idx)
            batch_texts.append(text)
            batch_chars += text_chars

        if batch_idx:
            batches.append((len(batches), batch_idx, batch_texts))

        return batches

    def _embed_with_backoff(self, batch_texts: list[str], batch_id: str) -> list[list[float]]:
        total_chars = sum(len(text) for text in batch_texts)
        avg_chars = total_chars / max(len(batch_texts), 1)
        logger.info(
            "[EmbedBatch] 시작: id=%s thread=%s size=%d avg_chars=%.1f total_chars=%d",
            batch_id,
            threading.current_thread().name,
            len(batch_texts),
            avg_chars,
            total_chars,
        )

        t0 = time.perf_counter()
        try:
            result = self.embedder.encode_batch(batch_texts)
        except EmbeddingPayloadTooLargeError:
            if len(batch_texts) == 1:
                logger.error(
                    "[EmbedBatch] 단일 청크도 TEI 한도를 초과했습니다: id=%s chars=%d",
                    batch_id,
                    total_chars,
                )
                raise

            mid = max(1, len(batch_texts) // 2)
            logger.warning(
                "[EmbedBatch] payload too large, splitting: id=%s size=%d total_chars=%d -> %d + %d",
                batch_id,
                len(batch_texts),
                total_chars,
                mid,
                len(batch_texts) - mid,
            )
            left = self._embed_with_backoff(batch_texts[:mid], f"{batch_id}.0")
            right = self._embed_with_backoff(batch_texts[mid:], f"{batch_id}.1")
            result = left + right

        dt = time.perf_counter() - t0
        logger.info(
            "[EmbedBatch] 완료: id=%s thread=%s size=%d avg_chars=%.1f total_chars=%d took=%.2fs",
            batch_id,
            threading.current_thread().name,
            len(batch_texts),
            avg_chars,
            total_chars,
            dt,
        )
        return result

    def _batch_embed_chunks(
        self,
        chunks: list,
        progress_callback=None,
        progress_stage: str = "embed",
        progress_meta_fn=None,
    ) -> tuple[list[list[float]], int, int, float]:
        """Embed uncached chunks in bounded batches and update the embedding cache."""
        total = len(chunks)
        cache_keys = [stable_hash(chunk.text) for chunk in chunks]

        vectors: list[list[float] | None] = [None] * total
        miss_indices: list[int] = []
        cache_hits = 0
        for index, key in enumerate(cache_keys):
            cached = self.embedding_cache_repository.get(key)
            if cached is not None:
                vectors[index] = cached["vector"]
                cache_hits += 1
            else:
                miss_indices.append(index)

        cache_misses = len(miss_indices)
        batch_size = self._embed_batch_size()
        batch_char_limit = self._embed_batch_char_limit()
        batches = self._split_embed_batches(chunks, miss_indices, batch_size, batch_char_limit)

        completed_chunks = 0
        workers = min(self._embed_parallel_workers(), len(batches)) or 1
        logger.info(
            "[EmbedConfig] batch_size=%d batch_char_limit=%d configured_workers=%d actual_workers=%d total_batches=%d",
            batch_size,
            batch_char_limit,
            self._embed_parallel_workers(),
            workers,
            len(batches),
        )

        te_total = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(self._embed_with_backoff, batch_texts, str(batch_id)): batch_idx
                for batch_id, batch_idx, batch_texts in batches
            }
            for future in as_completed(future_map):
                batch_idx = future_map[future]
                batch_vectors = future.result()
                for index, vector in zip(batch_idx, batch_vectors):
                    vectors[index] = vector
                    self.embedding_cache_repository.set(cache_keys[index], {"vector": vector})
                completed_chunks += len(batch_idx)
                if progress_callback and total > 0:
                    meta = progress_meta_fn(batch_idx[-1]) if progress_meta_fn else {}
                    progress_callback(progress_stage, cache_hits + completed_chunks, total, meta)

        embed_api_total = time.perf_counter() - te_total
        logger.info(
            "[Embed] 배치 임베딩 완료: 청크 %d개(캐시히트 %d / 미스 %d), API %.2fs, 요청 %d회(병렬 %d), 청크당 평균 %.3fs",
            total,
            cache_hits,
            cache_misses,
            embed_api_total,
            len(batches),
            workers,
            embed_api_total / max(cache_misses, 1),
        )
        return vectors, cache_hits, cache_misses, embed_api_total  # type: ignore[return-value]

    def rebuild_index(self, source_paths: list[Path], progress_callback=None) -> dict:
        t0 = time.perf_counter()
        logger.info("[Timing][rebuild] 시작 (파일 %d개)", len(source_paths))

        cache_clear_start = time.perf_counter()
        self.embedding_cache_repository.clear()
        logger.info("[Timing][rebuild] embedding cache cleared: %.2fs", time.perf_counter() - cache_clear_start)

        clear_start = time.perf_counter()
        self.index_repository.save([], [])
        logger.info("[Timing][rebuild] 기존 인덱스 초기화 완료: %.2fs", time.perf_counter() - clear_start)

        indexed_files = 0
        indexed_chunks = 0
        skipped_files = 0

        for source_path in source_paths:
            def _progress(stage, current, total, meta=None):
                if not progress_callback:
                    return
                progress_callback(
                    stage,
                    current,
                    total,
                    {
                        "file_name": source_path.name,
                        "source_path": str(source_path),
                        **(meta or {}),
                    },
                )

            result = self.index_single_file(source_path, progress_callback=_progress)
            indexed_chunks += result.get("indexed_chunks", 0)
            if result.get("skipped"):
                skipped_files += 1
            else:
                indexed_files += 1

        logger.info(
            "[Timing][rebuild] 전체 소요: %.2fs (파일별 순차 추출/임베딩/저장)",
            time.perf_counter() - t0,
        )
        return {
            "indexed_files": indexed_files,
            "indexed_chunks": indexed_chunks,
            "skipped_files": skipped_files,
        }

    def index_markdown_file(self, source_path: Path, doc_type: str = "operation_manual") -> dict:
        """Index a markdown document directly without PDF extraction."""
        markdown_text = source_path.read_text(encoding="utf-8")
        if not markdown_text.strip():
            return {"indexed_chunks": 0, "indexed_pages": 0, "skipped": True}

        doc_id = stable_hash(str(source_path))
        document_group = self._document_group_for_source_path(source_path)
        document = Document(
            doc_id=doc_id,
            source_path=str(source_path),
            page_number=1,
            text=markdown_text,
            metadata={"loader": "markdown", "doc_type": doc_type, "document_group": document_group},
        )
        chunks = self.structured_chunker.split([document], markdown_text=markdown_text)

        version_tag = self.version_manager.detect_version_from_path(source_path)
        for chunk in chunks:
            chunk.metadata["doc_type"] = doc_type
            chunk.metadata["document_group"] = document_group
            if version_tag:
                chunk.metadata["version_tag"] = version_tag

        vectors, _, _, _ = self._batch_embed_chunks(chunks)

        self.index_repository.upsert_document(str(source_path), chunks, vectors)
        logger.info("[IndexMarkdown] %s -> %d chunks", source_path.name, len(chunks))
        return {"indexed_chunks": len(chunks), "indexed_pages": 1, "skipped": False}

    def index_single_file(self, source_path: Path, progress_callback=None) -> dict:
        t0 = time.perf_counter()
        logger.info("[Timing][%s] 시작", source_path.name)

        documents, _skipped = self.ingestor.ingest_paths([source_path], progress_callback=progress_callback)
        t1 = time.perf_counter()
        logger.info("[Timing][%s] PDF 추출 완료: %.2fs, 페이지 %d개", source_path.name, t1 - t0, len(documents))

        if not documents:
            return {"indexed_chunks": 0, "indexed_pages": 0, "skipped": True}

        document_group = self._document_group_for_source_path(source_path)
        doc_type = self._doc_type_for_source_path(source_path)

        markdown_text = self.load_extracted_markdown(source_path)
        if not markdown_text and source_path.suffix.lower() == ".md":
            markdown_text = documents[0].text

        strategy = "structured_markdown"
        chunks = self.structured_chunker.split(documents, markdown_text=markdown_text)
        t2 = time.perf_counter()
        logger.info(
            "[Timing][%s] 청킹 완료: %.2fs (전략=%s, doc_type=%s), 청크 %d개",
            source_path.name,
            t2 - t1,
            strategy,
            doc_type,
            len(chunks),
        )

        loaders = [doc.metadata.get("loader") for doc in documents if doc.metadata.get("loader")]
        representative_loader = loaders[0] if loaders else None
        version_tag = self.version_manager.detect_version_from_path(source_path)
        for chunk in chunks:
            chunk.metadata["doc_type"] = doc_type
            chunk.metadata["document_group"] = document_group
            if representative_loader and "loader" not in chunk.metadata:
                chunk.metadata["loader"] = representative_loader
            if version_tag:
                chunk.metadata["version_tag"] = version_tag

        file_name = source_path.name
        vectors, cache_hits, cache_misses, embed_api_total = self._batch_embed_chunks(
            chunks,
            progress_callback=progress_callback,
            progress_meta_fn=lambda _i: {"file_name": file_name, "source_path": str(source_path)},
        )

        t3 = time.perf_counter()
        logger.info(
            "[Timing][%s] 임베딩 완료: %.2fs (API %.2fs, 캐시히트 %d / 미스 %d)",
            source_path.name,
            t3 - t2,
            embed_api_total,
            cache_hits,
            cache_misses,
        )

        self.index_repository.upsert_document(str(source_path), chunks, vectors)
        t4 = time.perf_counter()
        logger.info("[Timing][%s] DB 저장 완료: %.2fs", source_path.name, t4 - t3)
        logger.info(
            "[Timing][%s] 전체 소요: %.2fs (추출 %.2fs / 청킹 %.2fs / 임베딩 %.2fs / DB %.2fs)",
            source_path.name,
            t4 - t0,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
        )

        return {
            "indexed_chunks": len(chunks),
            "indexed_pages": len(documents),
            "skipped": False,
        }

    def chunk_documents(self, documents: list) -> list:
        documents_by_path: dict[str, list] = defaultdict(list)
        for document in documents:
            documents_by_path[document.source_path].append(document)

        chunks = []
        for source_path, source_documents in documents_by_path.items():
            markdown_text = self.load_extracted_markdown(Path(source_path))
            source_chunks = self.structured_chunker.split(source_documents, markdown_text=markdown_text)
            loaders = [doc.metadata.get("loader") for doc in source_documents if doc.metadata.get("loader")]
            representative_loader = loaders[0] if loaders else None
            document_group = self._document_group_for_source_path(Path(source_path))
            for chunk in source_chunks:
                chunk.metadata["document_group"] = document_group
                if representative_loader and "loader" not in chunk.metadata:
                    chunk.metadata["loader"] = representative_loader
            chunks.extend(source_chunks)
        return chunks

    def load_extracted_markdown(self, source_path: Path) -> str | None:
        markdown_path = extracted_markdown_path(self.settings.rag_extract_dir, source_path)
        if not markdown_path.exists():
            return None
        return markdown_path.read_text(encoding="utf-8")

    _LIBRARY_EXTS = {".pdf", ".md"}

    def list_library_documents(self) -> dict:
        source_files = [
            path
            for path in self.settings.rag_source_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self._LIBRARY_EXTS
            and "chat_uploads" not in str(path).replace("\\", "/")
            and not (
                path.suffix.lower() == ".md"
                and self._document_group_for_source_path(path) == "customer_generated"
            )
        ]
        all_docs = self.index_repository.list_documents()

        indexed_by_path = {doc["source_path"]: doc for doc in all_docs}
        indexed_by_rel: dict[str, dict] = {}
        for doc in all_docs:
            stored = doc["source_path"].replace("\\", "/")
            src_dir = str(self.settings.rag_source_dir).replace("\\", "/")
            for sep in (src_dir + "/", "pdfs/", "corpus/pdfs/"):
                if sep in stored:
                    indexed_by_rel[stored.split(sep, 1)[-1]] = doc
                    break

        indexed_documents = []
        for path in sorted(source_files):
            aggregated = indexed_by_path.get(str(path))
            if aggregated is None:
                rel_key = str(path.relative_to(self.settings.rag_source_dir)).replace("\\", "/")
                aggregated = indexed_by_rel.get(rel_key)
            if aggregated is None:
                aggregated = {
                    "file_name": path.name,
                    "source_path": str(path),
                    "extension": path.suffix.lower(),
                    "indexed_pages": 0,
                    "indexed_chunks": 0,
                    "loaders": [],
                }

            document_group = self._document_group_for_source_path(path)
            doc_type = self._doc_type_for_source_path(path)
            indexed_documents.append(
                {
                    "file_name": aggregated["file_name"],
                    "source_path": aggregated["source_path"],
                    "extension": aggregated["extension"],
                    "indexed_pages": aggregated["indexed_pages"],
                    "indexed_chunks": aggregated["indexed_chunks"],
                    "loaders": aggregated["loaders"],
                    "doc_type": doc_type,
                    "document_group": document_group,
                }
            )

        return {
            "source_dir": str(self.settings.rag_source_dir),
            "total_files": len(source_files),
            "indexed_documents": indexed_documents,
        }

    def delete_library_document(self, source_path: Path) -> dict:
        self.index_repository.delete_document(str(source_path))
        library_state = self.list_library_documents()
        indexed_chunks = sum(document["indexed_chunks"] for document in library_state["indexed_documents"])
        return {
            "indexed_files": library_state["total_files"],
            "indexed_chunks": indexed_chunks,
            "skipped_files": 0,
        }

    def sync_unindexed_documents(self, startup_state: dict) -> None:
        library = self.list_library_documents()
        unindexed = [doc for doc in library["indexed_documents"] if doc["indexed_chunks"] == 0]
        if not unindexed:
            logger.info("모든 문서가 이미 인덱싱된 상태입니다.")
            startup_state.update(status="done", progress_pct=100)
            return

        total = len(unindexed)
        startup_state.update(
            status="indexing",
            total_files=total,
            completed_files=0,
            current_file="",
            current_stage="",
            current_chunk=0,
            total_chunks=0,
            progress_pct=0,
        )
        logger.info("%d개 미인덱싱 문서를 발견해 자동 인덱싱을 시작합니다.", total)

        for index, doc in enumerate(unindexed):
            source_path = Path(doc["source_path"])
            startup_state.update(
                current_file=doc["file_name"],
                completed_files=index,
                current_stage="prepare",
                current_chunk=0,
                total_chunks=0,
                progress_pct=int((index / max(total, 1)) * 100),
            )
            if not source_path.exists():
                logger.warning("파일이 존재하지 않아 건너뜁니다: %s", source_path)
                continue

            def progress_callback(stage, current, chunk_total, meta=None):
                file_progress = current / max(chunk_total, 1)
                overall_progress = ((index + file_progress) / max(total, 1)) * 100
                startup_state.update(
                    current_file=doc["file_name"],
                    current_stage=stage,
                    current_chunk=current,
                    total_chunks=chunk_total,
                    progress_pct=min(100, int(overall_progress)),
                )

            try:
                result = self.index_single_file(source_path, progress_callback)
                logger.info(
                    "자동 인덱싱 완료: %s (청크 %d개 / 페이지 %d개)",
                    doc["file_name"],
                    result.get("indexed_chunks", 0),
                    result.get("indexed_pages", 0),
                )
                startup_state.update(
                    completed_files=index + 1,
                    current_stage="done",
                    current_chunk=result.get("indexed_chunks", 0),
                    total_chunks=result.get("indexed_chunks", 0),
                    progress_pct=min(100, int(((index + 1) / max(total, 1)) * 100)),
                )
            except Exception:
                logger.exception("자동 인덱싱 실패: %s", doc["file_name"])

        startup_state.update(
            status="done",
            completed_files=total,
            current_file="",
            current_stage="done",
            current_chunk=0,
            total_chunks=0,
            progress_pct=100,
        )

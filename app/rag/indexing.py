from __future__ import annotations

import logging
import re
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.config import Settings
from app.storage import CacheRepository, IndexRepository
from app.rag.bge_embeddings import BGEOllamaEmbedder
from app.rag.chunking import TextChunker
from app.rag.chunking_markdown import StructuredMarkdownChunker
from app.rag.ingestion_pdf import DocumentIngestor
from app.rag.types import Document
from app.rag.utils import extracted_markdown_path, stable_hash
from app.rag.version_manager import VersionManager

logger = logging.getLogger("rag.startup")


class IndexingService:
    def __init__(
        self,
        settings: Settings,
        ingestor: DocumentIngestor,
        chunker: TextChunker,
        structured_chunker: StructuredMarkdownChunker,
        embedder: BGEOllamaEmbedder,
        index_repository: IndexRepository,
        embedding_cache_repository: CacheRepository,
    ) -> None:
        self.settings = settings
        self.ingestor = ingestor
        self.chunker = chunker
        self.structured_chunker = structured_chunker
        self.embedder = embedder
        self.index_repository = index_repository
        self.embedding_cache_repository = embedding_cache_repository
        self.version_manager = VersionManager()

    _EMBED_BATCH_SIZE = 32  # Ollama 한 번 요청에 보낼 청크 수
    _EMBED_PARALLEL_WORKERS = 2  # 동시 배치 요청 수

    def _encode_chunk(self, text: str) -> list[float]:
        return self.embedder.encode_passage(text)

    def _batch_embed_chunks(
        self,
        chunks: list,
        progress_callback=None,
        progress_stage: str = "embed",
        progress_meta_fn=None,
    ) -> tuple[list[list[float]], int, int, float]:
        """캐시를 먼저 확인 후 미스된 청크만 배치로 임베딩한다.

        Returns:
            (vectors, cache_hits, cache_misses, embed_api_total_seconds)
        """
        total = len(chunks)
        cache_keys = [stable_hash(c.text) for c in chunks]

        # 1단계: 캐시 조회
        vectors: list[list[float] | None] = [None] * total
        miss_indices: list[int] = []
        cache_hits = 0
        for i, (chunk, key) in enumerate(zip(chunks, cache_keys)):
            cached = self.embedding_cache_repository.get(key)
            if cached is not None:
                vectors[i] = cached["vector"]
                cache_hits += 1
            else:
                miss_indices.append(i)

        cache_misses = len(miss_indices)
        embed_api_total = 0.0

        # 2단계: 미스된 청크만 배치 임베딩 (병렬 요청)
        batch_size = self._EMBED_BATCH_SIZE
        batches: list[tuple[int, list[int], list[str]]] = []
        for batch_start in range(0, cache_misses, batch_size):
            batch_idx = miss_indices[batch_start: batch_start + batch_size]
            batch_texts = [chunks[i].text for i in batch_idx]
            batches.append((batch_start, batch_idx, batch_texts))

        completed_chunks = 0

        def _embed_one_batch(batch_texts: list[str], batch_id: int) -> list[list[float]]:
            total_chars = sum(len(t) for t in batch_texts)
            avg_chars = total_chars / max(len(batch_texts), 1)

            logger.info(
                "[EmbedBatch] 시작: id=%d thread=%s size=%d avg_chars=%.1f total_chars=%d",
                batch_id,
                threading.current_thread().name,
                len(batch_texts),
                avg_chars,
                total_chars,
            )

            t0 = time.perf_counter()
            result = self.embedder.encode_batch(batch_texts)
            dt = time.perf_counter() - t0

            logger.info(
                "[EmbedBatch] 완료: id=%d thread=%s size=%d avg_chars=%.1f total_chars=%d took=%.2fs",
                batch_id,
                threading.current_thread().name,
                len(batch_texts),
                avg_chars,
                total_chars,
                dt,
            )

            return result

        workers = min(self._EMBED_PARALLEL_WORKERS, len(batches)) or 1
        logger.info(
            "[EmbedConfig] batch_size=%d configured_workers=%d actual_workers=%d total_batches=%d",
            self._EMBED_BATCH_SIZE,
            self._EMBED_PARALLEL_WORKERS,
            workers,
            len(batches),
        )
        te_total = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(_embed_one_batch, batch_texts, idx): (batch_start, batch_idx)
                for idx, (batch_start, batch_idx, batch_texts) in enumerate(batches)
            }
            for future in as_completed(future_map):
                batch_start, batch_idx = future_map[future]
                batch_vectors = future.result()
                for i, vec in zip(batch_idx, batch_vectors):
                    vectors[i] = vec
                    self.embedding_cache_repository.set(cache_keys[i], {"vector": vec})
                completed_chunks += len(batch_idx)
                if progress_callback and total > 0:
                    meta = progress_meta_fn(batch_idx[-1]) if progress_meta_fn else {}
                    progress_callback(progress_stage, cache_hits + completed_chunks, total, meta)

        embed_api_total = time.perf_counter() - te_total

        logger.info(
            "[Embed] 배치 임베딩 완료: 청크 %d개 (캐시히트 %d / 미스 %d), API %.2fs, 요청 %d회 (병렬 %d), 청크당 평균 %.3fs",
            total, cache_hits, cache_misses, embed_api_total,
            len(batches), workers,
            embed_api_total / max(cache_misses, 1),
        )
        return vectors, cache_hits, cache_misses, embed_api_total  # type: ignore[return-value]

    def rebuild_index(self, source_paths: list[Path], progress_callback=None) -> dict:
        t0 = time.perf_counter()
        logger.info("[Timing][rebuild] 시작 (파일 %d개)", len(source_paths))

        documents, skipped = self.ingestor.ingest_paths(source_paths)
        t1 = time.perf_counter()
        logger.info("[Timing][rebuild] PDF 추출 완료: %.2fs, 페이지 %d개", t1 - t0, len(documents))

        chunks = self.chunk_documents(documents)
        t2 = time.perf_counter()
        logger.info("[Timing][rebuild] 청킹 완료: %.2fs, 청크 %d개", t2 - t1, len(chunks))

        for chunk in chunks:
            source_path = Path(chunk.source_path)
            version_tag = self.version_manager.detect_version_from_path(source_path)
            if version_tag:
                chunk.metadata["version_tag"] = version_tag

        def _meta(i: int) -> dict:
            return {"file_name": Path(chunks[i].source_path).name, "source_path": chunks[i].source_path}

        vectors, cache_hits, cache_misses, embed_api_total = self._batch_embed_chunks(
            chunks, progress_callback=progress_callback, progress_meta_fn=_meta
        )

        t3 = time.perf_counter()
        logger.info(
            "[Timing][rebuild] 임베딩 완료: %.2fs (API 순수 %.2fs, 캐시히트 %d / 미스 %d)",
            t3 - t2, embed_api_total, cache_hits, cache_misses,
        )

        self.index_repository.save(chunks, vectors)
        t4 = time.perf_counter()
        logger.info("[Timing][rebuild] DB 저장 완료: %.2fs", t4 - t3)
        logger.info("[Timing][rebuild] 전체 소요: %.2fs (추출 %.2fs / 청킹 %.2fs / 임베딩 %.2fs / DB %.2fs)",
                    t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3)

        return {
            "indexed_files": len({chunk.source_path for chunk in chunks}),
            "indexed_chunks": len(chunks),
            "skipped_files": len(skipped),
        }

    def index_markdown_file(self, source_path: Path, doc_type: str = "operation_manual") -> dict:
        """마크다운 파일을 직접 인덱싱한다 (PDF 없이 텍스트만 사용).

        고객사 운영 매뉴얼 등 PDF가 아닌 마크다운 문서를 RAG에 포함할 때 사용.
        """
        markdown_text = source_path.read_text(encoding="utf-8")
        if not markdown_text.strip():
            return {"indexed_chunks": 0, "indexed_pages": 0, "skipped": True}

        doc_id = stable_hash(str(source_path))
        document = Document(
            doc_id=doc_id,
            source_path=str(source_path),
            page_number=1,
            text=markdown_text,
            metadata={"loader": "markdown", "doc_type": doc_type},
        )
        chunks = self.structured_chunker.split([document], markdown_text=markdown_text)

        version_tag = self.version_manager.detect_version_from_path(source_path)
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = "structured_markdown"
            chunk.metadata["doc_type"] = doc_type
            if version_tag:
                chunk.metadata["version_tag"] = version_tag

        vectors, _, _, _ = self._batch_embed_chunks(chunks)

        self.index_repository.upsert_document(str(source_path), chunks, vectors)
        logger.info("[IndexMarkdown] %s → %d chunks", source_path.name, len(chunks))
        return {"indexed_chunks": len(chunks), "indexed_pages": 1, "skipped": False}

    def index_single_file(self, source_path: Path, progress_callback=None) -> dict:
        t0 = time.perf_counter()
        logger.info("[Timing][%s] 시작", source_path.name)

        documents, _skipped = self.ingestor.ingest_paths([source_path], progress_callback=progress_callback)
        t1 = time.perf_counter()
        logger.info("[Timing][%s] PDF 추출 완료: %.2fs, 페이지 %d개", source_path.name, t1 - t0, len(documents))

        if not documents:
            return {"indexed_chunks": 0, "indexed_pages": 0, "skipped": True}

        # generated/ 하위 파일은 고객사 메뉴얼
        generated_dir = self.settings.rag_source_dir / "generated"
        is_manual = str(source_path).startswith(str(generated_dir))
        doc_type = "operation_manual" if is_manual else "official"

        markdown_text = self.load_extracted_markdown(source_path)
        if not markdown_text and source_path.suffix.lower() == ".md":
            # 마크다운은 원본 텍스트를 마크다운으로 사용
            markdown_text = documents[0].text

        strategy = self.select_chunking_strategy(markdown_text, documents)
        chunker = self.structured_chunker if strategy == "structured_markdown" else self.chunker
        chunks = chunker.split(documents, markdown_text=markdown_text)
        t2 = time.perf_counter()
        logger.info("[Timing][%s] 청킹 완료: %.2fs (전략=%s, doc_type=%s), 청크 %d개", source_path.name, t2 - t1, strategy, doc_type, len(chunks))

        loaders = [doc.metadata.get("loader") for doc in documents if doc.metadata.get("loader")]
        representative_loader = loaders[0] if loaders else None
        version_tag = self.version_manager.detect_version_from_path(source_path)
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = strategy
            chunk.metadata["doc_type"] = doc_type
            if representative_loader and "loader" not in chunk.metadata:
                chunk.metadata["loader"] = representative_loader
            if version_tag:
                chunk.metadata["version_tag"] = version_tag

        file_name = source_path.name
        vectors, cache_hits, cache_misses, embed_api_total = self._batch_embed_chunks(
            chunks,
            progress_callback=progress_callback,
            progress_meta_fn=lambda _i: {"file_name": file_name},
        )

        t3 = time.perf_counter()
        logger.info(
            "[Timing][%s] 임베딩 완료: %.2fs (API 순수 %.2fs, 캐시히트 %d / 미스 %d)",
            source_path.name, t3 - t2, embed_api_total, cache_hits, cache_misses,
        )

        self.index_repository.upsert_document(str(source_path), chunks, vectors)
        t4 = time.perf_counter()
        logger.info("[Timing][%s] DB 저장 완료: %.2fs", source_path.name, t4 - t3)
        logger.info("[Timing][%s] 전체 소요: %.2fs (추출 %.2fs / 청킹 %.2fs / 임베딩 %.2fs / DB %.2fs)",
                    source_path.name, t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3)

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
            strategy = self.select_chunking_strategy(markdown_text, source_documents)
            chunker = self.structured_chunker if strategy == "structured_markdown" else self.chunker
            source_chunks = chunker.split(source_documents, markdown_text=markdown_text)
            loaders = [doc.metadata.get("loader") for doc in source_documents if doc.metadata.get("loader")]
            representative_loader = loaders[0] if loaders else None
            for chunk in source_chunks:
                chunk.metadata["chunking_strategy"] = strategy
                if representative_loader and "loader" not in chunk.metadata:
                    chunk.metadata["loader"] = representative_loader
            chunks.extend(source_chunks)
        return chunks

    def load_extracted_markdown(self, source_path: Path) -> str | None:
        markdown_path = extracted_markdown_path(self.settings.rag_extract_dir, source_path)
        if not markdown_path.exists():
            return None
        return markdown_path.read_text(encoding="utf-8")

    def select_chunking_strategy(self, markdown_text: str | None, documents: list) -> str:
        strategy = (self.settings.chunking_strategy or "auto").strip().lower()
        if strategy in {"page_window", "structured_markdown"}:
            return strategy

        if not markdown_text:
            return "page_window"

        page_count = max(sum(1 for line in markdown_text.splitlines() if line.startswith("## Page ")), 1)
        raw_lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
        content_lines = [
            line
            for line in raw_lines
            if not line.startswith("# ") and not line.startswith("## Page ") and not line.startswith("- loader:")
        ]
        short_lines = [line for line in content_lines if len(line) <= 60]
        bullet_lines = [line for line in content_lines if re.match(r"^[-*]\s", line)]
        avg_chars_per_page = sum(len(document.text) for document in documents) / max(page_count, 1)
        short_line_ratio = len(short_lines) / max(len(content_lines), 1)
        bullet_ratio = len(bullet_lines) / max(len(content_lines), 1)

        if avg_chars_per_page <= 420 and (short_line_ratio >= 0.55 or bullet_ratio >= 0.2):
            return "page_window"
        return "structured_markdown"

    _LIBRARY_EXTS = {".pdf", ".md"}

    def list_library_documents(self) -> dict:
        source_files = [
            path
            for path in self.settings.rag_source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self._LIBRARY_EXTS
        ]
        indexed_by_path = {document["source_path"]: document for document in self.index_repository.list_documents()}

        generated_dir = self.settings.rag_source_dir / "generated"

        indexed_documents = []
        for path in sorted(source_files):
            aggregated = indexed_by_path.get(
                str(path),
                {
                    "file_name": path.name,
                    "source_path": str(path),
                    "extension": path.suffix.lower(),
                    "indexed_pages": 0,
                    "indexed_chunks": 0,
                    "loaders": [],
                },
            )
            # generated/ 하위 파일은 고객사 메뉴얼
            is_manual = str(path).startswith(str(generated_dir))
            indexed_documents.append(
                {
                    "file_name": aggregated["file_name"],
                    "source_path": aggregated["source_path"],
                    "extension": aggregated["extension"],
                    "indexed_pages": aggregated["indexed_pages"],
                    "indexed_chunks": aggregated["indexed_chunks"],
                    "loaders": aggregated["loaders"],
                    "doc_type": "operation_manual" if is_manual else "official",
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

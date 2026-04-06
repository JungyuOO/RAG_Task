from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import re

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

    def _encode_chunk(self, text: str) -> list[float]:
        return self.embedder.encode_passage(text)

    def rebuild_index(self, source_paths: list[Path], progress_callback=None) -> dict:
        documents, skipped = self.ingestor.ingest_paths(source_paths)
        chunks = self.chunk_documents(documents)

        for chunk in chunks:
            source_path = Path(chunk.source_path)
            version_tag = self.version_manager.detect_version_from_path(source_path)
            if version_tag:
                chunk.metadata["version_tag"] = version_tag

        vectors: list[list[float]] = []
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            cache_key = stable_hash(chunk.text)
            cached = self.embedding_cache_repository.get(cache_key)
            if cached is None:
                vector = self._encode_chunk(chunk.text)
                self.embedding_cache_repository.set(cache_key, {"vector": vector})
            else:
                vector = cached["vector"]
            vectors.append(vector)
            if progress_callback and total_chunks > 0:
                progress_callback(
                    "embed",
                    i + 1,
                    total_chunks,
                    {"file_name": Path(chunk.source_path).name, "source_path": chunk.source_path},
                )

        self.index_repository.save(chunks, vectors)
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

        vectors: list[list[float]] = []
        for chunk in chunks:
            cache_key = stable_hash(chunk.text)
            cached = self.embedding_cache_repository.get(cache_key)
            if cached is None:
                vector = self._encode_chunk(chunk.text)
                self.embedding_cache_repository.set(cache_key, {"vector": vector})
            else:
                vector = cached["vector"]
            vectors.append(vector)

        self.index_repository.upsert_document(str(source_path), chunks, vectors)
        logger.info("[IndexMarkdown] %s → %d chunks", source_path.name, len(chunks))
        return {"indexed_chunks": len(chunks), "indexed_pages": 1, "skipped": False}

    def index_single_file(self, source_path: Path, progress_callback=None) -> dict:
        documents, _skipped = self.ingestor.ingest_paths([source_path], progress_callback=progress_callback)
        if not documents:
            return {"indexed_chunks": 0, "indexed_pages": 0, "skipped": True}

        markdown_text = self.load_extracted_markdown(source_path)
        strategy = self.select_chunking_strategy(markdown_text, documents)
        chunker = self.structured_chunker if strategy == "structured_markdown" else self.chunker
        chunks = chunker.split(documents, markdown_text=markdown_text)

        loaders = [doc.metadata.get("loader") for doc in documents if doc.metadata.get("loader")]
        representative_loader = loaders[0] if loaders else None
        version_tag = self.version_manager.detect_version_from_path(source_path)
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = strategy
            if representative_loader and "loader" not in chunk.metadata:
                chunk.metadata["loader"] = representative_loader
            if version_tag:
                chunk.metadata["version_tag"] = version_tag

        total_chunks = len(chunks)
        vectors: list[list[float]] = []
        for i, chunk in enumerate(chunks):
            cache_key = stable_hash(chunk.text)
            cached = self.embedding_cache_repository.get(cache_key)
            if cached is None:
                vector = self._encode_chunk(chunk.text)
                self.embedding_cache_repository.set(cache_key, {"vector": vector})
            else:
                vector = cached["vector"]
            vectors.append(vector)
            if progress_callback and total_chunks > 0:
                progress_callback("embed", i + 1, total_chunks, {"file_name": source_path.name})

        self.index_repository.upsert_document(str(source_path), chunks, vectors)
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

    def list_library_documents(self) -> dict:
        source_files = [
            path
            for path in self.settings.rag_source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
        indexed_by_path = {document["source_path"]: document for document in self.index_repository.list_documents()}

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
            indexed_documents.append(
                {
                    "file_name": aggregated["file_name"],
                    "source_path": aggregated["source_path"],
                    "extension": aggregated["extension"],
                    "indexed_pages": aggregated["indexed_pages"],
                    "indexed_chunks": aggregated["indexed_chunks"],
                    "loaders": aggregated["loaders"],
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

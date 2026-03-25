from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from app.config import Settings
from app.rag.artifacts import extracted_markdown_path
from app.rag.chunking import StructuredMarkdownChunker, TextChunker
from app.rag.embeddings import HashingEmbedder
from app.rag.ingestion import DocumentIngestor
from app.rag.utils import stable_hash
from app.repositories.cache_repository import CacheRepository
from app.repositories.index_repository import IndexRepository


class IndexingService:
    def __init__(
        self,
        settings: Settings,
        ingestor: DocumentIngestor,
        chunker: TextChunker,
        structured_chunker: StructuredMarkdownChunker,
        embedder: HashingEmbedder,
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

    def rebuild_index(self, source_paths: list[Path]) -> dict:
        documents, skipped = self.ingestor.ingest_paths(source_paths)
        chunks = self.chunk_documents(documents)

        vectors: list[list[float]] = []
        for chunk in chunks:
            cache_key = stable_hash(chunk.text)
            cached = self.embedding_cache_repository.get(cache_key)
            if cached is None:
                vector = self.embedder.encode(chunk.text)
                self.embedding_cache_repository.set(cache_key, {"vector": vector})
            else:
                vector = cached["vector"]
            vectors.append(vector)

        self.index_repository.save(chunks, vectors)
        return {
            "indexed_files": len({chunk.source_path for chunk in chunks}),
            "indexed_chunks": len(chunks),
            "skipped_files": len(skipped),
        }

    def index_single_file(self, source_path: Path, progress_callback=None) -> dict:
        """단일 파일을 인제스트·청킹·임베딩하여 인덱스에 upsert한다."""
        documents, skipped = self.ingestor.ingest_paths([source_path], progress_callback=progress_callback)
        if not documents:
            return {"indexed_chunks": 0, "indexed_pages": 0, "skipped": True}

        markdown_text = self.load_extracted_markdown(source_path)
        strategy = self.select_chunking_strategy(markdown_text, documents)
        chunker = self.structured_chunker if strategy == "structured_markdown" else self.chunker
        chunks = chunker.split(documents, markdown_text=markdown_text)

        loaders = [doc.metadata.get("loader") for doc in documents if doc.metadata.get("loader")]
        representative_loader = loaders[0] if loaders else None
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = strategy
            if representative_loader and "loader" not in chunk.metadata:
                chunk.metadata["loader"] = representative_loader

        total_chunks = len(chunks)
        vectors: list[list[float]] = []
        for i, chunk in enumerate(chunks):
            cache_key = stable_hash(chunk.text)
            cached = self.embedding_cache_repository.get(cache_key)
            if cached is None:
                vector = self.embedder.encode(chunk.text)
                self.embedding_cache_repository.set(cache_key, {"vector": vector})
            else:
                vector = cached["vector"]
            vectors.append(vector)
            if progress_callback and total_chunks > 0:
                progress_callback("embed", i + 1, total_chunks)

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

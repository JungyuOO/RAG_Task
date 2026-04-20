from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field

from apps.api.core.text import tokenize
from apps.api.api.schemas.indexing import ChunkRecord
from apps.api.rag.retrieval.legacy_pgvector_runtime import LegacyPgvectorRuntime

logger = logging.getLogger("rag.index_writer")


@dataclass(slots=True)
class LegacyIndexChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    tokens: list[str]
    chunk_order: int
    page_number: int | None
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)


class LegacyPgvectorIndexWriter:
    """Writes new-path chunks into the current pgvector-backed legacy index."""

    def __init__(self, *, runtime: LegacyPgvectorRuntime | None = None) -> None:
        self._runtime = runtime or LegacyPgvectorRuntime()

    def _get_runtime(self):
        return self._runtime.get()

    def upsert_chunks(self, source_path: str, chunks: list[ChunkRecord], *, stage_callback=None, status_callback=None) -> bool:
        if not chunks:
            return False
        runtime = self._get_runtime()
        legacy_chunks = [
            LegacyIndexChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_path=chunk.source_path,
                text=chunk.display_text,
                tokens=tokenize(chunk.retrieval_text),
                chunk_order=chunk.chunk_order,
                page_number=chunk.page_start,
                metadata={
                    **dict(chunk.metadata),
                    "section_title": chunk.section_title or "",
                    "section_path": list(chunk.section_path or []),
                    "html_anchor": chunk.html_anchor or "",
                },
            )
            for chunk in chunks
        ]
        embed_texts = [chunk.retrieval_text for chunk in chunks]
        embed_kwargs = {}
        embed_signature = inspect.signature(runtime.embedder.encode_batch)
        if "progress_callback" in embed_signature.parameters:
            embed_kwargs["progress_callback"] = lambda current, total: self._emit_progress(
                source_path=source_path,
                current=current,
                total=total,
                stage_callback=stage_callback,
                status_callback=status_callback,
            )
        vectors = runtime.embedder.encode_batch(embed_texts, **embed_kwargs)
        runtime.index_repository.upsert_document(source_path, legacy_chunks, vectors)
        message = f"[doc={source_path}] upsert completed vectors={len(vectors)}"
        logger.info(message)
        if stage_callback:
            stage_callback(message)
        if status_callback:
            status_callback(step="upserting", message=f"upsert completed vectors={len(vectors)}", current_file=source_path)
        return True

    @staticmethod
    def _emit_progress(*, source_path: str, current: int, total: int, stage_callback=None, status_callback=None) -> None:
        message = f"[doc={source_path}] embed progress batch={current}/{total}"
        logger.info(message)
        if stage_callback:
            stage_callback(message)
        if status_callback:
            status_callback(
                step="EMBEDDING",
                message=f"{current}/{total} batches processed",
                current_file=source_path,
            )


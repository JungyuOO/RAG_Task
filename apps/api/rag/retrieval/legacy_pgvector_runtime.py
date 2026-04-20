from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.api.core.pgvector_settings import PgvectorRuntimeSettings, get_pgvector_runtime_settings
from apps.api.rag.retrieval.embedding_clients import BGETEIEmbedder, BGEOllamaEmbedder
from apps.api.rag.retrieval.pgvector_store import PgvectorIndex, PgvectorIndexRepository


@dataclass(slots=True)
class LegacyPgvectorRuntimeDeps:
    settings: PgvectorRuntimeSettings
    embedder: Any
    index_repository: PgvectorIndexRepository


class LegacyPgvectorRuntime:
    """Minimal legacy pgvector runtime without the full legacy app container."""

    def __init__(self, *, settings: PgvectorRuntimeSettings | None = None) -> None:
        self._settings = settings
        self._deps: LegacyPgvectorRuntimeDeps | None = None
        self._asymmetric_enabled = True

    def get(self) -> LegacyPgvectorRuntimeDeps:
        if self._deps is None:
            settings = self._settings or get_pgvector_runtime_settings()
            if settings.embedding_backend == "tei":
                embedder = BGETEIEmbedder(
                    base_url=settings.tei_base_url,
                    model=settings.tei_embedding_model,
                    timeout=settings.tei_timeout,
                    batch_size=settings.embedding_batch_size,
                    batch_char_limit=settings.embedding_batch_char_limit,
                    parallel_workers=settings.embedding_parallel_workers,
                )
            else:
                embedder = BGEOllamaEmbedder(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_embedding_model,
                    timeout=settings.ollama_timeout,
                    batch_size=settings.embedding_batch_size,
                    batch_char_limit=settings.embedding_batch_char_limit,
                    parallel_workers=settings.embedding_parallel_workers,
                )
            index = PgvectorIndex(settings.db_dsn)
            index_repository = PgvectorIndexRepository(index, rag_source_dir=settings.rag_source_dir)
            self._deps = LegacyPgvectorRuntimeDeps(
                settings=settings,
                embedder=embedder,
                index_repository=index_repository,
            )
        return self._deps

    def set_asymmetric_enabled(self, enabled: bool) -> None:
        self._asymmetric_enabled = bool(enabled)



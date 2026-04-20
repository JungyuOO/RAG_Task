from __future__ import annotations

import shutil
from pathlib import Path

from apps.api.api.schemas.indexing import IndexResetResponse
from apps.api.rag.retrieval.legacy_pgvector_runtime import LegacyPgvectorRuntime
from apps.api.rag.indexing.batch_job_service import BatchIndexJobService


class IndexAdminService:
    def __init__(
        self,
        *,
        runtime: LegacyPgvectorRuntime | None = None,
        batch_job_service: BatchIndexJobService | None = None,
        root_dir: Path | None = None,
    ) -> None:
        self.runtime = runtime or LegacyPgvectorRuntime()
        self.batch_job_service = batch_job_service
        self.root_dir = root_dir or Path.cwd() / "data"

    def reset_all(self, *, preserve_embedding_cache: bool = True) -> IndexResetResponse:
        runtime = self.runtime.get()
        runtime.index_repository.backend.clear_index()

        cleared_files = {
            "cache_answers": self._clear_dir(self.root_dir / "cache" / "answers"),
            "cache_embeddings": 0 if preserve_embedding_cache else self._clear_dir(self.root_dir / "cache" / "embeddings"),
            "extracted_markdown": self._clear_dir(self.root_dir / "extracted_markdown"),
            "normalized_documents": self._clear_dir(self.root_dir / "normalized" / "documents"),
            "normalized_blocks": self._clear_dir(self.root_dir / "normalized" / "blocks"),
            "normalized_chunks": self._clear_dir(self.root_dir / "normalized" / "chunks"),
        }
        if self.batch_job_service is not None:
            self.batch_job_service.job_repository.clear()
        return IndexResetResponse(
            cleared_tables=["documents", "chunks"],
            cleared_files=cleared_files,
        )

    @staticmethod
    def _clear_dir(path: Path) -> int:
        if not path.exists():
            return 0
        removed = 0
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
            else:
                child.unlink(missing_ok=True)
                removed += 1
        return removed

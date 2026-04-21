from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from apps.api.schemas.library import LibrarySourceBreakdownItem, LibrarySummaryResponse
from apps.api.rag.indexing.batch_job_service import BatchIndexJobService
from apps.api.rag.retrieval.legacy_pgvector_runtime import LegacyPgvectorRuntime


def _resolve_runtime_path(raw: str, *, fallback: Path) -> Path:
    candidate = Path(raw).expanduser() if raw else fallback
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _list_real_files(root: Path, *, excluded_names: set[str] | None = None) -> list[Path]:
    if not root.exists():
        return []
    excluded_names = excluded_names or set()
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded_names and not path.name.startswith(".")
    ]


def _relative_display(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return path_value


class LibrarySummaryService:
    def __init__(
        self,
        *,
        runtime: LegacyPgvectorRuntime | None = None,
        batch_job_service: BatchIndexJobService | None = None,
        extract_root: Path | None = None,
    ) -> None:
        self._runtime = runtime or LegacyPgvectorRuntime()
        self._batch_job_service = batch_job_service
        self._extract_root = extract_root

    def get_summary(self, *, workspace_id: str = "") -> LibrarySummaryResponse:
        runtime = self._runtime.get()
        source_root = Path(runtime.settings.rag_source_dir).resolve()
        extract_root = (
            self._extract_root.resolve()
            if self._extract_root is not None
            else _resolve_runtime_path(
                str(os.environ.get("RAG_EXTRACT_DIR") or ""),
                fallback=Path.cwd() / "data" / "extracted_markdown",
            )
        )

        corpus_files = _list_real_files(source_root, excluded_names={"manifest.json", ".gitkeep"})
        corpus_files = [path for path in corpus_files if not self._is_legacy_file(path, source_root)]
        extracted_files = _list_real_files(extract_root, excluded_names={".gitkeep"})
        manifest_entries = self._count_manifest_entries(source_root / "manifest.json")

        source_breakdown: dict[str, int] = {}
        for file_path in corpus_files:
            key = file_path.suffix.lower().lstrip(".") or "no_ext"
            source_breakdown[key] = source_breakdown.get(key, 0) + 1

        try:
            index_stats = runtime.index_repository.get_index_stats()
            indexed_paths = sorted(runtime.index_repository.get_indexed_source_paths())
            index_error = ""
        except Exception as exc:
            index_stats = {"documents": 0, "chunks": 0}
            indexed_paths = []
            index_error = str(exc)

        jobs: list[Any] = []
        if self._batch_job_service is not None:
            jobs = list(getattr(self._batch_job_service.list_recent(limit=20), "jobs", []))

        indexed_documents = int(index_stats.get("documents") or 0)
        indexed_chunks = int(index_stats.get("chunks") or 0)
        batch_jobs = len(jobs)
        latest_batch_status = str(getattr(jobs[0], "status", "")) if jobs else ""

        if indexed_documents > 0 and batch_jobs == 0:
            message = (
                "The pgvector index already contains documents, but the current runtime does not have recent batch job history. "
                "Library can still show indexed state from the stored index."
            )
        elif indexed_documents > 0:
            message = "Indexed documents and recent batch reindex history are available."
        elif corpus_files:
            message = "Source documents exist, but no visible pgvector index state was found yet. Reindexing is still required."
        elif index_error:
            message = f"Source documents were checked, but index state could not be loaded: {index_error}"
        else:
            message = "No source documents or index state are available yet."

        return LibrarySummaryResponse(
            workspace_id=workspace_id,
            source_root=str(source_root),
            extract_root=str(extract_root),
            corpus_files=len(corpus_files),
            manifest_entries=manifest_entries,
            extracted_artifacts=len(extracted_files),
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            batch_jobs=batch_jobs,
            latest_batch_status=latest_batch_status,
            source_breakdown=[
                LibrarySourceBreakdownItem(label=label, count=count)
                for label, count in sorted(source_breakdown.items(), key=lambda item: (-item[1], item[0]))
            ],
            indexed_samples=[_relative_display(path_value) for path_value in indexed_paths[:6]],
            message=message,
        )

    @staticmethod
    def _count_manifest_entries(manifest_path: Path) -> int:
        if not manifest_path.exists():
            return 0
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            return len(payload)
        return 0

    @staticmethod
    def _is_legacy_file(path: Path, source_root: Path) -> bool:
        try:
            relative_path = path.relative_to(source_root).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        return relative_path.startswith("legacy/")


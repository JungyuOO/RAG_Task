from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apps.api.rag.library.summary_service import LibrarySummaryService


class _FakeIndexRepository:
    def get_index_stats(self) -> dict[str, int]:
        return {"documents": 26, "chunks": 7581}

    def get_indexed_source_paths(self) -> set[str]:
        return {
            "/workspace/data/corpus/pdfs/customer/demo_customer_manual.md",
            "/workspace/data/corpus/pdfs/official/en/authentication_and_authorization.md",
        }


class _FakeRuntime:
    def __init__(self, source_root: Path) -> None:
        self._deps = SimpleNamespace(
            settings=SimpleNamespace(rag_source_dir=source_root),
            index_repository=_FakeIndexRepository(),
        )

    def get(self):
        return self._deps


class _FakeBatchJobService:
    def __init__(self, jobs) -> None:
        self._jobs = list(jobs)

    def list_recent(self, limit: int = 20):
        return SimpleNamespace(jobs=self._jobs[:limit])


class LibrarySummaryServiceTests(unittest.TestCase):
    def test_summary_reports_existing_index_even_without_batch_history(self) -> None:
        source_root = Path("/virtual/data/corpus/pdfs")
        extract_root = Path("/virtual/data/extracted_markdown")
        corpus_files = [
            source_root / "customer" / "demo_customer_manual.md",
            source_root / "customer" / "sample.pdf",
        ]
        extracted_files = [extract_root / "demo_customer_manual.json"]

        service = LibrarySummaryService(
            runtime=_FakeRuntime(source_root),  # type: ignore[arg-type]
            batch_job_service=_FakeBatchJobService([]),  # type: ignore[arg-type]
            extract_root=extract_root,
        )

        with patch("apps.api.rag.library.summary_service._list_real_files") as list_real_files:
            list_real_files.side_effect = [corpus_files, extracted_files]
            with patch.object(LibrarySummaryService, "_count_manifest_entries", return_value=1):
                summary = service.get_summary()

        self.assertEqual(summary.corpus_files, 2)
        self.assertEqual(summary.manifest_entries, 1)
        self.assertEqual(summary.extracted_artifacts, 1)
        self.assertEqual(summary.indexed_documents, 26)
        self.assertEqual(summary.indexed_chunks, 7581)
        self.assertEqual(summary.batch_jobs, 0)
        self.assertIn("pgvector index already contains documents", summary.message)
        self.assertEqual(summary.source_breakdown[0].label, "md")
        self.assertEqual(summary.source_breakdown[0].count, 1)


if __name__ == "__main__":
    unittest.main()

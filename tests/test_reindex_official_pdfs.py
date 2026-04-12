from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from scripts.reindex_official_pdfs import (
    collect_official_pdfs,
    delete_extracted_artifacts,
    extracted_artifact_candidates,
    purge_legacy_index_entries,
    source_path_key,
)


class ReindexOfficialPdfsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_reindex_official_pdfs")
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "source" / "ocp-4.20").mkdir(parents=True, exist_ok=True)
        (self.root / "extract").mkdir(parents=True, exist_ok=True)

        self.pdf_a = self.root / "source" / "ocp-4.20" / "guide-a.pdf"
        self.pdf_b = self.root / "source" / "ocp-4.20" / "guide-b.pdf"
        self.pdf_a.write_bytes(b"%PDF-1.4\n")
        self.pdf_b.write_bytes(b"%PDF-1.4\n")

        for name in (
            "guide-a-12345678.md",
            "guide-a-12345678.html",
            "guide-a-12345678.json",
            "guide-b-87654321.md",
        ):
            (self.root / "extract" / name).write_text("artifact", encoding="utf-8")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def test_collect_official_pdfs_reads_version_dirs(self) -> None:
        files = collect_official_pdfs(self.root / "source", ["4.20"])
        self.assertEqual(files, [self.pdf_a, self.pdf_b])

    def test_extracted_artifact_candidates_include_globbed_files(self) -> None:
        candidates = extracted_artifact_candidates(self.root / "extract", self.pdf_a)
        names = {path.name for path in candidates}
        self.assertIn("guide-a-12345678.md", names)
        self.assertIn("guide-a-12345678.html", names)
        self.assertIn("guide-a-12345678.json", names)

    def test_delete_extracted_artifacts_removes_targeted_files_only(self) -> None:
        deleted = delete_extracted_artifacts(self.root / "extract", [self.pdf_a])
        self.assertEqual(deleted, 3)
        self.assertFalse((self.root / "extract" / "guide-a-12345678.md").exists())
        self.assertFalse((self.root / "extract" / "guide-a-12345678.html").exists())
        self.assertFalse((self.root / "extract" / "guide-a-12345678.json").exists())
        self.assertTrue((self.root / "extract" / "guide-b-87654321.md").exists())

    def test_source_path_key_normalizes_runtime_variants(self) -> None:
        source_root = self.root / "source"
        canonical = source_path_key(self.pdf_a, source_root)
        app_runtime = source_path_key("/app/data/corpus/pdfs/ocp-4.20/guide-a.pdf", source_root)
        relative_runtime = source_path_key(r"data\corpus\pdfs\ocp-4.20\guide-a.pdf", source_root)
        self.assertEqual(canonical, "ocp-4.20/guide-a.pdf")
        self.assertEqual(app_runtime, canonical)
        self.assertEqual(relative_runtime, canonical)

    def test_purge_legacy_index_entries_removes_same_document_with_old_source_paths(self) -> None:
        self_target = self.pdf_a
        other_target = self.root / "source" / "ocp-4.20" / "guide-b.pdf"

        class _Repo:
            def __init__(self) -> None:
                self.deleted: list[str] = []

            def list_documents(self):
                return [
                    {"source_path": "/app/data/corpus/pdfs/ocp-4.20/guide-a.pdf"},
                    {"source_path": r"data\corpus\pdfs\ocp-4.20\guide-a.pdf"},
                    {"source_path": str(self_target)},
                    {"source_path": str(other_target)},
                ]

            def delete_document(self, source_path: str):
                self.deleted.append(source_path)

        repo = _Repo()
        deleted = purge_legacy_index_entries(repo, self.root / "source", self_target)

        self.assertEqual(deleted, 2)
        self.assertEqual(
            repo.deleted,
            [
                "/app/data/corpus/pdfs/ocp-4.20/guide-a.pdf",
                r"data\corpus\pdfs\ocp-4.20\guide-a.pdf",
            ],
        )


if __name__ == "__main__":
    unittest.main()

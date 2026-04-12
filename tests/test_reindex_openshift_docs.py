from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from scripts.reindex_openshift_docs import collect_markdown_sources, delete_missing_documents, markdown_root


class _IndexRepositoryStub:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs
        self.deleted: list[str] = []

    def list_documents(self) -> list[dict]:
        return list(self._docs)

    def delete_document(self, source_path: str) -> None:
        self.deleted.append(source_path)


class ReindexOpenshiftDocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_reindex_openshift_docs")
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "ocp-4.20-openshift-docs" / "installing").mkdir(parents=True, exist_ok=True)
        (self.root / "ocp-4.20-openshift-docs" / "networking").mkdir(parents=True, exist_ok=True)
        self.doc_a = self.root / "ocp-4.20-openshift-docs" / "installing" / "guide-a.md"
        self.doc_b = self.root / "ocp-4.20-openshift-docs" / "networking" / "guide-b.md"
        self.doc_a.write_text("# A\n", encoding="utf-8")
        self.doc_b.write_text("# B\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def test_markdown_root_uses_versioned_directory(self) -> None:
        self.assertEqual(markdown_root(self.root, "4.20"), self.root / "ocp-4.20-openshift-docs")

    def test_collect_markdown_sources_defaults_to_all(self) -> None:
        files = collect_markdown_sources(self.root / "ocp-4.20-openshift-docs", [])
        self.assertEqual(files, [self.doc_a.resolve(), self.doc_b.resolve()])

    def test_collect_markdown_sources_accepts_subpath(self) -> None:
        files = collect_markdown_sources(self.root / "ocp-4.20-openshift-docs", ["installing"])
        self.assertEqual(files, [self.doc_a.resolve()])

    def test_delete_missing_documents_removes_only_stale_docs_under_root(self) -> None:
        missing = self.root / "ocp-4.20-openshift-docs" / "installing" / "gone.md"
        repo = _IndexRepositoryStub(
            [
                {"source_path": str(self.doc_a)},
                {"source_path": str(missing)},
                {"source_path": str(self.root / "other-root" / "leave.md")},
            ]
        )
        deleted = delete_missing_documents(repo, self.root / "ocp-4.20-openshift-docs")
        self.assertEqual(deleted, 1)
        self.assertEqual(repo.deleted, [str(missing)])


if __name__ == "__main__":
    unittest.main()

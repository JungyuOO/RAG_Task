from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from apps.api.rag.library.document_service import LibraryDocumentService


class _FakeIndexRepository:
    def get_document_chunk_counts(self):
        return [
            {"source_path": "/virtual/data/corpus/pdfs/official/en/authentication_and_authorization.md", "chunk_count": 12},
            {"source_path": "/virtual/data/corpus/pdfs/customer/demo_customer_manual.md", "chunk_count": 4},
        ]

    def list_chunks(self):
        return [
            {
                "source_path": "/virtual/data/corpus/pdfs/customer/demo_customer_manual.md",
                "chunk_id": "chunk-1",
                "text": "customer preview text",
                "page_number": 1,
                "metadata": {"section_title": "Overview", "block_types": ["paragraph"]},
            }
        ]


class _FakeRuntime:
    def __init__(self, source_root: Path) -> None:
        self._deps = SimpleNamespace(
            settings=SimpleNamespace(rag_source_dir=source_root),
            index_repository=_FakeIndexRepository(),
        )

    def get(self):
        return self._deps


class LibraryDocumentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp/library-document-service").resolve()
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "official" / "en").mkdir(parents=True, exist_ok=True)
        (self.root / "customer").mkdir(parents=True, exist_ok=True)
        (self.root / "customer_pdf").mkdir(parents=True, exist_ok=True)
        (self.root / "official" / "en" / "authentication_and_authorization.md").write_text(
            "# Authentication and Authorization\n\ncontent",
            encoding="utf-8",
        )
        (self.root / "customer" / "demo_customer_manual.md").write_text(
            "# Demo Customer Manual\n\nbody",
            encoding="utf-8",
        )
        (self.root / "customer_pdf" / "demo_customer_manual.pdf").write_bytes(b"%PDF-1.4")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_catalog_separates_official_and_customer_documents(self) -> None:
        service = LibraryDocumentService(runtime=_FakeRuntime(self.root))  # type: ignore[arg-type]

        catalog = service.get_catalog()

        self.assertEqual(len(catalog.official_documents), 1)
        self.assertEqual(len(catalog.customer_documents), 1)
        official = catalog.official_documents[0]
        self.assertTrue(official.indexed)
        self.assertEqual(official.chunk_count, 12)
        self.assertEqual(official.original_kind, "markdown")

        customer_md = catalog.customer_documents[0]
        self.assertEqual(customer_md.original_kind, "pdf")
        self.assertEqual(customer_md.original_key, "customer_pdf/demo_customer_manual.pdf")

    def test_get_chunks_and_markdown_content(self) -> None:
        service = LibraryDocumentService(runtime=_FakeRuntime(self.root))  # type: ignore[arg-type]

        chunks = service.get_chunks("customer/demo_customer_manual.md")
        content = service.get_markdown_content("official/en/authentication_and_authorization.md")

        self.assertEqual(chunks.chunk_count, 1)
        self.assertEqual(chunks.chunks[0].section_title, "Overview")
        self.assertIn("Authentication and Authorization", content.content)


if __name__ == "__main__":
    unittest.main()

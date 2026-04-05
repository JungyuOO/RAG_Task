from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.rag.ingestion_pdf import DocumentIngestor, PdfExtractor
from app.rag.types import Document


class Phase0IngestionTests(unittest.TestCase):
    def test_document_ingestor_keeps_text_loader_path(self) -> None:
        settings = MagicMock()
        ingestor = DocumentIngestor(settings)
        tmp_dir = Path("tests/.tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / "sample.txt"
        try:
            path.write_text("sample text", encoding="utf-8")
            documents, skipped = ingestor.ingest_paths([path])
        finally:
            if path.exists():
                path.unlink()

        self.assertEqual(len(skipped), 0)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["loader"], "text")
        self.assertEqual(documents[0].text, "sample text")

    def test_pdf_extractor_detects_yaml_code_block(self) -> None:
        extractor = PdfExtractor(MagicMock())
        text = (
            "설명 텍스트\n"
            "apiVersion: route.openshift.io/v1\n"
            "kind: Route\n"
            "metadata:\n"
            "  name: route-edge\n"
            "spec:\n"
            "  tls:\n"
            "    termination: edge\n"
        )

        result = extractor._detect_code_blocks(text)

        self.assertIn("```yaml", result)
        self.assertIn("kind: Route", result)

    def test_pdf_extractor_merges_cross_page_yaml_blocks(self) -> None:
        extractor = PdfExtractor(MagicMock())
        documents = [
            Document(
                doc_id="doc-2",
                source_path="networking.pdf",
                page_number=2,
                text="LoadBalancer 설명\nExternalName 설명\n# demo-svc.yaml\napiVersion: v1",
                metadata={"file_name": "networking.pdf", "loader": "pdf_text"},
            ),
            Document(
                doc_id="doc-3",
                source_path="networking.pdf",
                page_number=3,
                text="```yaml\nkind: Service\nmetadata:\n  name: demo-svc\nspec:\n  ports:\n  - port: 80\n```",
                metadata={"file_name": "networking.pdf", "loader": "pdf_text"},
            ),
        ]
        markdown_sections = [
            {
                "page_number": 2,
                "loader": "pdf_text",
                "chars": len(documents[0].text),
                "text": "LoadBalancer 설명\nExternalName 설명\n# demo-svc.yaml\napiVersion: v1",
            },
            {
                "page_number": 3,
                "loader": "pdf_text",
                "chars": len(documents[1].text),
                "text": "```yaml\nkind: Service\nmetadata:\n  name: demo-svc\nspec:\n  ports:\n  - port: 80\n```",
            },
        ]

        extractor._merge_cross_page_yaml_blocks(documents, markdown_sections)

        merged_text = markdown_sections[0]["text"]
        self.assertIn("```yaml", merged_text)
        self.assertIn("apiVersion: v1", merged_text)
        self.assertIn("kind: Service", merged_text)
        self.assertEqual(markdown_sections[1]["text"], "")
        self.assertIn("kind: Service", documents[0].text)
        self.assertEqual(documents[1].text, "")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.rag.ingestion_pdf import PdfExtractor
from app.rag.types import Document
from app.rag.utils import (
    extracted_html_candidates,
    extracted_html_path,
    extracted_markdown_path,
    extracted_metadata_path,
)


class ExtractedArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path("tests/.tmp_extracted_artifacts")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_extracted_artifact_paths_share_stem_hash(self) -> None:
        extract_dir = Path("/tmp/extracted")
        source_path = Path("/data/docs/sample.pdf")

        markdown_path = extracted_markdown_path(extract_dir, source_path)
        html_path = extracted_html_path(extract_dir, source_path)
        metadata_path = extracted_metadata_path(extract_dir, source_path)

        self.assertEqual(markdown_path.stem, html_path.stem)
        self.assertEqual(markdown_path.stem, metadata_path.stem)
        self.assertEqual(markdown_path.suffix, ".md")
        self.assertEqual(html_path.suffix, ".html")
        self.assertEqual(metadata_path.suffix, ".json")

    def test_pdf_export_writes_markdown_html_and_metadata(self) -> None:
        settings = SimpleNamespace(
            rag_extract_dir=self.tmp_root,
            save_extracted_markdown=True,
            save_extracted_html=True,
            save_extracted_metadata=True,
        )
        extractor = PdfExtractor(settings)
        source_path = Path("/docs/guide.pdf")
        documents = [
            Document(
                doc_id="doc-1",
                source_path=str(source_path),
                page_number=1,
                text="example text",
                metadata={"loader": "pdf_text"},
            )
        ]
        sections = [
            {
                "page_number": 1,
                "loader": "pdf_text",
                "chars": 80,
                "text": "# Install prerequisites\n\n- verify DNS\n- verify network\n\n```yaml\nkind: Pod\nmetadata:\n  name: demo\n```",
            }
        ]

        extractor.export_artifacts(source_path, documents, sections)

        markdown_text = extracted_markdown_path(self.tmp_root, source_path).read_text(encoding="utf-8")
        html_text = extracted_html_path(self.tmp_root, source_path).read_text(encoding="utf-8")
        metadata = json.loads(extracted_metadata_path(self.tmp_root, source_path).read_text(encoding="utf-8"))

        self.assertIn("# guide.pdf", markdown_text)
        self.assertIn("Page 1", markdown_text)
        self.assertIn("<section id=\"page-1\" class=\"page-block\"", html_text)
        self.assertIn("<article id=\"page-1-block-1\"", html_text)
        self.assertIn("block-badge", html_text)
        self.assertIn("kind: Pod", html_text)
        self.assertEqual(metadata["file_name"], "guide.pdf")
        self.assertEqual(metadata["extracted_pages"], 1)
        self.assertEqual(metadata["pages"][0]["page_number"], 1)
        self.assertEqual(metadata["pages"][0]["html_anchor"], "page-1")
        self.assertIn("code", metadata["pages"][0]["block_types"])
        self.assertIn("heading", metadata["pages"][0]["block_types"])
        self.assertIn("list", metadata["pages"][0]["block_types"])
        self.assertFalse(metadata["pages"][0]["is_toc"])
        self.assertEqual(metadata["pages"][0]["section_title"], "Install prerequisites")
        self.assertEqual(metadata["pages"][0]["section_path"], "Install prerequisites")
        self.assertEqual(metadata["pages"][0]["blocks"][0]["block_id"], "page-1-block-1")
        self.assertEqual(metadata["pages"][0]["blocks"][0]["block_type"], "heading")
        self.assertEqual(metadata["pages"][0]["blocks"][0]["attributes"]["title"], "Install prerequisites")
        self.assertEqual(metadata["pages"][0]["blocks"][1]["block_type"], "list")
        self.assertEqual(metadata["pages"][0]["blocks"][1]["attributes"]["item_count"], 2)
        self.assertEqual(metadata["pages"][0]["blocks"][2]["block_type"], "code")
        self.assertEqual(metadata["pages"][0]["blocks"][2]["attributes"]["language"], "yaml")
        self.assertEqual(metadata["pages"][0]["blocks"][2]["attributes"]["resource_kind"], "Pod")

    def test_front_matter_helpers_detect_cover_legal_and_toc_pages(self) -> None:
        settings = SimpleNamespace(
            rag_extract_dir=self.tmp_root,
            save_extracted_markdown=True,
            save_extracted_html=True,
            save_extracted_metadata=True,
        )
        extractor = PdfExtractor(settings)

        cover_text = "\n".join(
            [
                "OpenShift Container Platform 4.20",
                "Architecture",
                "An overview of the architecture for OpenShift Container Platform",
                "Last Updated: 2026-03-18",
            ]
        )
        legal_text = "\n".join(
            [
                "Legal Notice",
                "Copyright Red Hat.",
                "Creative Commons Attribution-Share Alike 3.0",
                "All other trademarks are the property of their respective owners.",
            ]
        )
        toc_text = "\n".join(
            [
                "Table of Contents",
                "CHAPTER 1. ARCHITECTURE OVERVIEW",
                "11",
                "1.1. About Kubernetes",
                "12",
                ". . . . . . . . . . . . . . . . .",
            ]
        )

        self.assertTrue(extractor._should_drop_front_matter_page(cover_text, 1))
        self.assertTrue(extractor._should_drop_front_matter_page(legal_text, 4))
        self.assertTrue(extractor._should_drop_front_matter_page(toc_text, 5))
        self.assertFalse(extractor._should_drop_front_matter_page("1. About networking\nActual content starts here.", 20))

    def test_extracted_html_candidates_find_same_stem_hashes(self) -> None:
        source_path = Path("/docs/guide.pdf")
        expected = self.tmp_root / "guide-12345678.html"
        expected.write_text("<html></html>", encoding="utf-8")

        candidates = extracted_html_candidates(self.tmp_root, source_path)

        self.assertIn(expected, candidates)


if __name__ == "__main__":
    unittest.main()

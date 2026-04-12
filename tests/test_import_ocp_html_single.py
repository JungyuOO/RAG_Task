from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from app.rag.utils import extracted_html_path, extracted_metadata_path
from scripts.import_ocp_html_single import import_ocp_html_single


class ImportOcpHtmlSingleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_import_ocp_html_single")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_html = self.root / "advanced_networking.html"
        self.raw_meta = self.root / "advanced_networking.meta.json"
        self.output_root = self.root / "corpus"
        self.extract_root = self.root / "extract"
        self.raw_html.write_text(
            """
            <html><body><main>
              <article>
                <h1 id="advanced-networking">Advanced networking</h1>
                <p>Advanced networking topics overview.</p>
                <h2 id="verifying-connectivity-endpoint">Verifying connectivity endpoint</h2>
                <p>Cluster Network Operator runs connectivity checks.</p>
                <ul><li>Kubernetes API server service</li><li>OpenShift API server endpoint</li></ul>
              </article>
            </main></body></html>
            """,
            encoding="utf-8",
        )
        self.raw_meta.write_text(
            json.dumps(
                {
                    "book_slug": "advanced_networking",
                    "book_title": "Advanced networking",
                    "ocp_version": "4.20",
                    "docs_language": "en",
                    "resolved_language": "en",
                    "source_url": "https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html-single/advanced_networking/index",
                    "resolved_source_url": "https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html-single/advanced_networking/index",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_import_generates_markdown_and_extracted_artifacts(self) -> None:
        result = import_ocp_html_single(self.raw_html, self.raw_meta, self.output_root, self.extract_root)

        source_path = Path(result["source_path"])
        self.assertTrue(source_path.exists())
        markdown = source_path.read_text(encoding="utf-8")
        self.assertIn("## Page 1", markdown)
        self.assertIn("# Advanced networking", markdown)
        self.assertIn("## Page 2", markdown)
        self.assertIn("## Verifying connectivity endpoint", markdown)

        html_artifact = extracted_html_path(self.extract_root, source_path)
        metadata_artifact = extracted_metadata_path(self.extract_root, source_path)
        self.assertTrue(html_artifact.exists())
        self.assertTrue(metadata_artifact.exists())

        metadata = json.loads(metadata_artifact.read_text(encoding="utf-8"))
        self.assertEqual(metadata["book_slug"], "advanced_networking")
        self.assertEqual(metadata["locale"], "en")
        self.assertEqual(metadata["pages"][0]["html_anchor"], "advanced-networking")
        self.assertEqual(metadata["pages"][1]["section_title"], "Verifying connectivity endpoint")


if __name__ == "__main__":
    unittest.main()

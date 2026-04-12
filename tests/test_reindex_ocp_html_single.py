from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from scripts.reindex_ocp_html_single import collect_markdown_sources


class ReindexOcpHtmlSingleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_reindex_ocp_html_single")
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "ocp-html-single-4.20-en").mkdir(parents=True, exist_ok=True)
        (self.root / "ocp-html-single-4.20-en" / "advanced_networking.md").write_text("# doc", encoding="utf-8")
        (self.root / "ocp-html-single-4.20-en" / "networking_overview.md").write_text("# doc", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_collect_markdown_sources_all(self) -> None:
        files = collect_markdown_sources(self.root, [])
        self.assertEqual(len(files), 2)

    def test_collect_markdown_sources_selected_path(self) -> None:
        files = collect_markdown_sources(self.root, ["ocp-html-single-4.20-en/advanced_networking.md"])
        self.assertEqual([path.name for path in files], ["advanced_networking.md"])


if __name__ == "__main__":
    unittest.main()

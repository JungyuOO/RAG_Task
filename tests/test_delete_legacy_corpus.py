from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from scripts.delete_legacy_corpus import collect_legacy_sources, is_legacy_official_source, is_legacy_official_source_ref


class DeleteLegacyCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_delete_legacy_corpus")
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "ocp-4.20").mkdir(parents=True, exist_ok=True)
        (self.root / "ocp-html-single-4.20-en").mkdir(parents=True, exist_ok=True)
        (self.root / "generated").mkdir(parents=True, exist_ok=True)
        (self.root / "ocp-4.20" / "legacy.pdf").write_bytes(b"%PDF")
        (self.root / "ocp-4.20-openshift-docs" / "guide.md").parent.mkdir(parents=True, exist_ok=True)
        (self.root / "ocp-4.20-openshift-docs" / "guide.md").write_text("# legacy", encoding="utf-8")
        (self.root / "ocp-html-single-4.20-en" / "advanced_networking.md").write_text("# new", encoding="utf-8")
        (self.root / "generated" / "customer-guide.md").write_text("# customer", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_is_legacy_official_source(self) -> None:
        self.assertTrue(is_legacy_official_source(self.root / "ocp-4.20" / "legacy.pdf"))
        self.assertTrue(is_legacy_official_source(self.root / "ocp-4.20-openshift-docs" / "guide.md"))
        self.assertFalse(is_legacy_official_source(self.root / "ocp-html-single-4.20-en" / "advanced_networking.md"))
        self.assertFalse(is_legacy_official_source(self.root / "generated" / "customer-guide.md"))

    def test_collect_legacy_sources(self) -> None:
        sources = collect_legacy_sources(self.root)
        names = sorted(path.name for path in sources)
        self.assertEqual(names, ["guide.md", "legacy.pdf"])

    def test_is_legacy_official_source_ref_handles_nonexistent_windows_paths(self) -> None:
        self.assertTrue(is_legacy_official_source_ref(r"C:\workspace\data\corpus\pdfs\ocp-4.20\guide.pdf"))
        self.assertTrue(is_legacy_official_source_ref(r"C:\workspace\data\corpus\pdfs\ocp-4.20-openshift-docs\networking\guide.md"))
        self.assertFalse(is_legacy_official_source_ref(r"C:\workspace\data\corpus\pdfs\ocp-html-single-4.20-en\advanced_networking.md"))
        self.assertFalse(is_legacy_official_source_ref(r"C:\workspace\data\corpus\pdfs\generated\customer-guide.md"))


if __name__ == "__main__":
    unittest.main()

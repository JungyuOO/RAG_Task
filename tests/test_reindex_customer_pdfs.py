from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from scripts.reindex_customer_pdfs import collect_customer_pdfs


class ReindexCustomerPdfsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_reindex_customer_pdfs")
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "generated_pdf" / "team-a").mkdir(parents=True, exist_ok=True)
        (self.root / "generated_pdf" / "team-b").mkdir(parents=True, exist_ok=True)

        self.pdf_a = self.root / "generated_pdf" / "team-a" / "guide-a.pdf"
        self.pdf_b = self.root / "generated_pdf" / "team-a" / "guide-b.pdf"
        self.pdf_c = self.root / "generated_pdf" / "team-b" / "guide-c.pdf"
        self.pdf_a.write_bytes(b"%PDF-1.4\n")
        self.pdf_b.write_bytes(b"%PDF-1.4\n")
        self.pdf_c.write_bytes(b"%PDF-1.4\n")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def test_collect_customer_pdfs_defaults_to_all(self) -> None:
        files = collect_customer_pdfs(self.root / "generated_pdf", [])
        self.assertEqual(files, [self.pdf_a.resolve(), self.pdf_b.resolve(), self.pdf_c.resolve()])

    def test_collect_customer_pdfs_accepts_subdirectory_target(self) -> None:
        files = collect_customer_pdfs(self.root / "generated_pdf", ["team-a"])
        self.assertEqual(files, [self.pdf_a.resolve(), self.pdf_b.resolve()])

    def test_collect_customer_pdfs_accepts_file_target(self) -> None:
        files = collect_customer_pdfs(self.root / "generated_pdf", ["team-b/guide-c.pdf"])
        self.assertEqual(files, [self.pdf_c.resolve()])

    def test_collect_customer_pdfs_rejects_escape_path(self) -> None:
        with self.assertRaises(ValueError):
            collect_customer_pdfs(self.root / "generated_pdf", ["../outside.pdf"])


if __name__ == "__main__":
    unittest.main()

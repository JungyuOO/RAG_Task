from __future__ import annotations

import unittest
from pathlib import Path

from scripts.golden_dataset import GoldenDataset, load_golden_dataset, summarize_golden_dataset


class GoldenDatasetTests(unittest.TestCase):
    def test_load_golden_dataset_fixture(self) -> None:
        dataset = load_golden_dataset(Path("tests/data/golden_dataset_v1.json"))
        self.assertIsInstance(dataset, GoldenDataset)
        self.assertEqual(dataset.version, "1.0")
        self.assertGreaterEqual(len(dataset.cases), 8)

    def test_summary_reports_groups_and_shapes(self) -> None:
        dataset = load_golden_dataset(Path("tests/data/golden_dataset_v1.json"))
        summary = summarize_golden_dataset(dataset)
        self.assertEqual(summary["case_count"], len(dataset.cases))
        self.assertIn("official_ocp", summary["groups"])
        self.assertIn("customer_generated", summary["groups"])
        self.assertIn("mixed", summary["groups"])
        self.assertIn("summary", summary["answer_shapes"])
        self.assertIn("status_analysis", summary["answer_shapes"])
        self.assertIn("checklist", summary["answer_shapes"])
        self.assertIn("procedure", summary["answer_shapes"])
        self.assertIn("comparison", summary["answer_shapes"])


if __name__ == "__main__":
    unittest.main()

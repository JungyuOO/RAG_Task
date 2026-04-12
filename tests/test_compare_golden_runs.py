from __future__ import annotations

import unittest

from scripts.compare_golden_runs import compare_runs, render_markdown_report


class CompareGoldenRunsTests(unittest.TestCase):
    def test_compare_runs_reports_improvement_and_regression(self) -> None:
        baseline = {
            "results": [
                {
                    "case_id": "case-a",
                    "checks": {"group_match": False, "answer_shape_match": False},
                    "elapsed_sec": 10.0,
                },
                {
                    "case_id": "case-b",
                    "checks": {"group_match": True, "answer_shape_match": True},
                    "elapsed_sec": 12.0,
                },
            ]
        }
        candidate = {
            "results": [
                {
                    "case_id": "case-a",
                    "checks": {"group_match": True, "answer_shape_match": True},
                    "elapsed_sec": 8.0,
                },
                {
                    "case_id": "case-b",
                    "checks": {"group_match": True, "answer_shape_match": False},
                    "elapsed_sec": 15.0,
                },
            ]
        }

        diff = compare_runs(baseline, candidate)
        self.assertEqual(diff["improved"], 1)
        self.assertEqual(diff["regressed"], 1)
        self.assertEqual(len(diff["changed_cases"]), 2)

    def test_render_markdown_report_contains_case_details(self) -> None:
        diff = {
            "improved": 1,
            "regressed": 0,
            "changed_cases": [
                {
                    "case_id": "case-a",
                    "before_pass": False,
                    "after_pass": True,
                    "before_elapsed_sec": 10.0,
                    "after_elapsed_sec": 8.0,
                    "changed_checks": {
                        "group_match": {"before": False, "after": True}
                    },
                }
            ],
        }
        report = render_markdown_report(diff, "before.json", "after.json")
        self.assertIn("# Golden Run Comparison", report)
        self.assertIn("### `case-a`", report)
        self.assertIn("Check `group_match`", report)


if __name__ == "__main__":
    unittest.main()

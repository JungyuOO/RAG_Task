from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.golden_dataset import GoldenCase, GoldenDataset
from scripts.run_golden_dataset import detect_answer_shape, evaluate_case, render_markdown_report, run_dataset, summarize_failures


class GoldenDatasetRunnerTests(unittest.TestCase):
    def test_detect_answer_shape_checklist(self) -> None:
        answer = "- check DNS\n- check network\n- check IAM"
        self.assertEqual(detect_answer_shape(answer), "checklist")

    def test_detect_answer_shape_procedure(self) -> None:
        answer = "1. Create the namespace\n2. Apply the manifest\n3. Verify the pod"
        self.assertEqual(detect_answer_shape(answer), "procedure")

    def test_detect_answer_shape_comparison(self) -> None:
        answer = "- Official guide: focuses on prerequisites\n- Customer guide: adds operational checks\n차이도 명확합니다."
        self.assertEqual(detect_answer_shape(answer), "comparison")

    def test_evaluate_case_scores_expected_checks(self) -> None:
        case = GoldenCase(
            id="official-summary",
            question="Explain OAuth and RBAC",
            version_tag="4.21",
            expected_group="official_ocp",
            expected_answer_shape="summary",
            expected_source_tokens=("Authentication_And_Authorization",),
            required_keywords=("OAuth", "RBAC"),
            forbidden_markers=("table of contents",),
        )
        turn_result = {
            "done": True,
            "answer": "OAuth handles authentication and RBAC handles authorization.",
            "retrieved_sources": [
                {
                    "file_name": "OpenShift_Container_Platform-4.21-Authentication_And_Authorization-en-US.pdf",
                    "group": "official_ocp",
                }
            ],
            "answer_citations": [{"file_name": "doc.pdf", "page_number": 12}],
            "elapsed_sec": 1.2,
        }
        result = evaluate_case(case, turn_result)
        self.assertTrue(result["checks"]["completed"])
        self.assertTrue(result["checks"]["group_match"])
        self.assertTrue(result["checks"]["source_match"])
        self.assertTrue(result["checks"]["keywords_present"])
        self.assertTrue(result["checks"]["forbidden_markers_absent"])
        self.assertTrue(result["checks"]["answer_shape_match"])
        self.assertTrue(result["checks"]["citations_present"])

    def test_run_dataset_honors_case_id_and_max_cases(self) -> None:
        dataset = GoldenDataset(
            version="1.0",
            title="fixture",
            cases=(
                GoldenCase(
                    id="case-a",
                    question="a",
                    version_tag="4.21",
                    expected_group="official_ocp",
                    expected_answer_shape="summary",
                    expected_source_tokens=("doc-a",),
                    required_keywords=("alpha",),
                    forbidden_markers=("table of contents",),
                ),
                GoldenCase(
                    id="case-b",
                    question="b",
                    version_tag="4.21",
                    expected_group="official_ocp",
                    expected_answer_shape="summary",
                    expected_source_tokens=("doc-b",),
                    required_keywords=("beta",),
                    forbidden_markers=("table of contents",),
                ),
            ),
        )

        fake_turn = {
            "done": True,
            "answer": "alpha",
            "retrieved_sources": [{"file_name": "doc-a.md", "group": "official_ocp"}],
            "answer_citations": [{"file_name": "doc-a.md"}],
            "elapsed_sec": 0.1,
        }

        with patch("scripts.run_golden_dataset.extract_turn_result", return_value=fake_turn):
            payload = run_dataset("http://localhost:8000", dataset, case_id="case-a", max_cases=1)

        self.assertEqual(payload["case_count"], 1)
        self.assertEqual(payload["results"][0]["case_id"], "case-a")

    def test_summarize_failures_counts_failed_checks(self) -> None:
        payload = {
            "results": [
                {"checks": {"completed": True, "group_match": False, "answer_shape_match": False}},
                {"checks": {"completed": True, "group_match": False, "answer_shape_match": True}},
            ]
        }
        summary = summarize_failures(payload)
        self.assertEqual(summary["failed_case_count"], 2)
        self.assertEqual(summary["failure_buckets"]["group_match"], 2)
        self.assertEqual(summary["failure_buckets"]["answer_shape_match"], 1)

    def test_render_markdown_report_includes_failures(self) -> None:
        payload = {
            "dataset_version": "1.0",
            "title": "fixture",
            "case_count": 1,
            "passed": 0,
            "failed": 1,
            "results": [
                {
                    "case_id": "case-a",
                    "question": "Explain OAuth",
                    "expected_group": "official_ocp",
                    "expected_answer_shape": "summary",
                    "detected_answer_shape": "checklist",
                    "elapsed_sec": 2.3,
                    "observed_groups": ["official_ocp"],
                    "matched_source_tokens": ["Authentication"],
                    "matched_keywords": ["OAuth"],
                    "forbidden_hits": [],
                    "checks": {
                        "completed": True,
                        "group_match": True,
                        "answer_shape_match": False,
                    },
                }
            ],
            "failure_summary": {
                "failed_case_count": 1,
                "failure_buckets": {"answer_shape_match": 1},
            },
        }
        report = render_markdown_report(payload)
        self.assertIn("# Golden Dataset Report", report)
        self.assertIn("`answer_shape_match`: 1", report)
        self.assertIn("### FAIL `case-a`", report)


if __name__ == "__main__":
    unittest.main()

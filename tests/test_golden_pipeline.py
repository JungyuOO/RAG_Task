from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

from scripts.golden_pipeline import execute_pipeline


class GoldenPipelineTests(unittest.TestCase):
    def test_execute_pipeline_writes_run_files(self) -> None:
        output_dir = Path("tests/results/_golden_pipeline_test")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(output_dir, ignore_errors=True))

        fake_payload = {
            "dataset_version": "1.0",
            "title": "fixture",
            "case_count": 1,
            "passed": 1,
            "failed": 0,
            "results": [
                {
                    "case_id": "case-a",
                    "question": "q",
                    "expected_group": "official_ocp",
                    "expected_answer_shape": "summary",
                    "detected_answer_shape": "summary",
                    "checks": {
                        "completed": True,
                        "group_match": True,
                        "source_match": True,
                        "keywords_present": True,
                        "forbidden_markers_absent": True,
                        "answer_shape_match": True,
                        "citations_present": True,
                    },
                    "matched_source_tokens": ["doc"],
                    "matched_keywords": ["oauth"],
                    "forbidden_hits": [],
                    "observed_groups": ["official_ocp"],
                    "observed_files": ["doc.md"],
                    "elapsed_sec": 1.0,
                    "answer_preview": "ok",
                }
            ],
        }

        with patch("scripts.golden_pipeline.run_dataset", return_value=fake_payload):
            result = execute_pipeline(
                dataset_path=Path("tests/data/golden_dataset_v1.json"),
                base_url="http://localhost:8000",
                output_dir=output_dir,
                compare_latest=False,
                max_cases=1,
            )

        json_path = Path(result["output_json"])
        md_path = Path(result["output_md"])
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["passed"], 1)


if __name__ == "__main__":
    unittest.main()

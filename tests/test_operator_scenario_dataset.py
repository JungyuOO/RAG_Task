from __future__ import annotations

import json
import unittest
from pathlib import Path


class OperatorScenarioDatasetTests(unittest.TestCase):
    def test_operator_scenario_dataset_contains_expected_mix(self) -> None:
        path = Path("tests/data/operator_scenarios_v1.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        scenarios = payload.get("scenarios", [])
        self.assertGreaterEqual(len(scenarios), 10)
        self.assertGreaterEqual(sum(1 for s in scenarios if len(s.get("turns", [])) >= 5), 3)
        kinds = {str(s.get("kind") or "") for s in scenarios}
        self.assertTrue({"document", "ocp", "mixed"}.issubset(kinds))
        scenario_ids = {str(s.get("id") or "") for s in scenarios}
        self.assertIn("ocp-system-pod-total-count", scenario_ids)
        self.assertIn("multiturn-doc-followup", scenario_ids)


if __name__ == "__main__":
    unittest.main()

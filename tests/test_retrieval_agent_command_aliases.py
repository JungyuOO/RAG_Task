from __future__ import annotations

import unittest

from app.llm.retrieval_agent import RetrievalAgent


class RetrievalAgentCommandAliasTests(unittest.TestCase):
    def test_namespace_command_query_expands_to_generic_cli_anchors(self) -> None:
        agent = RetrievalAgent(llm_client=None)

        result = agent._fast_path_expand("namespace 확인 명령어 뭐야?", [])

        self.assertIn("oc", result["expanded_query"])
        self.assertIn("project", result["expanded_query"])
        self.assertIn("namespace", result["expanded_query"])

    def test_yaml_command_query_adds_generic_yaml_anchors(self) -> None:
        agent = RetrievalAgent(llm_client=None)

        result = agent._fast_path_expand("yaml 보려면 무슨 명령어 써?", [])

        self.assertIn("oc", result["expanded_query"])
        self.assertIn("manifest", result["expanded_query"])
        self.assertIn("-o", result["expanded_query"])

    def test_namespace_command_query_interpretation_tracks_namespace_resource(self) -> None:
        agent = RetrievalAgent(llm_client=None)

        interpretation = agent.interpret("namespace 확인 명령어 뭐야?", query_result={}, topic_state={})

        self.assertIn("namespace", interpretation["resources"])
        self.assertTrue(interpretation["generic_command_query"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.llm.retrieval_agent import RetrievalAgent
from app.rag.pipeline_streaming import ChatTurnOrchestrator


class VersionFilterBehaviorTests(unittest.TestCase):
    def test_retrieval_agent_does_not_inject_single_available_version(self) -> None:
        agent = RetrievalAgent(llm_client=None)

        detected = agent._detect_versions(
            "pod 확인 명령어 뭐야?",
            [{"version": "4.20"}],
        )

        self.assertEqual(detected, [])

    def test_streaming_orchestrator_leaves_version_unset_without_explicit_signal(self) -> None:
        resolved = ChatTurnOrchestrator._resolve_effective_version_tag(
            None,
            "namespace 확인 명령어 뭐야?",
            {"selected_versions": []},
            {"selected_versions": []},
        )

        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from apps.api.schemas.copilot_chat import CopilotChatHistoryTurn
from apps.api.rag.query.intent_agent import IntentAgent


class _FakeLlmClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.is_enabled = True

    async def generate(self, messages, *, max_tokens=None, temperature=None):  # noqa: ANN001
        del messages, max_tokens, temperature
        return self.response


class IntentAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_lane_classification_is_used_when_available(self) -> None:
        agent = IntentAgent(llm_client=_FakeLlmClient('{"lane":"mixed","doc_query":"문서 설명","live_query":"show pods"}'))

        result = await agent.classify(
            message="문서 설명하고 show pods도 같이 해줘",
            has_connection=True,
            recent_turns=[],
        )

        self.assertEqual(result.lane, "mixed")
        self.assertEqual(result.doc_query, "문서 설명")
        self.assertEqual(result.live_query, "show pods")

    async def test_rule_fallback_handles_document_followup(self) -> None:
        agent = IntentAgent()

        result = await agent.classify(
            message="Can you explain that in more detail?",
            has_connection=True,
            recent_turns=[CopilotChatHistoryTurn(role="assistant", text="Previous document answer", lane="doc_hybrid")],
        )

        self.assertEqual(result.lane, "doc")
        self.assertEqual(result.doc_query, "Can you explain that in more detail?")

    async def test_existing_cluster_nlb_question_prefers_doc_lane(self) -> None:
        agent = IntentAgent()

        result = await agent.classify(
            message="이미 올라가 있는 aws 클러스터에 ingress nlb 붙이려면?",
            has_connection=False,
            recent_turns=[],
        )

        self.assertEqual(result.lane, "doc")
        self.assertEqual(result.doc_query, "이미 올라가 있는 aws 클러스터에 ingress nlb 붙이려면?")

    async def test_cluster_update_flow_prefers_doc_lane(self) -> None:
        agent = IntentAgent()

        result = await agent.classify(
            message="openshift 클러스터 업데이트 큰 흐름이 어떻게 되는지",
            has_connection=False,
            recent_turns=[],
        )

        self.assertEqual(result.lane, "doc")
        self.assertEqual(result.doc_query, "openshift 클러스터 업데이트 큰 흐름이 어떻게 되는지")

    async def test_current_pod_count_remains_live_without_connection(self) -> None:
        agent = IntentAgent()

        result = await agent.classify(
            message="현재 클러스터에 pod 몇 개 있어?",
            has_connection=False,
            recent_turns=[],
        )

        self.assertEqual(result.lane, "needs_connection")
        self.assertEqual(result.live_query, "현재 클러스터에 pod 몇 개 있어?")


if __name__ == "__main__":
    unittest.main()



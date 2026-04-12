from __future__ import annotations

import asyncio
import unittest

from app.llm.intent_agent import IntentAgent


class _LlmStub:
    async def generate(self, messages, max_tokens=None):  # pragma: no cover
        return ""


class IntentAgentTests(unittest.TestCase):
    def test_storage_relationship_question_is_out_of_scope_general(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("StorageClass와 PV 관계를 설명해줘", context={}))

        self.assertEqual(result["intent"], "general")

    def test_route_creation_guidance_question_is_rag_by_heuristic(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("Route를 생성할 때 기본적으로 확인해야 할 내용 설명해줘", context={}))

        self.assertEqual(result["intent"], "rag")

    def test_pod_command_question_is_rag_by_heuristic(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("보통 pod 확인하는 명령어 뭐야?", context={}))

        self.assertEqual(result["intent"], "rag")

    def test_namespace_command_question_is_rag_by_heuristic(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("namespace 확인 명령어 뭐야?", context={}))

        self.assertEqual(result["intent"], "rag")


if __name__ == "__main__":
    unittest.main()

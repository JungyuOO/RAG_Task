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

        result = asyncio.run(agent.classify("StorageClass? PV ??? ????", context={}))

        self.assertEqual(result["intent"], "general")

    def test_route_creation_guidance_question_is_rag_by_heuristic(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("Route? ??? ? ????? ???? ? ?? ????", context={}))

        self.assertEqual(result["intent"], "rag")

    def test_pod_command_question_is_rag_by_heuristic(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("?? pod ???? ??? ???", context={}))

        self.assertEqual(result["intent"], "rag")

    def test_namespace_command_question_is_rag_by_heuristic(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("namespace ?? ??? ???", context={}))

        self.assertEqual(result["intent"], "rag")

    def test_capability_question_is_general_by_heuristic(self) -> None:
        agent = IntentAgent(_LlmStub())

        result = asyncio.run(agent.classify("?? ??? ? ??? ???", context={}))

        self.assertEqual(result["intent"], "general")


if __name__ == "__main__":
    unittest.main()

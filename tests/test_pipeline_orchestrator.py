from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.rag.pipeline import PipelineOrchestrator


class TestAgentOrchestration(unittest.TestCase):
    def test_greeting_skips_retrieval(self) -> None:
        mock_intent = MagicMock()
        mock_intent.classify = AsyncMock(return_value={"intent": "greeting", "confidence": 0.95})

        orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orchestrator.intent_agent = mock_intent
        orchestrator.retrieval_agent = MagicMock()

        result = asyncio.run(orchestrator.classify_intent("안녕하세요", {}))

        self.assertEqual(result["intent"], "greeting")
        orchestrator.retrieval_agent.expand.assert_not_called()

    def test_rag_intent_triggers_retrieval(self) -> None:
        mock_intent = MagicMock()
        mock_intent.classify = AsyncMock(return_value={"intent": "rag", "search_query": "OCP Pod", "keywords": ["OCP"]})

        mock_retrieval = MagicMock()
        mock_retrieval.expand = AsyncMock(return_value={"expanded_query": "OpenShift Pod deployment", "alternatives": [], "target_versions": [], "multi_source": False})

        orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orchestrator.intent_agent = mock_intent
        orchestrator.retrieval_agent = mock_retrieval

        intent = asyncio.run(orchestrator.classify_intent("OCP Pod 배포", {}))
        expanded = asyncio.run(orchestrator.expand_query("OCP Pod 배포", intent, []))

        self.assertEqual(intent["intent"], "rag")
        self.assertIn("expanded_query", expanded)


if __name__ == "__main__":
    unittest.main()

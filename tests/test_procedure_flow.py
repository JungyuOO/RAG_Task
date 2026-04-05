from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock


class TestProcedureFlow(unittest.TestCase):
    def test_procedure_detected_offers_choice(self):
        """다단계 절차 감지 시 순차/일괄 선택 제안"""
        from app.rag.pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

        mock_answer = MagicMock()
        mock_answer.check_procedure = AsyncMock(return_value={
            "has_procedure": True,
            "total_steps": 10,
            "offer_message": "총 10단계로 설명해드릴 수 있습니다. 모든 단계를 한번에 설명해드릴까요, 순차적으로 설명해드릴까요?"
        })
        orch.answer_agent = mock_answer

        result = asyncio.run(orch.check_procedure("OCP 배포 단계별 설명", [{"text": "Step 1...Step 10..."}]))
        self.assertTrue(result["has_procedure"])
        self.assertEqual(result["total_steps"], 10)

    def test_step_navigation_next(self):
        """'다음 단계' 요청 처리"""
        from app.rag.pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

        mock_session_repository = MagicMock()
        mock_session_repository.topic_state = MagicMock(return_value={
            "procedure_state": {"current_step": 2, "total_steps": 5}
        })
        orch.session_repository = mock_session_repository

        intent = {"intent": "step_navigation", "step_target": "next"}
        result = asyncio.run(orch.handle_step_navigation(intent, "session-1", []))
        self.assertEqual(result["target_step"], 3)
        self.assertEqual(result["total_steps"], 5)

    def test_step_navigation_specific(self):
        """'3단계 설명해줘' 요청 처리"""
        from app.rag.pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

        mock_session_repository = MagicMock()
        mock_session_repository.topic_state = MagicMock(return_value={
            "procedure_state": {"current_step": 1, "total_steps": 5}
        })
        orch.session_repository = mock_session_repository

        intent = {"intent": "step_navigation", "step_target": "3"}
        result = asyncio.run(orch.handle_step_navigation(intent, "session-1", []))
        self.assertEqual(result["target_step"], 3)

    def test_step_navigation_prev(self):
        """'이전 단계' 요청 처리"""
        from app.rag.pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

        mock_session_repository = MagicMock()
        mock_session_repository.topic_state = MagicMock(return_value={
            "procedure_state": {"current_step": 3, "total_steps": 5}
        })
        orch.session_repository = mock_session_repository

        intent = {"intent": "step_navigation", "step_target": "prev"}
        result = asyncio.run(orch.handle_step_navigation(intent, "session-1", []))
        self.assertEqual(result["target_step"], 2)


if __name__ == "__main__":
    unittest.main()

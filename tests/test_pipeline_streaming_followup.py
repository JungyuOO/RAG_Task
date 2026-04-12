from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.rag.pipeline_streaming import ChatTurnDeps, ChatTurnOrchestrator


class _SessionRepositoryStub:
    def __init__(self, recent_turns: list[SimpleNamespace]) -> None:
        self._recent_turns = recent_turns

    def recent_turns(self, _session_id: str) -> list[SimpleNamespace]:
        return list(self._recent_turns)


def _build_orchestrator(recent_turns: list[SimpleNamespace]) -> ChatTurnOrchestrator:
    deps = ChatTurnDeps(
        detect_non_korean_query=None,
        session_repository=_SessionRepositoryStub(recent_turns),
        should_skip_procedure_shortcut=None,
        detect_procedure_followup=None,
        build_procedure_followup_answer=None,
        looks_like_step_navigation_without_state=None,
        resolve_turn_context=None,
        domain_guard_state=None,
        prepare_retrieval_state=None,
        resolve_answer_route=None,
        interleave_context_items_by_source=None,
        build_context_blocks=None,
        ensure_topic_for_resolution=None,
        build_answer_cache_key=None,
        canonical_cache_query=None,
        build_policy_answer=None,
        build_missing_extractive_answer=None,
        select_code_example_context_items=None,
        resolve_requested_resource_kinds=None,
        prefer_block_type_items=None,
        finalize_answer=None,
        store_assistant_turn=None,
        build_llm_failure_fallback=None,
        get_prompt_composer=None,
        answer_service=None,
        answer_cache_repository=None,
        answer_rewrite_agent=None,
        llm=None,
    )
    return ChatTurnOrchestrator(deps)


class PipelineStreamingFollowupTests(unittest.TestCase):
    def test_transform_previous_answer_for_answer_summary_request(self) -> None:
        recent_turns = [SimpleNamespace(role="assistant", content="previous answer", metadata={})]
        self.assertTrue(
            ChatTurnOrchestrator._should_transform_previous_answer(
                "this answer summary in 3 lines",
                recent_turns,
            )
        )

    def test_do_not_transform_for_document_reference_question(self) -> None:
        recent_turns = [SimpleNamespace(role="assistant", content="previous answer", metadata={})]
        self.assertFalse(
            ChatTurnOrchestrator._should_transform_previous_answer(
                "this document explain why Route 53 is needed",
                recent_turns,
            )
        )

    def test_followup_allowed_sources_only_for_document_reference(self) -> None:
        metadata = {
            "answer_citations": [
                {"source_path": "/docs/one.html"},
                {"source_path": "/docs/two.html"},
            ]
        }
        orchestrator = _build_orchestrator([SimpleNamespace(role="assistant", content="previous answer", metadata=metadata)])
        self.assertEqual(
            orchestrator._derive_followup_allowed_sources("session-1", "this document explain again"),
            {"/docs/one.html", "/docs/two.html"},
        )
        self.assertIsNone(
            orchestrator._derive_followup_allowed_sources("session-1", "summarize this answer in 3 lines")
        )

    def test_coerce_checklist_shape(self) -> None:
        orchestrator = _build_orchestrator([])
        answer = "First validation item. Second validation item. Third validation item."
        coerced = orchestrator._coerce_answer_shape(answer, "checklist")
        self.assertTrue(orchestrator._answer_matches_shape(coerced, "checklist"))

    def test_coerce_procedure_shape(self) -> None:
        orchestrator = _build_orchestrator([])
        answer = "First step description. Second step description. Third step description."
        coerced = orchestrator._coerce_answer_shape(answer, "procedure")
        self.assertTrue(orchestrator._answer_matches_shape(coerced, "procedure"))

    def test_coerce_comparison_shape(self) -> None:
        orchestrator = _build_orchestrator([])
        answer = (
            "One side explains installation prerequisites. "
            "The other side explains operational preparation and validation."
        )
        coerced = orchestrator._coerce_answer_shape(answer, "comparison")
        self.assertTrue(orchestrator._answer_matches_shape(coerced, "comparison"))


if __name__ == "__main__":
    unittest.main()

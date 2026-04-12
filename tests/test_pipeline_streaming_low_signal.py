from __future__ import annotations

import unittest

from app.rag.pipeline_streaming import ChatTurnDeps, ChatTurnOrchestrator


class _AnswerServiceStub:
    def looks_like_low_signal_retrieved_answer(self, answer: str) -> bool:
        return "table of contents" in answer.casefold()

    def build_extractive_text_answer(self, _items: list[dict]) -> str | None:
        return "- fallback text"

    def build_extractive_compare_answer(self, _items: list[dict]) -> str | None:
        return "- fallback compare"


def _build_orchestrator() -> ChatTurnOrchestrator:
    deps = ChatTurnDeps(
        detect_non_korean_query=None,
        session_repository=None,
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
        answer_service=_AnswerServiceStub(),
        answer_cache_repository=None,
        answer_rewrite_agent=None,
        llm=None,
    )
    return ChatTurnOrchestrator(deps)


class PipelineStreamingLowSignalTests(unittest.TestCase):
    def test_low_signal_grounded_generation_falls_back_to_extractive_text(self) -> None:
        orchestrator = _build_orchestrator()
        answer = orchestrator._fallback_low_signal_answer(
            "Table of contents\n1. Intro\n2. Overview",
            use_retrieved_context=True,
            answer_route="grounded_generation",
            selected_context_items=[],
        )
        self.assertEqual(answer, "- fallback text")

    def test_low_signal_compare_falls_back_to_extractive_compare(self) -> None:
        orchestrator = _build_orchestrator()
        answer = orchestrator._fallback_low_signal_answer(
            "Table of contents\n1. Intro\n2. Overview",
            use_retrieved_context=True,
            answer_route="extractive_compare",
            selected_context_items=[],
        )
        self.assertEqual(answer, "- fallback compare")

    def test_non_low_signal_answer_is_preserved(self) -> None:
        orchestrator = _build_orchestrator()
        answer = orchestrator._fallback_low_signal_answer(
            "This answer explains the actual operational checks.",
            use_retrieved_context=True,
            answer_route="grounded_generation",
            selected_context_items=[],
        )
        self.assertEqual(answer, "This answer explains the actual operational checks.")


if __name__ == "__main__":
    unittest.main()

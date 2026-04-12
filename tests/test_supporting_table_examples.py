from __future__ import annotations

import unittest

from app.rag.answer import AnswerGenerator
from app.rag.pipeline_streaming import ChatTurnDeps, ChatTurnOrchestrator


class SupportingTableExampleTests(unittest.TestCase):
    def test_build_supporting_table_example_returns_structured_payload(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "chunk": {
                    "source_path": "/docs/sample.pdf",
                    "page_number": 5,
                    "metadata": {"page_start": 5, "page_end": 5, "html_anchor": "page-5", "primary_block_anchor": "page-5-block-2"},
                    "text": "| Name | Value |\n| --- | --- |\n| Pod | demo |",
                }
            }
        ]
        appendix = generator.build_supporting_table_example(context_items)
        self.assertIsNotNone(appendix)
        self.assertEqual(appendix["type"], "table")
        self.assertEqual(appendix["title"], "관련 표")
        self.assertEqual(appendix["page_start"], "5")
        self.assertEqual(appendix["source_path"], "/docs/sample.pdf")
        self.assertEqual(appendix["html_anchor"], "page-5")
        self.assertEqual(appendix["block_anchor"], "page-5-block-2")
        self.assertIn("| Name | Value |", appendix["content"])

    def test_build_supporting_examples_skips_table_when_answer_already_contains_table(self) -> None:
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
            select_code_example_context_items=lambda *_args, **_kwargs: [],
            resolve_requested_resource_kinds=lambda _qi: set(),
            prefer_block_type_items=lambda items, **_kwargs: items,
            finalize_answer=None,
            store_assistant_turn=None,
            build_llm_failure_fallback=None,
            get_prompt_composer=None,
            answer_service=AnswerGenerator(retrieval_service=None),
            answer_cache_repository=None,
            answer_rewrite_agent=None,
            llm=None,
        )
        orchestrator = ChatTurnOrchestrator(deps)
        examples = orchestrator._build_supporting_examples(
            answer="| Name | Value |\n| --- | --- |\n| Pod | demo |",
            deps=deps,
            user_message="pod를 표로 보여줘",
            query_interpretation={"response_shape": "table"},
            ordered_context_items=[],
            selected_context_items=[],
        )
        self.assertEqual(examples, [])


if __name__ == "__main__":
    unittest.main()

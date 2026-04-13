from __future__ import annotations

import unittest

from app.rag.retrieval_state_builder import RetrievalStateBuilder
from app.rag.types import TurnPolicyDecision


class RetrievalFollowupFastpathTests(unittest.TestCase):
    def test_followup_fast_query_inherits_resource_and_yaml_focus(self) -> None:
        result = RetrievalStateBuilder._build_followup_fast_query_result(
            "그거 yaml로 보려면?",
            {
                "last_explicit_resources": ["pod"],
                "last_code_resource_kind": "pod",
            },
            "그거 yaml로 보려면?",
        )

        self.assertIn("pod", result["resources"])
        self.assertIn("yaml", result["format_constraints"])
        self.assertEqual(result["response_shape"], "code")
        self.assertIn("pod", result["refined_query"])
        self.assertIn("yaml", result["refined_query"])

    def test_short_new_question_with_topic_anchor_does_not_force_followup_fastpath(self) -> None:
        policy = TurnPolicyDecision(
            turn_type="document_query",
            response_mode="rag",
            use_retrieval=True,
            use_memory_rewrite=False,
            allow_preview=True,
            allow_citations=True,
        )

        should_skip = RetrievalStateBuilder._should_skip_expand_with_llm(
            policy,
            {"selected_sources": ["cli_tools.md"], "last_explicit_resources": ["namespace"]},
            "pod 확인하는 명령어 뭐야?",
        )

        self.assertFalse(should_skip)

    def test_document_followup_without_reference_does_not_force_followup_fastpath(self) -> None:
        policy = TurnPolicyDecision(
            turn_type="document_followup",
            response_mode="rag",
            use_retrieval=True,
            use_memory_rewrite=False,
            allow_preview=True,
            allow_citations=True,
        )

        should_skip = RetrievalStateBuilder._should_skip_expand_with_llm(
            policy,
            {"selected_sources": ["cli_tools.md"], "last_explicit_resources": ["namespace"]},
            "pod 확인하는 명령어 뭐야?",
        )

        self.assertFalse(should_skip)

    def test_select_procedure_token_matches_prefers_chunks_with_strong_query_token(self) -> None:
        ordered = [
            {
                "chunk": {
                    "chunk_id": "c1",
                    "text": "Review Machine API Operator pod status\n$ oc describe pod/<machine_api_operator_pod_name> -n openshift-machine-api",
                    "metadata": {"block_types": "code", "section_title": "Investigating etcd installation issues"},
                }
            },
            {
                "chunk": {
                    "chunk_id": "c2",
                    "text": "Check the status of etcd pods\n$ oc get pods -n openshift-etcd",
                    "metadata": {"block_types": "code", "section_title": "Investigating etcd installation issues"},
                }
            },
        ]

        matches = RetrievalStateBuilder._select_procedure_token_matches(
            ordered,
            user_message="etcd pod 상태 확인 명령어 알려줘",
            query_interpretation={"normalized_keywords": ["etcd", "pod", "상태", "확인", "명령어"]},
            limit=2,
        )

        self.assertEqual(matches[0]["chunk"]["chunk_id"], "c2")

    def test_followup_fast_query_prefers_active_slot_resources(self) -> None:
        result = RetrievalStateBuilder._build_followup_fast_query_result(
            "that yaml please",
            {
                "active_slot": {
                    "lane": "document",
                    "resources": ["deployment"],
                    "code_resource_kind": "deployment",
                    "selected_versions": ["4.20"],
                },
                "last_explicit_resources": ["pod"],
            },
            "that yaml please",
        )

        self.assertEqual(result["resources"][0], "deployment")
        self.assertEqual(result["target_versions"], ["4.20"])
        self.assertIn("yaml", result["format_constraints"])

    def test_document_followup_with_active_document_slot_and_reference_uses_fastpath(self) -> None:
        policy = TurnPolicyDecision(
            turn_type="document_followup",
            response_mode="rag",
            use_retrieval=True,
            use_memory_rewrite=False,
            allow_preview=True,
            allow_citations=True,
        )

        should_skip = RetrievalStateBuilder._should_skip_expand_with_llm(
            policy,
            {
                "active_slot": {
                    "lane": "document",
                    "sources": ["cli_tools.md"],
                    "resources": ["pod"],
                },
            },
            "that yaml please",
        )

        self.assertTrue(should_skip)



if __name__ == "__main__":
    unittest.main()

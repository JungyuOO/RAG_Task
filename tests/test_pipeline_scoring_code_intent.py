from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.rag.pipeline_scoring import PipelineRetrievalMixin


class _ScoringHarness(PipelineRetrievalMixin):
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            retrieval_final_ce_weight=0.65,
            retrieval_final_metadata_weight=0.25,
            retrieval_final_anchor_weight=0.10,
            retrieval_explain_focus_boost=0.08,
            retrieval_gate_threshold=0.30,
            retrieval_explain_gate_threshold=0.24,
        )

    @staticmethod
    def _resolve_requested_resource_kinds(query_interpretation):  # noqa: ANN001
        return {str(v).casefold() for v in (query_interpretation or {}).get("resources", []) if v}

    @staticmethod
    def _heading_overlap_score(user_message, metadata):  # noqa: ANN001
        return 0.0


class PipelineScoringCodeIntentTests(unittest.TestCase):
    def test_code_intent_boosts_dense_signal(self) -> None:
        harness = _ScoringHarness()
        item = {
            "chunk": {
                "text": "```text\n$ oc get pods -n demo\n```",
                "source_path": "/docs/support.md",
                "metadata": {
                    "block_types": "code",
                    "code_language": "bash",
                    "code_subtype": "cli_command",
                    "has_cli_block": True,
                },
            },
            "dense_score": 0.52,
            "sparse_score": 0.01,
            "ce_score": 0.0,
            "rerank_score": 0.0,
            "score": 0.0,
        }
        query = {
            "intent": "cli_example",
            "response_shape": "code",
            "format_constraints": ["cli"],
            "actions": ["explain"],
            "resources": ["pod"],
            "normalized_keywords": ["pod", "명령어"],
            "document_group_preference": "official_ocp",
        }

        scored = harness._metadata_aware_score("pod 확인 명령어", query, item)

        self.assertGreaterEqual(scored["ce_score"], 0.33)
        self.assertGreater(scored["final_retrieval_score"], 0.20)

    def test_code_intent_uses_lower_gate_threshold(self) -> None:
        harness = _ScoringHarness()
        retrieved = [{"final_retrieval_score": 0.13}]
        decision = harness._should_use_retrieved_context(
            SimpleNamespace(turn_type="document_query"),
            retrieved,
            top_score=0.13,
            query_interpretation={"intent": "cli_example", "response_shape": "code"},
        )

        self.assertTrue(decision)

    def test_metadata_scoring_prefers_strong_query_token_match_in_procedure_chunk(self) -> None:
        harness = _ScoringHarness()
        item_match = {
            "chunk": {
                "text": "Check the status of etcd pods\n$ oc get pods -n openshift-etcd",
                "source_path": "/docs/support.md",
                "metadata": {
                    "block_types": "code",
                    "code_language": "bash",
                    "code_subtype": "cli_command",
                    "has_cli_block": True,
                    "is_procedure": True,
                    "section_title": "Investigating etcd installation issues",
                },
            },
            "dense_score": 0.45,
            "sparse_score": 0.03,
            "ce_score": 0.0,
            "rerank_score": 0.0,
            "score": 0.0,
        }
        item_miss = {
            "chunk": {
                "text": "Check machine-api operator status\n$ oc describe pod/<machine_api_operator_pod_name> -n openshift-machine-api",
                "source_path": "/docs/support.md",
                "metadata": {
                    "block_types": "code",
                    "code_language": "bash",
                    "code_subtype": "cli_command",
                    "has_cli_block": True,
                    "is_procedure": True,
                    "section_title": "Investigating operator issues",
                },
            },
            "dense_score": 0.45,
            "sparse_score": 0.03,
            "ce_score": 0.0,
            "rerank_score": 0.0,
            "score": 0.0,
        }
        query = {
            "intent": "cli_example",
            "response_shape": "code",
            "format_constraints": ["cli"],
            "actions": ["explain"],
            "resources": ["pod"],
            "normalized_keywords": ["etcd", "pod", "상태", "확인", "명령어"],
            "document_group_preference": "official_ocp",
        }

        scored_match = harness._metadata_aware_score("etcd pod 상태 확인 명령어 알려줘", query, item_match)
        scored_miss = harness._metadata_aware_score("etcd pod 상태 확인 명령어 알려줘", query, item_miss)

        self.assertGreater(scored_match["final_retrieval_score"], scored_miss["final_retrieval_score"])


if __name__ == "__main__":
    unittest.main()

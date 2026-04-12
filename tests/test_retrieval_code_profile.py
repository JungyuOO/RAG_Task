from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.rag.retrieval_state_builder import RetrievalStateBuilder, RetrievalStateDeps


def _build_builder() -> RetrievalStateBuilder:
    deps = RetrievalStateDeps(
        resolve_turn_context=None,
        build_non_retrieval_state=None,
        build_rewrite_context_from_topic=None,
        rewrite_query_with_llm=None,
        index_repository=None,
        intent_agent=None,
        retrieval_agent=None,
        expand_query_with_resource_aliases=None,
        expand_query_with_context=None,
        embedder=None,
        retrieval_service=None,
        retriever=SimpleNamespace(candidate_pool_size=15, top_k=5),
        reranker=None,
        metadata_aware_rerank=None,
        expand_local_context_items=None,
        expand_topic_anchor_context_items=None,
        should_use_retrieved_context=None,
        apply_precision_filter=None,
        apply_focus_filter=None,
        find_fallback_code_context_items=None,
        settings=SimpleNamespace(pgvector_prefilter_limit=0),
    )
    return RetrievalStateBuilder(deps)


class RetrievalCodeProfileTests(unittest.TestCase):
    def test_code_intent_profile_biases_dense_and_skips_cross_encoder(self) -> None:
        builder = _build_builder()

        profile = builder._build_retrieval_profile(
            {
                "intent": "cli_example",
                "response_shape": "code",
                "format_constraints": ["cli"],
            }
        )

        self.assertGreater(profile["dense_rrf_weight"], profile["sparse_rrf_weight"])
        self.assertTrue(profile["skip_cross_encoder"])
        self.assertLess(profile["rerank_limit"], 10)

    def test_text_intent_profile_keeps_balanced_rrf_weights(self) -> None:
        builder = _build_builder()

        profile = builder._build_retrieval_profile(
            {
                "intent": "explain",
                "response_shape": "text",
                "format_constraints": [],
            }
        )

        self.assertEqual(profile["dense_rrf_weight"], 1.0)
        self.assertEqual(profile["sparse_rrf_weight"], 1.0)
        self.assertFalse(profile["skip_cross_encoder"])
        self.assertEqual(profile["rerank_limit"], 10)

    def test_format_focused_queries_for_yaml_resource(self) -> None:
        builder = _build_builder()

        queries = builder._build_format_focused_queries(
            {
                "resources": ["pod"],
                "format_constraints": ["yaml"],
                "response_shape": "code",
            }
        )

        self.assertIn("oc get pod -o yaml", queries)
        self.assertIn("oc describe pod", queries)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.rag.retrieval_state_builder import RetrievalStateBuilder, RetrievalStateDeps


def _build_deps(index_repository) -> RetrievalStateDeps:
    return RetrievalStateDeps(
        resolve_turn_context=None,
        build_non_retrieval_state=None,
        build_rewrite_context_from_topic=None,
        rewrite_query_with_llm=None,
        index_repository=index_repository,
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


class _DenseRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def search_dense_candidates(self, query_vector, **kwargs):
        self.calls.append(("dense", kwargs))
        return [{"chunk": {"chunk_id": f"a-{idx}"}} for idx in range(20)]

    def load(self, **kwargs):
        self.calls.append(("load", kwargs))
        return [{"chunk": {"chunk_id": "b"}}]


class _FallbackRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def search_dense_candidates(self, query_vector, **kwargs):
        self.calls.append(("dense", kwargs))
        raise RuntimeError("no pgvector")

    def load(self, **kwargs):
        self.calls.append(("load", kwargs))
        return [{"chunk": {"chunk_id": "b"}}]


class _SupplementRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def search_dense_candidates(self, query_vector, **kwargs):
        self.calls.append(("dense", kwargs))
        return [{"chunk": {"chunk_id": "dense-only"}}]

    def load(self, **kwargs):
        self.calls.append(("load", kwargs))
        return [
            {"chunk": {"chunk_id": "dense-only"}},
            {"chunk": {"chunk_id": "fallback-1"}},
            {"chunk": {"chunk_id": "fallback-2"}},
        ]


class RetrievalStateBuilderPgvectorTests(unittest.TestCase):
    def test_load_index_candidate_pool_prefers_dense_candidates(self) -> None:
        repo = _DenseRepo()
        builder = RetrievalStateBuilder(_build_deps(repo))
        items, strategy = builder._load_index_candidate_pool(
            builder.deps,
            query_vector=[0.1, 0.2],
            source_filter=["/docs/a.pdf"],
            target_versions=["4.21"],
            doc_type="official",
            document_group_preference="official_ocp",
        )
        self.assertEqual(strategy, "pgvector_dense")
        self.assertEqual(items[0]["chunk"]["chunk_id"], "a-0")
        self.assertEqual(repo.calls[0][0], "dense")

    def test_load_index_candidate_pool_falls_back_to_full_load(self) -> None:
        repo = _FallbackRepo()
        builder = RetrievalStateBuilder(_build_deps(repo))
        items, strategy = builder._load_index_candidate_pool(
            builder.deps,
            query_vector=[0.1, 0.2],
            source_filter=None,
            target_versions=[],
            doc_type=None,
            document_group_preference="auto",
        )
        self.assertEqual(strategy, "full_load")
        self.assertEqual(items[0]["chunk"]["chunk_id"], "b")
        self.assertEqual(repo.calls[-1][0], "load")

    def test_candidate_prefilter_limit_uses_candidate_pool_size(self) -> None:
        deps = _build_deps(_DenseRepo())
        builder = RetrievalStateBuilder(deps)
        self.assertEqual(builder._candidate_prefilter_limit(deps.retriever, deps.settings), 60)

    def test_candidate_prefilter_limit_uses_explicit_setting_when_present(self) -> None:
        deps = _build_deps(_DenseRepo())
        deps.settings.pgvector_prefilter_limit = 128
        builder = RetrievalStateBuilder(deps)
        self.assertEqual(builder._candidate_prefilter_limit(deps.retriever, deps.settings), 128)

    def test_load_index_candidate_pool_supplements_dense_results_when_too_small(self) -> None:
        repo = _SupplementRepo()
        deps = _build_deps(repo)
        deps.retriever = SimpleNamespace(candidate_pool_size=4, top_k=2)
        builder = RetrievalStateBuilder(deps)
        items, strategy = builder._load_index_candidate_pool(
            builder.deps,
            query_vector=[0.1, 0.2],
            source_filter=None,
            target_versions=[],
            doc_type=None,
            document_group_preference="auto",
        )
        self.assertEqual(strategy, "pgvector_dense_supplemented")
        self.assertEqual([item["chunk"]["chunk_id"] for item in items], ["dense-only", "fallback-1", "fallback-2"])
        self.assertEqual(repo.calls[0][0], "dense")
        self.assertEqual(repo.calls[1][0], "load")


if __name__ == "__main__":
    unittest.main()

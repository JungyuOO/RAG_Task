from __future__ import annotations

import hashlib
import math
import unittest

from app.rag.retrieval import HybridRetriever
from app.rag.utils import tokenize


def _make_test_vector(text: str, dim: int = 32) -> list[float]:
    """Deterministic fake vector for testing purposes."""
    digest = hashlib.md5(text.encode()).digest()
    vals = [((digest[i % 16] ^ (i * 7)) - 128) / 128.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _make_retriever(**overrides) -> HybridRetriever:
    defaults = dict(
        top_k=6, candidate_pool_size=14,
        dense_weight=0.45, sparse_weight=0.25, title_weight=0.15,
        bm25_k1=1.2, bm25_b=0.75,
        rerank_base_weight=0.68, rerank_overlap_weight=0.17,
        rerank_title_weight=0.08, rerank_title_bonus_weight=0.07,
        rerank_compact_bonus_weight=0.12,
        title_match_bonus=0.35,
    )
    defaults.update(overrides)
    return HybridRetriever(**defaults)


class RetrievalTests(unittest.TestCase):
    def test_compact_query_matches_spaced_chunk_tokens(self) -> None:
        retriever = _make_retriever(top_k=3, candidate_pool_size=3)
        query = "staticprovisioning example"
        query_vector = _make_test_vector(query, dim=32)
        items = [
            {
                "vector": _make_test_vector("static provisioning example with manual pv creation", dim=32),
                "chunk": {
                    "chunk_id": "chunk-static",
                    "source_path": "스토리지.pdf",
                    "tokens": tokenize("static provisioning example with manual pv creation"),
                    "text": "static provisioning example with manual pv creation",
                },
            },
            {
                "vector": _make_test_vector("dynamic provisioning with storageclass", dim=32),
                "chunk": {
                    "chunk_id": "chunk-dynamic",
                    "source_path": "스토리지.pdf",
                    "tokens": tokenize("dynamic provisioning with storageclass"),
                    "text": "dynamic provisioning with storageclass",
                },
            },
        ]

        results = retriever.search(query, query_vector, items)

        self.assertEqual(results[0]["chunk"]["chunk_id"], "chunk-static")
        self.assertGreater(results[0].get("compact_match_bonus", 0.0), 0.0)


    def test_compute_retrieval_metrics_empty(self) -> None:
        """결과가 없으면 모든 지표가 0이다."""
        retriever = _make_retriever()
        metrics = retriever.compute_retrieval_metrics([], min_score=0.12)
        self.assertEqual(metrics["hit_count"], 0)
        self.assertEqual(metrics["total_count"], 0)
        self.assertAlmostEqual(metrics["mean_score"], 0.0)

    def test_compute_retrieval_metrics_with_results(self) -> None:
        """적중 수, 평균 점수, 1·2위 점수 차이가 올바르게 계산된다."""
        retriever = _make_retriever()
        results = [
            {"rerank_score": 0.5, "dense_score": 0.6, "sparse_score": 0.3},
            {"rerank_score": 0.3, "dense_score": 0.4, "sparse_score": 0.5},
            {"rerank_score": 0.05, "dense_score": 0.1, "sparse_score": 0.02},
        ]
        metrics = retriever.compute_retrieval_metrics(results, min_score=0.12)
        self.assertEqual(metrics["hit_count"], 2)
        self.assertEqual(metrics["total_count"], 3)
        self.assertAlmostEqual(metrics["top_score"], 0.5)
        self.assertAlmostEqual(metrics["score_gap"], 0.2)
        self.assertGreater(metrics["hit_rate"], 0.0)
        # dense/sparse 순위 상관도가 -1~1 범위인지 확인
        self.assertGreaterEqual(metrics["dense_sparse_correlation"], -1.0)
        self.assertLessEqual(metrics["dense_sparse_correlation"], 1.0)

    def test_rank_values_with_ties(self) -> None:
        """동점 값에 대해 평균 순위를 반환한다."""
        ranks = HybridRetriever._rank_values([0.5, 0.3, 0.5])
        # 0.5 두 개가 공동 1~2위 → 평균 1.5, 0.3은 3위
        self.assertAlmostEqual(ranks[0], 1.5)
        self.assertAlmostEqual(ranks[2], 1.5)
        self.assertAlmostEqual(ranks[1], 3.0)


if __name__ == "__main__":
    unittest.main()

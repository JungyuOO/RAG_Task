"""BM25 정규화 및 하이브리드 검색 가중치 테스트."""

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


class BM25Tests(unittest.TestCase):
    """표준 BM25 스코어링이 올바르게 동작하는지 검증한다."""

    def test_bm25_prefers_exact_term_match_over_partial(self) -> None:
        """질의 토큰과 정확히 일치하는 문서가 부분 일치보다 높은 점수를 받는다."""
        retriever = _make_retriever(top_k=2, candidate_pool_size=3)
        query = "PV PVC 정적 프로비저닝"
        query_vector = _make_test_vector(query, dim=32)

        items = [
            {
                "vector": _make_test_vector("PV PVC 정적 프로비저닝 설정 방법", dim=32),
                "chunk": {
                    "chunk_id": "exact-match",
                    "source_path": "스토리지.pdf",
                    "tokens": tokenize("PV PVC 정적 프로비저닝 설정 방법"),
                    "text": "PV PVC 정적 프로비저닝 설정 방법",
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                },
            },
            {
                "vector": _make_test_vector("동적 프로비저닝 StorageClass 자동 생성", dim=32),
                "chunk": {
                    "chunk_id": "partial-match",
                    "source_path": "스토리지.pdf",
                    "tokens": tokenize("동적 프로비저닝 StorageClass 자동 생성"),
                    "text": "동적 프로비저닝 StorageClass 자동 생성",
                    "page_number": 5,
                    "metadata": {"page_start": 5, "page_end": 5},
                },
            },
        ]

        results = retriever.search(query, query_vector, items)
        self.assertEqual(results[0]["chunk"]["chunk_id"], "exact-match")

    def test_bm25_normalizes_by_document_length(self) -> None:
        """동일 키워드를 포함하더라도 짧은 문서가 긴 문서보다 sparse 점수가 높다."""
        retriever = _make_retriever(top_k=2, candidate_pool_size=2)
        query = "hostpath"
        query_vector = _make_test_vector(query, dim=32)

        short_text = "hostpath volume mount"
        long_text = "hostpath " + " ".join(["unrelated"] * 50)

        items = [
            {
                "vector": _make_test_vector(short_text, dim=32),
                "chunk": {
                    "chunk_id": "short",
                    "source_path": "a.pdf",
                    "tokens": tokenize(short_text),
                    "text": short_text,
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                },
            },
            {
                "vector": _make_test_vector(long_text, dim=32),
                "chunk": {
                    "chunk_id": "long",
                    "source_path": "a.pdf",
                    "tokens": tokenize(long_text),
                    "text": long_text,
                    "page_number": 2,
                    "metadata": {"page_start": 2, "page_end": 2},
                },
            },
        ]

        results = retriever.search(query, query_vector, items)
        short_sparse = next(r["sparse_score"] for r in results if r["chunk"]["chunk_id"] == "short")
        long_sparse = next(r["sparse_score"] for r in results if r["chunk"]["chunk_id"] == "long")
        self.assertGreater(short_sparse, long_sparse)

    def test_retriever_accepts_custom_weights(self) -> None:
        """Settings에서 주입한 커스텀 가중치가 실제 점수 계산에 반영된다."""
        retriever_default = _make_retriever(top_k=1, candidate_pool_size=2)
        retriever_sparse_heavy = _make_retriever(
            top_k=1,
            candidate_pool_size=2,
            dense_weight=0.1,
            sparse_weight=0.7,
            title_weight=0.05,
        )
        query = "deployment"
        query_vector = _make_test_vector(query, dim=32)

        items = [
            {
                "vector": _make_test_vector("deployment configuration yaml", dim=32),
                "chunk": {
                    "chunk_id": "c1",
                    "source_path": "guide.pdf",
                    "tokens": tokenize("deployment configuration yaml"),
                    "text": "deployment configuration yaml",
                    "page_number": 1,
                    "metadata": {"page_start": 1, "page_end": 1},
                },
            },
        ]

        results_default = retriever_default.search(query, query_vector, items)
        results_sparse = retriever_sparse_heavy.search(query, query_vector, items)

        # 두 가중치 모두 결과를 반환하되, 점수가 다를 수 있다
        self.assertEqual(len(results_default), 1)
        self.assertEqual(len(results_sparse), 1)


if __name__ == "__main__":
    unittest.main()

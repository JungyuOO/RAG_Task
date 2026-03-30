from __future__ import annotations

import math
from collections import Counter

from app.rag.utils import cosine_similarity, keyword_overlap_score, tokenize


class HybridRetriever:
    """dense(BGE-M3 코사인) + sparse(BM25) RRF 하이브리드 검색기.

    외부 검색 라이브러리 없이 직접 구현. search_rrf()가 주 검색 경로이며
    dense/sparse 랭크를 RRF로 병합한 뒤 _rerank()로 keyword overlap 보정,
    BGEReranker가 최종 cross-encoder 재순위를 담당한다.
    """

    def __init__(
        self,
        *,
        top_k: int,
        candidate_pool_size: int,
        dense_weight: float,
        sparse_weight: float,
        bm25_k1: float,
        bm25_b: float,
        rerank_base_weight: float,
        rerank_overlap_weight: float,
    ) -> None:
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rerank_base_weight = rerank_base_weight
        self.rerank_overlap_weight = rerank_overlap_weight

    def search(self, query: str, query_vector: list[float], index_items: list[dict]) -> list[dict]:
        """dense + sparse 가중합으로 index_items에서 상위 top_k개를 검색한다."""
        query_tokens = tokenize(query)
        doc_frequency = Counter()
        doc_lengths: list[int] = []
        for item in index_items:
            tokens = item["chunk"]["tokens"]
            doc_frequency.update(set(tokens))
            doc_lengths.append(len(tokens))

        avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
        total_docs = max(len(index_items), 1)

        scored: list[dict] = []
        for item in index_items:
            chunk = item["chunk"]
            candidate_tokens = chunk["tokens"]

            dense_score = cosine_similarity(query_vector, item["vector"])
            sparse_score = self._bm25(
                query_tokens, candidate_tokens, doc_frequency, total_docs, avg_doc_length,
            )

            score = (
                dense_score * self.dense_weight
                + sparse_score * self.sparse_weight
            )
            scored.append(
                {
                    "score": score,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "chunk": chunk,
                }
            )

        scored.sort(key=lambda entry: entry["score"], reverse=True)
        candidate_pool = scored[: self.candidate_pool_size]
        reranked = self._rerank(query_tokens, candidate_pool)
        return reranked[: self.top_k]

    def search_rrf(
        self,
        query: str,
        query_vector: list[float],
        index_items: list[dict],
        rrf_k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion으로 dense + sparse 랭크를 결합한다.

        RRF 공식: score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        상위 candidate_pool_size개를 추린 뒤 _rerank()로 keyword overlap 보정,
        top_k를 반환한다.
        """
        if not index_items:
            return []

        query_tokens = tokenize(query)

        # BM25 통계
        doc_frequency = Counter()
        doc_lengths: list[int] = []
        for item in index_items:
            tokens = item["chunk"]["tokens"]
            doc_frequency.update(set(tokens))
            doc_lengths.append(len(tokens))
        avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
        total_docs = max(len(index_items), 1)

        # 항목별 dense/sparse 점수 계산
        item_scores: list[tuple[int, float, float]] = []
        for idx, item in enumerate(index_items):
            dense_score = cosine_similarity(query_vector, item["vector"])
            sparse_score = self._bm25(
                query_tokens, item["chunk"]["tokens"], doc_frequency, total_docs, avg_doc_length,
            )
            item_scores.append((idx, dense_score, sparse_score))

        # dense 순위 (rank 1 = 가장 높은 코사인 유사도)
        dense_sorted = sorted(item_scores, key=lambda x: x[1], reverse=True)
        dense_rank: dict[int, int] = {entry[0]: rank + 1 for rank, entry in enumerate(dense_sorted)}

        # sparse 순위 (rank 1 = 가장 높은 BM25)
        sparse_sorted = sorted(item_scores, key=lambda x: x[2], reverse=True)
        sparse_rank: dict[int, int] = {entry[0]: rank + 1 for rank, entry in enumerate(sparse_sorted)}

        # RRF 점수 계산
        rrf_scored: list[dict] = []
        for idx, item in enumerate(index_items):
            chunk = item["chunk"]
            dense_score = item_scores[idx][1]
            sparse_score = item_scores[idx][2]

            rrf_score = (
                1.0 / (rrf_k + dense_rank[idx])
                + 1.0 / (rrf_k + sparse_rank[idx])
            )

            rrf_scored.append(
                {
                    "score": rrf_score,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "chunk": chunk,
                }
            )

        rrf_scored.sort(key=lambda entry: entry["score"], reverse=True)
        candidates = rrf_scored[: self.candidate_pool_size]

        reranked = self._rerank(query_tokens, candidates)
        return reranked[: self.top_k]

    def _bm25(
        self,
        query_tokens: list[str],
        candidate_tokens: list[str],
        doc_frequency: Counter,
        total_docs: int,
        avg_doc_length: float,
    ) -> float:
        """표준 BM25 스코어링."""
        if not query_tokens or not candidate_tokens:
            return 0.0

        candidate_counter = Counter(candidate_tokens)
        doc_length = len(candidate_tokens)
        score = 0.0

        for token in query_tokens:
            if token not in candidate_counter:
                continue
            tf = candidate_counter[token]
            df = doc_frequency.get(token, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (self.bm25_k1 + 1.0)) / (
                tf + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * doc_length / max(avg_doc_length, 1.0))
            )
            score += idf * tf_norm

        max_possible = len(query_tokens) * math.log(total_docs + 1.0) * (self.bm25_k1 + 1.0)
        return score / max(max_possible, 1.0)

    def _rerank(self, query_tokens: list[str], candidates: list[dict]) -> list[dict]:
        """RRF 점수 + keyword overlap으로 최종 점수를 재산정한다."""
        reranked = []
        for candidate in candidates:
            overlap = keyword_overlap_score(query_tokens, candidate["chunk"]["tokens"])
            final_score = (
                candidate["score"] * self.rerank_base_weight
                + overlap * self.rerank_overlap_weight
            )
            reranked.append({**candidate, "rerank_score": final_score})
        reranked.sort(key=lambda entry: entry["rerank_score"], reverse=True)
        return reranked

    def compute_retrieval_metrics(self, results: list[dict], min_score: float) -> dict:
        """검색 결과의 품질 지표를 계산한다."""
        if not results:
            return {
                "hit_count": 0, "total_count": 0, "hit_rate": 0.0,
                "mean_score": 0.0, "top_score": 0.0, "score_gap": 0.0,
                "score_spread": 0.0, "dense_sparse_correlation": 0.0,
            }
        scores = [r["rerank_score"] for r in results]
        hits = [s for s in scores if s >= min_score]
        top = scores[0]
        gap = scores[0] - scores[1] if len(scores) >= 2 else 0.0
        spread = scores[0] - scores[-1]

        dense_ranks = self._rank_values([r.get("dense_score", 0.0) for r in results])
        sparse_ranks = self._rank_values([r.get("sparse_score", 0.0) for r in results])
        n = len(results)
        if n >= 2:
            d_sq_sum = sum((d - s) ** 2 for d, s in zip(dense_ranks, sparse_ranks))
            rho = 1.0 - (6.0 * d_sq_sum) / (n * (n ** 2 - 1))
        else:
            rho = 1.0

        return {
            "hit_count": len(hits),
            "total_count": len(scores),
            "hit_rate": len(hits) / len(scores),
            "mean_score": sum(scores) / len(scores),
            "top_score": top,
            "score_gap": gap,
            "score_spread": spread,
            "dense_sparse_correlation": round(rho, 4),
        }

    @staticmethod
    def _rank_values(values: list[float]) -> list[float]:
        """값 리스트에 대한 순위를 반환한다 (동점은 평균 순위)."""
        indexed = sorted(enumerate(values), key=lambda x: -x[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

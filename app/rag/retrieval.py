from __future__ import annotations

import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from app.rag.utils import cosine_similarity, keyword_overlap_score, tokenize


class HybridRetriever:
    """밀도(dense) + BM25 희소(sparse) + 제목 매칭을 결합한 하이브리드 검색기.

    외부 검색 라이브러리 없이 직접 구현하며, 가중치와 BM25 파라미터를
    외부에서 주입받아 튜닝할 수 있다.
    """

    def __init__(
        self,
        top_k: int = 6,
        candidate_pool_size: int = 14,
        dense_weight: float = 0.45,
        sparse_weight: float = 0.25,
        title_weight: float = 0.15,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        rerank_base_weight: float = 0.68,
        rerank_overlap_weight: float = 0.17,
        rerank_title_weight: float = 0.08,
        rerank_title_bonus_weight: float = 0.07,
        rerank_compact_bonus_weight: float = 0.12,
    ) -> None:
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.title_weight = title_weight
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rerank_base_weight = rerank_base_weight
        self.rerank_overlap_weight = rerank_overlap_weight
        self.rerank_title_weight = rerank_title_weight
        self.rerank_title_bonus_weight = rerank_title_bonus_weight
        self.rerank_compact_bonus_weight = rerank_compact_bonus_weight

    def search(self, query: str, query_vector: list[float], index_items: list[dict]) -> list[dict]:
        """질의 벡터와 텍스트를 사용해 index_items에서 상위 top_k개를 검색한다."""
        query_tokens = tokenize(query)
        query_compact = "".join(query_tokens)
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
            candidate_compact = "".join(candidate_tokens)
            source_name = Path(chunk["source_path"]).stem
            source_tokens = tokenize(source_name)
            source_compact = "".join(source_tokens)

            dense_score = cosine_similarity(query_vector, item["vector"])
            sparse_score = self._bm25(
                query_tokens, candidate_tokens, doc_frequency, total_docs, avg_doc_length,
            )
            title_score = keyword_overlap_score(query_tokens, source_tokens)

            title_match_bonus = 0.0
            compact_match_bonus = 0.0
            if query_compact and source_compact:
                if source_compact in query_compact or query_compact in source_compact:
                    title_match_bonus = 0.35
            if query_compact and candidate_compact and len(query_compact) >= 4:
                compact_match_bonus = self._compact_overlap_bonus(query_compact, candidate_compact)

            score = (
                dense_score * self.dense_weight
                + sparse_score * self.sparse_weight
                + title_score * self.title_weight
                + title_match_bonus
                + compact_match_bonus
            )
            scored.append(
                {
                    "score": score,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "title_score": title_score,
                    "title_match_bonus": title_match_bonus,
                    "compact_match_bonus": compact_match_bonus,
                    "chunk": chunk,
                }
            )

        scored.sort(key=lambda entry: entry["score"], reverse=True)
        reranked = self._rerank(query_tokens, scored[: self.candidate_pool_size])
        return reranked[: self.top_k]

    def _bm25(
        self,
        query_tokens: list[str],
        candidate_tokens: list[str],
        doc_frequency: Counter,
        total_docs: int,
        avg_doc_length: float,
    ) -> float:
        """표준 BM25 스코어링 — k1/b 파라미터로 term frequency 포화와 문서 길이를 정규화한다."""
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
            # 표준 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            # 표준 TF 정규화: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
            tf_norm = (tf * (self.bm25_k1 + 1.0)) / (
                tf + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * doc_length / max(avg_doc_length, 1.0))
            )
            score += idf * tf_norm

        max_possible = len(query_tokens) * math.log(total_docs + 1.0) * (self.bm25_k1 + 1.0)
        return score / max(max_possible, 1.0)

    def _rerank(self, query_tokens: list[str], candidates: list[dict]) -> list[dict]:
        """후보 풀에서 keyword overlap과 가중 조합으로 최종 점수를 재산정한다."""
        reranked = []
        for candidate in candidates:
            overlap = keyword_overlap_score(query_tokens, candidate["chunk"]["tokens"])
            final_score = (
                candidate["score"] * self.rerank_base_weight
                + overlap * self.rerank_overlap_weight
                + candidate.get("title_score", 0.0) * self.rerank_title_weight
                + candidate.get("title_match_bonus", 0.0) * self.rerank_title_bonus_weight
                + candidate.get("compact_match_bonus", 0.0) * self.rerank_compact_bonus_weight
            )
            reranked.append({**candidate, "rerank_score": final_score})
        reranked.sort(key=lambda entry: entry["rerank_score"], reverse=True)
        return reranked

    def compute_retrieval_metrics(self, results: list[dict], min_score: float = 0.12) -> dict:
        """검색 결과의 품질 지표를 계산한다.

        Args:
            results: search()가 반환한 리랭킹 결과 리스트.
            min_score: 관련성 있는 결과로 판정할 최소 rerank_score 임계값.

        Returns:
            hit_count: min_score 이상인 결과 수.
            total_count: 전체 결과 수.
            hit_rate: 적중률 (hit_count / total_count).
            mean_score: 전체 결과의 평균 rerank_score.
            top_score: 최상위 결과의 rerank_score.
            score_gap: 1위와 2위 점수 차이 (결과가 2개 미만이면 0).
            score_spread: 1위와 최하위 점수 차이.
            dense_sparse_correlation: dense와 sparse 점수 순위의 상관도 (Spearman rho 근사).
        """
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

        # dense/sparse 순위 상관도 — 둘 다 높은 순위에 합의하면 결과 신뢰도가 높다.
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

    def _compact_overlap_bonus(self, query_compact: str, candidate_compact: str) -> float:
        """질의와 후보의 연속 부분 문자열 매칭 비율로 보너스 점수를 계산한다.

        임계값 0.35: 35% 미만의 짧은 우연적 매칭(예: "는" 등 조사)은 무시.
        상한 0.32: dense(0.45) + sparse(0.25) 대비 보조적 역할을 유지하되,
        긴 구문이 정확히 일치할 때 의미 있는 부스트를 제공하는 수준.
        """
        matcher = SequenceMatcher(None, query_compact, candidate_compact)
        match = matcher.find_longest_match(0, len(query_compact), 0, len(candidate_compact))
        if match.size <= 0:
            return 0.0
        overlap_ratio = match.size / max(min(len(query_compact), len(candidate_compact)), 1)
        if overlap_ratio < 0.35:
            return 0.0
        return min(0.32 * overlap_ratio, 0.32)

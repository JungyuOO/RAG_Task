from __future__ import annotations

import math
import logging
import time
from collections import Counter

from sentence_transformers import CrossEncoder

from app.rag.utils import cosine_similarity, keyword_overlap_score, tokenize

logger = logging.getLogger("rag.retrieval")

class HybridRetriever:
    """Dense(BGE-M3 cosine) + sparse(BM25) RRF hybrid retriever."""

    def __init__(
        self,
        *,
        top_k: int,
        candidate_pool_size: int,
        bm25_k1: float,
        bm25_b: float,
        rerank_base_weight: float,
        rerank_overlap_weight: float,
        rerank_title_weight: float = 0.15,
    ) -> None:
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rerank_base_weight = rerank_base_weight
        self.rerank_overlap_weight = rerank_overlap_weight
        self.rerank_title_weight = rerank_title_weight

    def search_rrf(
        self,
        query: str,
        query_vector: list[float],
        index_items: list[dict],
        rrf_k: int = 60,
        limit: int | None = None,
        target_versions: list | None = None,
        version_map: dict | None = None,
        keyword_query: str | None = None,
    ) -> list[dict]:
        """Fuse dense and sparse ranks with RRF, then apply overlap rerank.

        limit이 지정되면 top_k 대신 해당 개수만큼 반환한다.
        cross-encoder에 넓은 후보 풀을 넘길 때 사용.

        keyword_query가 지정되면 BM25 및 keyword-overlap 계산에 이 쿼리를 사용한다.
        확장된 query는 dense 검색에만 사용되고, BM25에는 핵심 키워드만 포함된
        짧은 쿼리를 사용하여 토큰 희석을 방지한다.

        target_versions가 지정되면 해당 버전 태그에 속하는 청크만 검색 대상에 포함한다.
        version_map이 제공되면 version_id 기반으로 필터링하고, 그렇지 않으면
        청크 metadata의 version_tag를 직접 비교한다.
        """
        t_version = time.perf_counter()
        original_count = len(index_items)
        if target_versions:
            target_set = set(target_versions)
            if version_map:
                target_ids = {vid for vid, vtag in version_map.items() if vtag in target_set}
                index_items = [item for item in index_items if item["chunk"].get("version_id") in target_ids]
            else:
                index_items = [
                    item for item in index_items
                    if item["chunk"].get("metadata", {}).get("version_tag") in target_set
                ]
        logger.info(
            "[Timing][HybridRetriever.search_rrf] version_filtering=%.3fs before=%d after=%d target_versions=%s",
            time.perf_counter() - t_version,
            original_count,
            len(index_items),
            target_versions,
        )
        if not index_items:
            logger.info("[Timing][HybridRetriever.search_rrf] total=%.3fs empty_after_filter", time.perf_counter() - t_total)
            return []
        t_tokenize = time.perf_counter()
        query_tokens = tokenize(keyword_query if keyword_query else query)
        logger.info(
            "[Timing][HybridRetriever.search_rrf] tokenize=%.3fs query_tokens=%d",
            time.perf_counter() - t_tokenize,
            len(query_tokens),
        )
        t_df = time.perf_counter()
        doc_frequency = Counter()
        doc_lengths: list[int] = []
        for item in index_items:
            tokens = item["chunk"]["tokens"]
            doc_frequency.update(set(tokens))
            doc_lengths.append(len(tokens))
        avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
        total_docs = max(len(index_items), 1)
        logger.info(
            "[Timing][HybridRetriever.search_rrf] corpus_stats=%.3fs total_docs=%d avg_doc_length=%.2f",
            time.perf_counter() - t_df,
            total_docs,
            avg_doc_length,

        )
        t_score = time.perf_counter()
        item_scores: list[tuple[int, float, float]] = []
        for idx, item in enumerate(index_items):
            dense_score = cosine_similarity(query_vector, item["vector"])
            sparse_score = self._bm25(
                query_tokens,
                item["chunk"]["tokens"],
                doc_frequency,
                total_docs,
                avg_doc_length,
            )
            item_scores.append((idx, dense_score, sparse_score))
        logger.info(
            "[Timing][HybridRetriever.search_rrf] dense_sparse_scoring=%.3fs items=%d",
            time.perf_counter() - t_score,
            len(index_items),
        )
        t_sort = time.perf_counter()
        dense_sorted = sorted(item_scores, key=lambda x: x[1], reverse=True)
        dense_rank: dict[int, int] = {
            entry[0]: rank + 1 for rank, entry in enumerate(dense_sorted)
        }

        sparse_sorted = sorted(item_scores, key=lambda x: x[2], reverse=True)
        sparse_rank: dict[int, int] = {
            entry[0]: rank + 1 for rank, entry in enumerate(sparse_sorted)
        }
        logger.info("[Timing][HybridRetriever.search_rrf] ranking_sort=%.3fs", time.perf_counter() - t_sort)

        t_rrf = time.perf_counter()
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
        logger.info(
            "[Timing][HybridRetriever.search_rrf] rrf_build=%.3fs candidates=%d",
            time.perf_counter() - t_rrf,
            len(candidates),
        )
        t_rerank = time.perf_counter()
        reranked = self._rerank(query_tokens, candidates)
        logger.info(
            "[Timing][HybridRetriever.search_rrf] lightweight_rerank=%.3fs",
            time.perf_counter() - t_rerank,
        )
        result_limit = limit if limit is not None else self.top_k
        logger.info(
            "[Timing][HybridRetriever.search_rrf] total=%.3fs result_limit=%d",
            time.perf_counter() - t_total,
            result_limit,
        )
        return reranked[: result_limit]

    def _bm25(
        self,
        query_tokens: list[str],
        candidate_tokens: list[str],
        doc_frequency: Counter,
        total_docs: int,
        avg_doc_length: float,
    ) -> float:
        """Standard BM25 scoring."""
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
                tf
                + self.bm25_k1
                * (1.0 - self.bm25_b + self.bm25_b * doc_length / max(avg_doc_length, 1.0))
            )
            score += idf * tf_norm

        max_possible = len(query_tokens) * math.log(total_docs + 1.0) * (self.bm25_k1 + 1.0)
        return score / max(max_possible, 1.0)

    def _rerank(self, query_tokens: list[str], candidates: list[dict]) -> list[dict]:
        """Combine RRF score with keyword overlap and section title match for a lightweight pre-rerank."""
        reranked = []
        for candidate in candidates:
            overlap = keyword_overlap_score(query_tokens, candidate["chunk"]["tokens"])

            # Section title boosting: query keywords matched against section heading
            metadata = candidate["chunk"].get("metadata") or {}
            section_title = str(metadata.get("section_title") or "").lower()
            title_tokens = tokenize(section_title) if section_title else []
            title_overlap = keyword_overlap_score(query_tokens, title_tokens) if title_tokens else 0.0

            final_score = (
                candidate["score"] * self.rerank_base_weight
                + overlap * self.rerank_overlap_weight
                + title_overlap * self.rerank_title_weight
            )
            reranked.append({**candidate, "rerank_score": final_score, "title_score": title_overlap})
        reranked.sort(key=lambda entry: entry["rerank_score"], reverse=True)
        return reranked

    def compute_retrieval_metrics(self, results: list[dict], min_score: float) -> dict:
        """Compute retrieval quality summary metrics."""
        if not results:
            return {
                "hit_count": 0,
                "total_count": 0,
                "hit_rate": 0.0,
                "mean_score": 0.0,
                "top_score": 0.0,
                "score_gap": 0.0,
                "score_spread": 0.0,
                "dense_sparse_correlation": 0.0,
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
            rho = 1.0 - (6.0 * d_sq_sum) / (n * (n**2 - 1))
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
        """Return average ranks for a list of values, preserving ties."""
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


class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder 리랭커.

    sentence-transformers CrossEncoder를 사용해 query-passage 쌍의 관련도를
    직접 스코어링한다. 후보 목록을 받아 상위 top_k를 재순위하여 반환한다.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        t_total = time.perf_counter()
        if not candidates:
            logger.info("[Timing][BGEReranker.rerank] total=%.3fs empty_candidates", time.perf_counter() - t_total)
            return []

        t_pairs = time.perf_counter()
        pairs = [(query, c["chunk"]["text"]) for c in candidates]
        logger.info(
            "[Timing][BGEReranker.rerank] pair_build=%.3fs candidates=%d",
            time.perf_counter() - t_pairs,
            len(pairs),
        )

        t_predict = time.perf_counter()
        scores = self._model.predict(pairs)
        logger.info(
            "[Timing][BGEReranker.rerank] cross_encoder_predict=%.3fs candidates=%d",
            time.perf_counter() - t_predict,
            len(pairs),
        )

        t_merge = time.perf_counter()
        reranked = []
        for candidate, score in zip(candidates, scores, strict=False):
            reranked.append({**candidate, "rerank_score": float(score)})
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info(
            "[Timing][BGEReranker.rerank] merge_sort=%.3fs",
            time.perf_counter() - t_merge,
        )
        logger.info(
            "[Timing][BGEReranker.rerank] total=%.3fs top_k=%d",
            time.perf_counter() - t_total,
            self.top_k,
        )
        return reranked[: self.top_k]

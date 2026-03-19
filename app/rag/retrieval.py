from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from app.rag.utils import cosine_similarity, keyword_overlap_score, tokenize


class HybridRetriever:
    def __init__(self, top_k: int = 6, candidate_pool_size: int = 14) -> None:
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size

    def search(self, query: str, query_vector: list[float], index_items: list[dict]) -> list[dict]:
        query_tokens = tokenize(query)
        query_compact = "".join(query_tokens)
        doc_frequency = Counter()
        for item in index_items:
            doc_frequency.update(set(item["chunk"]["tokens"]))

        scored: list[dict] = []
        total_docs = max(len(index_items), 1)
        for item in index_items:
            chunk = item["chunk"]
            candidate_tokens = chunk["tokens"]
            candidate_compact = "".join(candidate_tokens)
            source_name = Path(chunk["source_path"]).stem
            source_tokens = tokenize(source_name)
            source_compact = "".join(source_tokens)
            dense_score = cosine_similarity(query_vector, item["vector"])
            sparse_score = self._bm25_like(query_tokens, candidate_tokens, doc_frequency, total_docs)
            title_score = keyword_overlap_score(query_tokens, source_tokens)
            title_match_bonus = 0.0
            compact_match_bonus = 0.0
            if query_compact and source_compact:
                if source_compact in query_compact or query_compact in source_compact:
                    title_match_bonus = 0.35
            if query_compact and candidate_compact and len(query_compact) >= 4:
                compact_match_bonus = self._compact_overlap_bonus(query_compact, candidate_compact)
            score = (
                dense_score * 0.45
                + sparse_score * 0.25
                + title_score * 0.15
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

    def _bm25_like(
        self,
        query_tokens: list[str],
        candidate_tokens: list[str],
        doc_frequency: Counter,
        total_docs: int,
    ) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0

        candidate_counter = Counter(candidate_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in candidate_counter:
                continue
            idf = total_docs / (1 + doc_frequency[token])
            score += candidate_counter[token] * idf
        return score / len(candidate_tokens)

    def _rerank(self, query_tokens: list[str], candidates: list[dict]) -> list[dict]:
        reranked = []
        for candidate in candidates:
            overlap = keyword_overlap_score(query_tokens, candidate["chunk"]["tokens"])
            final_score = (
                candidate["score"] * 0.68
                + overlap * 0.17
                + candidate.get("title_score", 0.0) * 0.08
                + candidate.get("title_match_bonus", 0.0) * 0.07
                + candidate.get("compact_match_bonus", 0.0) * 0.12
            )
            reranked.append({**candidate, "rerank_score": final_score})
        reranked.sort(key=lambda entry: entry["rerank_score"], reverse=True)
        return reranked

    def _compact_overlap_bonus(self, query_compact: str, candidate_compact: str) -> float:
        matcher = SequenceMatcher(None, query_compact, candidate_compact)
        match = matcher.find_longest_match(0, len(query_compact), 0, len(candidate_compact))
        if match.size <= 0:
            return 0.0
        overlap_ratio = match.size / max(min(len(query_compact), len(candidate_compact)), 1)
        if overlap_ratio < 0.35:
            return 0.0
        return min(0.32 * overlap_ratio, 0.32)

from __future__ import annotations

from sentence_transformers import CrossEncoder


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
        """후보 청크 목록을 cross-encoder 점수로 재순위하여 top_k를 반환한다.

        candidates는 {"chunk": {...}, "score": float, ...} 형태.
        반환 결과의 "rerank_score"가 cross-encoder 점수로 덮어씌워진다.
        """
        if not candidates:
            return []

        pairs = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        scored = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[: self.top_k]

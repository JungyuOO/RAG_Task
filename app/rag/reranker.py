from __future__ import annotations
import httpx


class OllamaReranker:
    """bge-reranker-v2-m3 cross-encoder를 Ollama API로 호출하는 리랭커.

    Ollama /api/embed 엔드포인트에 [query, passage] 쌍을 입력하면
    cross-encoder가 스칼라 관련도 점수([score])를 반환한다.
    후보 목록을 받아 각 청크를 개별 스코어링 후 top_k로 재순위한다.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-reranker-v2-m3",
        top_k: int = 5,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_k = top_k
        self.timeout = timeout

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """후보 청크 목록을 cross-encoder 점수로 재순위하여 top_k를 반환한다.

        candidates는 {"chunk": {...}, "vector": [...], "score": float, "rerank_score": float} 형태.
        반환 결과에 "rerank_score"가 cross-encoder 점수로 덮어씌워진다.
        """
        if not candidates:
            return []

        scored = []
        for candidate in candidates:
            text = candidate["chunk"]["text"]
            score = self._score_pair(query, text)
            scored.append({**candidate, "rerank_score": score})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[: self.top_k]

    def _score_pair(self, query: str, passage: str) -> float:
        """query-passage 쌍의 관련도 점수를 반환한다."""
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": [query, passage]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        # cross-encoder 모델은 첫 번째 임베딩이 스칼라 점수([score])
        embeddings = data.get("embeddings", [[0.0]])
        return float(embeddings[0][0]) if embeddings else 0.0

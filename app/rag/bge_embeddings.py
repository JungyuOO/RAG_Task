from __future__ import annotations
import math
import httpx


class BGEOllamaEmbedder:
    """BGE-M3 임베딩 모델을 Ollama API로 호출하는 임베더.

    Ollama /api/embed 엔드포인트를 사용한다. 출력은 1024-dim L2 정규화 벡터.
    BGE-M3는 쿼리/패시지 접두사 없이 텍스트를 그대로 입력한다.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dim = 1024

    def encode(self, text: str) -> list[float]:
        """단일 텍스트를 1024-dim L2 정규화 벡터로 인코딩한다."""
        return self.encode_batch([text])[0]

    def encode_passage(self, text: str) -> list[float]:
        """청크(passage) 텍스트를 인코딩한다. BGE-M3는 접두사 없이 encode와 동일."""
        return self.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 배치로 인코딩한다."""
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [_l2_normalize(vec) for vec in data["embeddings"]]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]

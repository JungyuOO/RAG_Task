from __future__ import annotations

import math

import httpx


class BGETEIEmbedder:
    """TEI-backed BGE embedding client."""

    _QUERY_CACHE_MAX = 256

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dim = 1024
        self._query_cache: dict[str, list[float]] = {}

    def encode(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        result = self.encode_batch([text])[0]
        if len(self._query_cache) >= self._QUERY_CACHE_MAX:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[text] = result
        return result

    def encode_passage(self, text: str) -> list[float]:
        return self.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"inputs": texts if len(texts) > 1 else texts[0]}
        response = httpx.post(
            f"{self.base_url}/embed",
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        # 단일 입력도 [[...]] 형태, 배치도 [[...], [...]] 형태라고 가정
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [_l2_normalize(vec) for vec in data]

        raise ValueError("Unexpected TEI embedding response format.")


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
from __future__ import annotations

import math

import httpx


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the configured Ollama embedding model is not available."""


class BGEOllamaEmbedder:
    """Ollama-backed BGE embedding client."""

    _QUERY_CACHE_MAX = 256

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
        # In-memory LRU cache for query vectors (process lifetime, ~256 entries)
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
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return [_l2_normalize(vec) for vec in data["embeddings"]]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            if self._is_model_not_found(exc):
                raise EmbeddingModelUnavailableError(
                    f"Ollama embedding model '{self.model}' is not available."
                ) from exc

        try:
            return [self._encode_legacy(text) for text in texts]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404 and self._is_model_not_found(exc):
                raise EmbeddingModelUnavailableError(
                    f"Ollama embedding model '{self.model}' is not available."
                ) from exc
            raise

    def _encode_legacy(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return _l2_normalize(data["embedding"])

    @staticmethod
    def _is_model_not_found(exc: httpx.HTTPStatusError) -> bool:
        try:
            payload = exc.response.json()
        except ValueError:
            return False
        message = str(payload.get("error", "")).lower()
        return "not found" in message and "model" in message


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

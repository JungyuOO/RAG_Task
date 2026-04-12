from __future__ import annotations

import math

import httpx


class EmbeddingPayloadTooLargeError(RuntimeError):
    """Raised when the TEI server rejects an embedding batch as too large."""


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
        try:
            response = httpx.post(
                f"{self.base_url}/embed",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            response_text = (exc.response.text or "").strip()
            response_lower = response_text.casefold()
            if exc.response.status_code == 413 or (
                exc.response.status_code == 400
                and "cannot be empty" not in response_lower
                and any(
                    marker in response_lower
                    for marker in (
                        "too large",
                        "payload",
                        "max",
                        "length",
                        "token",
                        "truncate",
                        "batch",
                        "input validation error",
                    )
                )
            ):
                raise EmbeddingPayloadTooLargeError(
                    "TEI rejected embedding payload: "
                    f"status={exc.response.status_code} input_count={len(texts)} "
                    f"body_preview={response_text[:300]!r}"
                ) from exc
            raise

        raw_text = response.text.strip()
        if not raw_text:
            raise ValueError(
                f"TEI returned empty body. status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type')}, "
                f"input_count={len(texts)}"
            )

        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(
                f"TEI returned non-JSON body. status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type')}, "
                f"body_preview={raw_text[:300]!r}"
            ) from e

        if isinstance(data, list) and data and isinstance(data[0], list):
            return [_l2_normalize(vec) for vec in data]

        if isinstance(data, dict) and "embeddings" in data:
            return [_l2_normalize(vec) for vec in data["embeddings"]]

        raise ValueError(f"Unexpected TEI embedding response format: {type(data)} / {str(data)[:300]}")


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

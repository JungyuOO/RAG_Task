from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import httpx


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the configured embedding model is not available."""


class EmbeddingPayloadTooLargeError(RuntimeError):
    """Raised when the TEI server rejects an embedding batch as too large."""


class BGEOllamaEmbedder:
    _QUERY_CACHE_MAX = 256

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        timeout: float = 120.0,
        batch_size: int = 16,
        batch_char_limit: int = 24000,
        parallel_workers: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = max(int(batch_size), 1)
        self.batch_char_limit = max(int(batch_char_limit), 1)
        self.parallel_workers = max(int(parallel_workers), 1)
        self.dim = 1024
        self._query_cache: dict[str, list[float]] = {}

    def encode(self, text: str, *, is_query: bool = False) -> list[float]:
        cache_key = f"q::{text}" if is_query else f"p::{text}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = f"Represent this query for retrieving relevant documents: {text}" if is_query else text
        result = self.encode_batch([payload])[0]
        if len(self._query_cache) >= self._QUERY_CACHE_MAX:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = result
        return result

    def encode_passage(self, text: str) -> list[float]:
        return self.encode(text, is_query=False)

    def encode_batch(self, texts: list[str], progress_callback=None) -> list[list[float]]:
        results: list[list[float]] = []
        batches = list(self._iter_batches(texts))
        for index, batch in enumerate(batches, start=1):
            try:
                response = httpx.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                results.extend(_l2_normalize(vec) for vec in data["embeddings"])
                if progress_callback:
                    progress_callback(index, len(batches))
                continue
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                if self._is_model_not_found(exc):
                    raise EmbeddingModelUnavailableError(
                        f"Ollama embedding model '{self.model}' is not available."
                    ) from exc

            for text in batch:
                try:
                    results.append(self._encode_legacy(text))
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404 and self._is_model_not_found(exc):
                        raise EmbeddingModelUnavailableError(
                            f"Ollama embedding model '{self.model}' is not available."
                        ) from exc
                    raise
            if progress_callback:
                progress_callback(index, len(batches))
        return results

    def _encode_legacy(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return _l2_normalize(data["embedding"])

    def _iter_batches(self, texts: list[str]):
        batch: list[str] = []
        char_count = 0
        for text in texts:
            next_text = str(text or "")
            next_len = len(next_text)
            if batch and (len(batch) >= self.batch_size or char_count + next_len > self.batch_char_limit):
                yield batch
                batch = []
                char_count = 0
            batch.append(next_text)
            char_count += next_len
        if batch:
            yield batch

    @staticmethod
    def _is_model_not_found(exc: httpx.HTTPStatusError) -> bool:
        try:
            payload = exc.response.json()
        except ValueError:
            return False
        message = str(payload.get("error", "")).lower()
        return "not found" in message and "model" in message


class BGETEIEmbedder:
    _QUERY_CACHE_MAX = 256

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        timeout: float = 120.0,
        batch_size: int = 32,
        batch_char_limit: int = 24000,
        parallel_workers: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = max(int(batch_size), 1)
        self.batch_char_limit = max(int(batch_char_limit), 1)
        self.parallel_workers = max(int(parallel_workers), 1)
        self.dim = 1024
        self._query_cache: dict[str, list[float]] = {}

    def encode(self, text: str, *, is_query: bool = False) -> list[float]:
        cache_key = f"q::{text}" if is_query else f"p::{text}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = f"Represent this query for retrieving relevant documents: {text}" if is_query else text
        result = self.encode_batch([payload])[0]
        if len(self._query_cache) >= self._QUERY_CACHE_MAX:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = result
        return result

    def encode_passage(self, text: str) -> list[float]:
        return self.encode(text, is_query=False)

    def encode_batch(self, texts: list[str], progress_callback=None) -> list[list[float]]:
        batches = list(self._iter_batches(texts))
        vectors: list[list[float]] = []
        if self.parallel_workers <= 1 or len(batches) <= 1:
            for index, batch in enumerate(batches, start=1):
                vectors.extend(self._request_batch(batch))
                if progress_callback:
                    progress_callback(index, len(batches))
            return vectors
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            for index, batch_vectors in enumerate(executor.map(self._request_batch, batches), start=1):
                vectors.extend(batch_vectors)
                if progress_callback:
                    progress_callback(index, len(batches))
        return vectors

    def _request_batch(self, texts: list[str]) -> list[list[float]]:
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
                if len(texts) == 1:
                    raise EmbeddingPayloadTooLargeError(
                        "TEI rejected embedding payload: "
                        f"status={exc.response.status_code} input_count={len(texts)} "
                        f"body_preview={response_text[:300]!r}"
                    ) from exc
                midpoint = max(len(texts) // 2, 1)
                return self._request_batch(texts[:midpoint]) + self._request_batch(texts[midpoint:])
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
        except ValueError as exc:
            raise ValueError(
                f"TEI returned non-JSON body. status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type')}, "
                f"body_preview={raw_text[:300]!r}"
            ) from exc

        if isinstance(data, list) and data and isinstance(data[0], list):
            return [_l2_normalize(vec) for vec in data]
        if isinstance(data, dict) and "embeddings" in data:
            return [_l2_normalize(vec) for vec in data["embeddings"]]
        raise ValueError(f"Unexpected TEI embedding response format: {type(data)} / {str(data)[:300]}")

    def _iter_batches(self, texts: list[str]):
        batch: list[str] = []
        char_count = 0
        for text in texts:
            next_text = str(text or "")
            next_len = len(next_text)
            if batch and (len(batch) >= self.batch_size or char_count + next_len > self.batch_char_limit):
                yield batch
                batch = []
                char_count = 0
            batch.append(next_text)
            char_count += next_len
        if batch:
            yield batch


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

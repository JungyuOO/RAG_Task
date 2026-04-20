from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from apps.api.core.llm_settings import ChatLlmSettings

logger = logging.getLogger("rag.llm")


class OpenAiCompatibleLlmClient:
    RETRYABLE_EXCEPTIONS = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
    )

    def __init__(self, settings: ChatLlmSettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def is_enabled(self) -> bool:
        return bool(self.settings.cllm_base_url.strip() and self.settings.cllm_model.strip())

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                connect=self.settings.llm_connect_timeout_seconds,
                read=self.settings.llm_read_timeout_seconds,
                write=self.settings.llm_write_timeout_seconds,
                pool=self.settings.llm_pool_timeout_seconds,
            )
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    @staticmethod
    def _timeout_for_purpose(purpose: str, default_timeout: float) -> float:
        if purpose == "router":
            return min(default_timeout, 4.0)
        if purpose in {"question_normalize", "generate", "rerank"}:
            return min(default_timeout, 6.0)
        if purpose in {"doc_synthesis", "doc_synthesis_stream", "mixed_synthesis", "mixed_synthesis_stream"}:
            return min(default_timeout, 10.0)
        return default_timeout

    @staticmethod
    def _attempts_for_purpose(purpose: str) -> int:
        if purpose in {"router", "question_normalize", "generate", "rerank"}:
            return 1
        return 2

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        purpose: str = "generate",
    ) -> str:
        if not self.is_enabled:
            raise RuntimeError("LLM client is not configured.")

        headers = {"Content-Type": "application/json"}
        if self.settings.cllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.cllm_api_key}"

        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": self.settings.llm_generate_temperature if temperature is None else temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.llm_generate_max_tokens,
            "stream": False,
        }

        timeout_s = self._timeout_for_purpose(purpose, self.settings.llm_total_timeout_seconds)
        max_attempts = self._attempts_for_purpose(purpose)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                logger.info("[llm:%s] request start attempt=%d max_tokens=%s temp=%s", purpose, attempt, payload["max_tokens"], payload["temperature"])
                async with asyncio.timeout(timeout_s):
                    response = await self._get_client().post(
                        f"{self.settings.cllm_base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = str(data["choices"][0]["message"].get("content", "") or "").strip()
                    logger.info("[llm:%s] request done chars=%d elapsed=%.2fs", purpose, len(content), time.perf_counter() - started)
                    return content
            except self.RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                logger.warning("[llm:%s] retryable error attempt=%d elapsed=%.2fs error=%s", purpose, attempt, time.perf_counter() - started, exc)
                if attempt < max_attempts:
                    await asyncio.sleep(1.25 * attempt)
                    continue
                raise RuntimeError("LLM request failed after retries.") from exc
            except TimeoutError as exc:
                last_error = exc
                logger.warning("[llm:%s] timeout elapsed=%.2fs", purpose, time.perf_counter() - started)
                raise RuntimeError("LLM request timed out.") from exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                logger.warning("[llm:%s] http error status=%d elapsed=%.2fs", purpose, exc.response.status_code, time.perf_counter() - started)
                raise RuntimeError(f"LLM request failed: HTTP {exc.response.status_code}") from exc

        if last_error is not None:
            raise RuntimeError("LLM request failed.") from last_error
        raise RuntimeError("LLM request failed.")

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        purpose: str = "stream",
    ) -> AsyncIterator[str]:
        if not self.is_enabled:
            raise RuntimeError("LLM client is not configured.")

        headers = {"Content-Type": "application/json"}
        if self.settings.cllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.cllm_api_key}"

        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": self.settings.llm_generate_temperature if temperature is None else temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.llm_generate_max_tokens,
            "stream": True,
        }
        started = time.perf_counter()
        logger.info("[llm:%s] stream start max_tokens=%s temp=%s", purpose, payload["max_tokens"], payload["temperature"])
        async with asyncio.timeout(self._timeout_for_purpose(purpose, self.settings.llm_total_timeout_seconds)):
            async with self._get_client().stream(
                "POST",
                f"{self.settings.cllm_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                first_token_at: float | None = None
                token_count = 0
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        logger.info(
                            "[llm:%s] stream done tokens=%d first_token=%.2fs total=%.2fs",
                            purpose,
                            token_count,
                            (first_token_at - started) if first_token_at is not None else -1.0,
                            time.perf_counter() - started,
                        )
                        return
                    try:
                        payload_json = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = payload_json.get("choices", [{}])[0].get("delta", {}).get("content")
                    if not delta:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        logger.info("[llm:%s] first token %.2fs", purpose, first_token_at - started)
                    token_count += 1
                    yield str(delta)


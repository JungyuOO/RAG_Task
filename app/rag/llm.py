from __future__ import annotations

import asyncio
import json
import time
import logging

from collections.abc import AsyncIterator

import httpx

from app.config import Settings

logger = logging.getLogger("rag.llm")
class LlmClient:
    """원격 LLM 엔드포인트에 httpx SSE로 스트리밍 요청을 보내는 클라이언트."""

    RETRYABLE_EXCEPTIONS = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stream_disabled_until = 0.0
        self._generate_disabled_until = 0.0
        self._last_error = ""
        self._client: httpx.AsyncClient | None = None

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

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _sleep_before_retry(self, attempt: int) -> None:
        await asyncio.sleep(min(1.5 * attempt, 5.0))

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        t_total = time.perf_counter()
        first_token_at: float | None = None
        token_count = 0
        char_count = 0

        now = time.monotonic()
        if now < self._stream_disabled_until:
            remaining = max(self._stream_disabled_until - now, 0.0)
            raise RuntimeError(
                f"LLM endpoint temporarily unavailable. Retrying after cooldown ({remaining:.1f}s remaining)."
            )

        headers = {"Content-Type": "application/json"}
        if self.settings.cllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.cllm_api_key}"

        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": self.settings.llm_stream_temperature,
            "stream": True,
        }

        max_attempts = 3
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                async with asyncio.timeout(self.settings.llm_total_timeout_seconds):
                    t_http = time.perf_counter()
                    async with self._get_client().stream(
                        "POST",
                        f"{self.settings.cllm_base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        logger.info(
                            "[Timing][LLM.stream_chat] attempt=%d connected=%.3fs",
                            attempt,
                            time.perf_counter() - t_http,
                        )
                        self._stream_disabled_until = 0.0
                        self._last_error = ""

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                logger.info(
                                    "[Timing][LLM.stream_chat] total=%.3fs first_token=%.3fs tokens=%d chars=%d",
                                    time.perf_counter() - t_total,
                                    (first_token_at - t_total) if first_token_at is not None else -1.0,
                                    token_count,
                                    char_count,
                                )
                                return
                            try:
                                chunk_payload = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            delta = chunk_payload["choices"][0]["delta"].get("content")
                            if delta:
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                    logger.info(
                                        "[Timing][LLM.stream_chat] first_token=%.3fs",
                                        first_token_at - t_total,
                                        )
                                token_count += 1
                                char_count += len(delta)
                                yield delta
                        logger.info(
                            "[Timing][LLM.stream_chat] total=%.3fs first_token=%.3fs tokens=%d chars=%d",
                            time.perf_counter() - t_total,
                            (first_token_at - t_total) if first_token_at is not None else -1.0,
                            token_count,
                            char_count,
                        )
                        return
            except TimeoutError as exc:
                last_exc = exc
                self._last_error = str(exc) or "LLM total timeout exceeded."
                if attempt < max_attempts:
                    await self._sleep_before_retry(attempt)
                    continue
                self._stream_disabled_until = time.monotonic() + max(
                    self.settings.llm_timeout_cooldown_seconds, 0.0
                )
                raise RuntimeError("LLM total timeout exceeded.") from exc
            except self.RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                self._last_error = str(exc)
                if attempt < max_attempts:
                    await self._sleep_before_retry(attempt)
                    continue
                self._stream_disabled_until = time.monotonic() + max(
                    self.settings.llm_failure_cooldown_seconds, 0.0
                )
                raise RuntimeError("LLM stream connection failed after retries.") from exc
            except (httpx.HTTPStatusError, RuntimeError) as exc:
                self._last_error = str(exc)
                self._stream_disabled_until = time.monotonic() + max(
                    self.settings.llm_failure_cooldown_seconds, 0.0
                )
                raise

        if last_exc:
            raise RuntimeError("LLM stream connection failed.") from last_exc

    @staticmethod
    def _extract_from_reasoning(reasoning: str) -> str:
        import re

        markers = ("검색쿼리:", "판정:", "대안1:", "키워드:", "확신도:", "재질문:")
        lines = reasoning.strip().splitlines()
        result_lines: list[str] = []
        capturing = False
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(m) for m in markers):
                capturing = True
            if capturing:
                result_lines.append(stripped)
        if result_lines:
            return "\n".join(result_lines)
        paragraphs = re.split(r"\n{2,}", reasoning.strip())
        return paragraphs[-1].strip() if paragraphs else ""

    async def generate(self, messages: list[dict], max_tokens: int | None = None) -> str:
        t_total = time.perf_counter()
        now = time.monotonic()
        if now < self._generate_disabled_until:
            raise RuntimeError("LLM endpoint temporarily unavailable.")

        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": self.settings.llm_generate_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.llm_generate_max_tokens,
            "stream": False,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

        gen_headers = {"Content-Type": "application/json"}
        if self.settings.cllm_api_key:
            gen_headers["Authorization"] = f"Bearer {self.settings.cllm_api_key}"

        max_attempts = 3
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                async with asyncio.timeout(self.settings.llm_total_timeout_seconds):
                    t_http = time.perf_counter()
                    response = await self._get_client().post(
                        f"{self.settings.cllm_base_url}/chat/completions",
                        headers=gen_headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    self._generate_disabled_until = 0.0
                    self._last_error = ""

                    message = data["choices"][0]["message"]
                    content = message.get("content", "") or ""

                    logger.info(
                        "[Timing][LLM.generate] attempt=%d http_total=%.3fs chars=%d",
                        attempt,
                        time.perf_counter() - t_http,
                        len(content),
                    )
                    logger.info(
                        "[Timing][LLM.generate] total=%.3fs",
                        time.perf_counter() - t_total,
                    )

                    return content.strip()
            except TimeoutError as exc:
                last_exc = exc
                self._last_error = str(exc) or "LLM timeout on generate."
                if attempt < max_attempts:
                    await self._sleep_before_retry(attempt)
                    continue
                self._generate_disabled_until = time.monotonic() + max(
                    self.settings.llm_timeout_cooldown_seconds, 0.0
                )
                raise RuntimeError("LLM timeout on generate.") from exc
            except self.RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                self._last_error = str(exc)
                if attempt < max_attempts:
                    await self._sleep_before_retry(attempt)
                    continue
                self._generate_disabled_until = time.monotonic() + max(
                    self.settings.llm_failure_cooldown_seconds, 0.0
                )
                raise RuntimeError("LLM generate connection failed after retries.") from exc
            except (httpx.HTTPStatusError, RuntimeError) as exc:
                self._last_error = str(exc)
                self._generate_disabled_until = time.monotonic() + max(
                    self.settings.llm_failure_cooldown_seconds, 0.0
                )
                raise

        if last_exc:
            raise RuntimeError("LLM generate connection failed.") from last_exc
        raise RuntimeError("LLM generate failed.")
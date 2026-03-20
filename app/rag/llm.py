from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from app.config import Settings


class LlmClient:
    """원격 LLM 엔드포인트에 httpx SSE로 스트리밍 요청을 보내는 클라이언트.

    타임아웃/실패 시 쿨다운을 적용하여 반복 요청을 방지하고,
    OpenAI 호환 /chat/completions API를 사용한다.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._disabled_until = 0.0
        self._last_error = ""

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        now = time.monotonic()
        if now < self._disabled_until:
            remaining = max(self._disabled_until - now, 0.0)
            raise RuntimeError(
                f"LLM endpoint temporarily unavailable. Retrying after cooldown ({remaining:.1f}s remaining)."
            )

        headers = {"Content-Type": "application/json"}

        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }

        timeout = httpx.Timeout(
            connect=self.settings.llm_connect_timeout_seconds,
            read=self.settings.llm_read_timeout_seconds,
            write=self.settings.llm_write_timeout_seconds,
            pool=self.settings.llm_pool_timeout_seconds,
        )

        try:
            async with asyncio.timeout(self.settings.llm_total_timeout_seconds):
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.settings.cllm_base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        self._disabled_until = 0.0
                        self._last_error = ""
                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            delta = payload["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
        except TimeoutError as exc:
            self._last_error = str(exc) or "LLM total timeout exceeded."
            self._disabled_until = time.monotonic() + max(self.settings.llm_timeout_cooldown_seconds, 0.0)
            raise RuntimeError("LLM total timeout exceeded.") from exc
        except (httpx.HTTPError, RuntimeError) as exc:
            self._last_error = str(exc)
            self._disabled_until = time.monotonic() + max(self.settings.llm_failure_cooldown_seconds, 0.0)
            raise

    async def generate(self, messages: list[dict], max_tokens: int = 256) -> str:
        """비스트리밍 LLM 호출. 질의 재작성 등 짧은 생성 작업에 사용한다."""
        now = time.monotonic()
        if now < self._disabled_until:
            raise RuntimeError("LLM endpoint temporarily unavailable.")

        timeout = httpx.Timeout(
            connect=self.settings.llm_connect_timeout_seconds,
            read=self.settings.llm_read_timeout_seconds,
            write=self.settings.llm_write_timeout_seconds,
            pool=self.settings.llm_pool_timeout_seconds,
        )
        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            async with asyncio.timeout(self.settings.llm_total_timeout_seconds):
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self.settings.cllm_base_url}/chat/completions",
                        headers={"Content-Type": "application/json"},
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    self._disabled_until = 0.0
                    self._last_error = ""
                    return data["choices"][0]["message"]["content"].strip()
        except TimeoutError as exc:
            self._last_error = str(exc) or "LLM timeout on generate."
            self._disabled_until = time.monotonic() + max(self.settings.llm_timeout_cooldown_seconds, 0.0)
            raise RuntimeError("LLM timeout on generate.") from exc
        except (httpx.HTTPError, RuntimeError) as exc:
            self._last_error = str(exc)
            self._disabled_until = time.monotonic() + max(self.settings.llm_failure_cooldown_seconds, 0.0)
            raise

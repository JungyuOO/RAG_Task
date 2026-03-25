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
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """httpx.AsyncClient를 최초 호출 시 생성하고 이후에는 재사용한다."""
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
        """AsyncClient를 닫는다."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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
            "temperature": 0.1,
            "stream": True,
        }

        try:
            async with asyncio.timeout(self.settings.llm_total_timeout_seconds):
                async with self._get_client().stream(
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

    @staticmethod
    def _extract_from_reasoning(reasoning: str) -> str:
        """Qwen3.5 reasoning 필드에서 최종 답변 형식을 추출한다.

        reasoning에 '검색쿼리:', '판정:' 등 구조화된 응답이 포함되어 있으면
        해당 부분을 추출하여 반환한다.
        """
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
        # 마지막 문단을 fallback으로 반환
        paragraphs = re.split(r"\n{2,}", reasoning.strip())
        return paragraphs[-1].strip() if paragraphs else ""

    async def generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        """비스트리밍 LLM 호출. 질의 재작성 등 짧은 생성 작업에 사용한다."""
        now = time.monotonic()
        if now < self._disabled_until:
            raise RuntimeError("LLM endpoint temporarily unavailable.")

        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        try:
            async with asyncio.timeout(self.settings.llm_total_timeout_seconds):
                response = await self._get_client().post(
                    f"{self.settings.cllm_base_url}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                self._disabled_until = 0.0
                self._last_error = ""
                message = data["choices"][0]["message"]
                content = message.get("content") or ""
                # Qwen3.5: content가 비어있으면 reasoning 필드에서 답변 추출 시도
                if not content.strip() and message.get("reasoning"):
                    content = self._extract_from_reasoning(message["reasoning"])
                return content.strip()
        except TimeoutError as exc:
            self._last_error = str(exc) or "LLM timeout on generate."
            self._disabled_until = time.monotonic() + max(self.settings.llm_timeout_cooldown_seconds, 0.0)
            raise RuntimeError("LLM timeout on generate.") from exc
        except (httpx.HTTPError, RuntimeError) as exc:
            self._last_error = str(exc)
            self._disabled_until = time.monotonic() + max(self.settings.llm_failure_cooldown_seconds, 0.0)
            raise

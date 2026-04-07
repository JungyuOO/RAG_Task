from __future__ import annotations

import json
import logging

from app.rag.llm import LlmClient

logger = logging.getLogger("rag.agent.base")


class BaseAgent:
    """LLM-backed helper base class."""

    def __init__(self, llm: LlmClient, system_prompt: str = "") -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def _render_system_prompt(self, **template_vars) -> str:
        if not self.system_prompt:
            return ""
        return self.system_prompt.format(**template_vars) if template_vars else self.system_prompt

    def _build_messages(self, user_message: str, **template_vars) -> list[dict]:
        messages: list[dict] = []
        system_prompt = self._render_system_prompt(**template_vars)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """LLM 응답에서 JSON 객체를 추출한다.

        코드 블록(```json ... ```)이나 앞뒤 텍스트가 있어도 처리한다.
        """
        if not text:
            return None
        # 1차: 그대로 파싱
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        # 2차: ```json ... ``` 또는 ``` ... ``` 블록에서 추출
        import re
        block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if block:
            try:
                return json.loads(block.group(1))
            except (json.JSONDecodeError, TypeError):
                pass
        # 3차: 첫 번째 { ... } 구간 추출
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    async def call(self, user_message: str, max_tokens: int | None = None, **template_vars) -> dict:
        messages = self._build_messages(user_message, **template_vars)
        response = await self.llm.generate(messages, max_tokens=max_tokens)
        result = self._extract_json(response)
        if result is not None:
            return result
        logger.warning("Agent JSON parse failed, returning raw")
        return {"raw": response}

    async def stream(self, user_message: str, **template_vars):
        messages = self._build_messages(user_message, **template_vars)
        async for token in self.llm.stream_chat(messages):
            yield token

    async def _generate(self, prompt: str, max_tokens: int) -> str:
        return await self.llm.generate([{"role": "user", "content": prompt}], max_tokens=max_tokens)

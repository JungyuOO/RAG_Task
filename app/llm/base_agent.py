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

    async def call(self, user_message: str, max_tokens: int | None = None, **template_vars) -> dict:
        messages = self._build_messages(user_message, **template_vars)
        response = await self.llm.generate(messages, max_tokens=max_tokens)
        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Agent JSON parse failed, returning raw")
            return {"raw": response}

    async def stream(self, user_message: str, **template_vars):
        messages = self._build_messages(user_message, **template_vars)
        async for token in self.llm.stream_chat(messages):
            yield token

    async def _generate(self, prompt: str, max_tokens: int) -> str:
        return await self.llm.generate([{"role": "user", "content": prompt}], max_tokens=max_tokens)

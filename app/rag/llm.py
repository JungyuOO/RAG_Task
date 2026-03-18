from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import Settings


class LlmClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": self.settings.cllm_model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.settings.cllm_base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
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

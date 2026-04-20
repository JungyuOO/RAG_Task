from __future__ import annotations

import json
import re
from dataclasses import dataclass

from apps.api.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient


@dataclass(slots=True)
class NormalizedQuestion:
    text: str
    language: str


class QuestionNormalizer:
    SYSTEM_PROMPT = """
Rewrite the user's question into a concise retrieval-friendly technical query.

Return JSON only:
{
  "normalized_query":"string",
  "language":"en|ko|mixed"
}

Rules:
- If the user writes in Korean or mixed Korean/English, rewrite into concise English technical wording for retrieval.
- Preserve Kubernetes/OpenShift resource names, acronyms, and commands.
- Do not answer the question.
- Do not add citations or commentary.
"""

    _KOREAN_PATTERN = re.compile(r"[가-힣]")

    def __init__(self, llm_client: OpenAiCompatibleLlmClient | None = None) -> None:
        self.llm_client = llm_client

    async def normalize(self, *, message: str, recent_turns: list[CopilotChatHistoryTurn]) -> NormalizedQuestion:
        text = str(message or "").strip()
        if not text:
            return NormalizedQuestion(text="", language="en")

        if self.llm_client is not None and self.llm_client.is_enabled and self._KOREAN_PATTERN.search(text):
            normalized = await self._normalize_with_llm(message=text, recent_turns=recent_turns)
            if normalized is not None:
                return normalized

        return NormalizedQuestion(text=text, language="ko" if self._KOREAN_PATTERN.search(text) else "en")

    async def _normalize_with_llm(
        self,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
    ) -> NormalizedQuestion | None:
        history_lines = [
            f"- {turn.role}: {turn.text}"
            for turn in recent_turns[-4:]
            if str(turn.text or "").strip()
        ]
        prompt = "\n".join(
            [
                "recent_turns:",
                *(history_lines or ["- none"]),
                f"user_message: {message}",
            ]
        )
        try:
            raw = await self.llm_client.generate(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.0,
                purpose="question_normalize",
            )
            parsed = self._extract_json(raw)
            if parsed is None:
                return None
            normalized = str(parsed.get("normalized_query") or "").strip()
            language = str(parsed.get("language") or "").strip() or "mixed"
            if not normalized:
                return None
            return NormalizedQuestion(text=normalized, language=language)
        except Exception:
            return None

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
        return None

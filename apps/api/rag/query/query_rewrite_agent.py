from __future__ import annotations

import json
from dataclasses import dataclass

from apps.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient
from apps.api.rag.query.chat_memory import build_chat_memory


@dataclass(slots=True)
class QueryRewriteDecision:
    rewritten_query: str
    allowed_source_paths: list[str]


class QueryRewriteAgent:
    SYSTEM_PROMPT = """
Rewrite the user's question into a standalone document-retrieval query.

Return JSON only:
{
  "rewritten_query":"string",
  "use_previous_doc_sources": true|false
}

Rules:
- Resolve follow-up references using recent conversation
- Prefer concise English technical wording for retrieval when the source corpus is English
- Keep important technical terms, Kubernetes resource names, acronyms, and commands
- If the question is a follow-up about the previous document answer, set use_previous_doc_sources=true
- Do not mention cluster status in the rewritten query
"""

    FOLLOWUP_MARKERS = (
        "again",
        "then",
        "that",
        "this",
        "those",
        "그거",
        "이거",
        "그 문서",
        "다시",
        "자세히",
        "이어서",
    )

    def __init__(self, llm_client: OpenAiCompatibleLlmClient | None = None) -> None:
        self.llm_client = llm_client

    async def rewrite(self, *, message: str, recent_turns: list[CopilotChatHistoryTurn]) -> QueryRewriteDecision:
        memory = build_chat_memory(recent_turns)
        if self.llm_client is not None and self.llm_client.is_enabled and self._should_use_llm(message=message, memory=memory):
            llm_result = await self._rewrite_with_llm(message=message, recent_turns=recent_turns, memory=memory)
            if llm_result is not None:
                return llm_result

        allowed = memory.last_doc_source_paths if memory.last_lane.startswith("doc") and self._looks_followup(message) else []
        return QueryRewriteDecision(rewritten_query=message.strip(), allowed_source_paths=allowed)

    async def _rewrite_with_llm(
        self,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        memory,
    ) -> QueryRewriteDecision | None:
        if self.llm_client is None:
            return None

        history_lines = [
            f"- {turn.role}: {turn.text} (lane={turn.lane or '-'})"
            for turn in recent_turns[-4:]
            if str(turn.text or "").strip()
        ]
        prompt = "\n".join(
            [
                "recent_turns:",
                *(history_lines or ["- none"]),
                f"last_doc_sources={memory.last_doc_source_paths}",
                f"user_message: {message.strip()}",
            ]
        )
        try:
            raw = await self.llm_client.generate(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=180,
                temperature=0.0,
            )
            parsed = self._extract_json(raw)
            if parsed is None:
                return None
            rewritten = str(parsed.get("rewritten_query") or "").strip() or message.strip()
            use_previous = bool(parsed.get("use_previous_doc_sources"))
            allowed = memory.last_doc_source_paths if use_previous else []
            return QueryRewriteDecision(rewritten_query=rewritten, allowed_source_paths=allowed)
        except Exception:
            return None

    @classmethod
    def _looks_followup(cls, message: str) -> bool:
        lowered = str(message or "").casefold()
        return any(marker in lowered for marker in cls.FOLLOWUP_MARKERS)

    @classmethod
    def _should_use_llm(cls, *, message: str, memory) -> bool:
        text = str(message or "").strip()
        if not text:
            return False
        if cls._looks_followup(text):
            return True
        if memory.last_lane == "mixed":
            return True
        if memory.last_lane.startswith("doc") and len(text) <= 28:
            return True
        return False

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


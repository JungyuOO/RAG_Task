from __future__ import annotations

import json
from dataclasses import dataclass

from apps.api.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient


@dataclass(slots=True)
class IntentDecision:
    lane: str
    doc_query: str = ""
    live_query: str = ""


class IntentAgent:
    DOC_STYLE_MARKERS = (
        "설명",
        "개념",
        "차이",
        "방법",
        "어떻게",
        "가이드",
        "문서",
        "공식",
        "체크사항",
        "체크리스트",
        "권장사항",
        "명령어",
        "커맨드",
        "authentication",
        "authorization",
        "rbac",
        "route",
        "what is",
        "explain",
        "difference",
        "how to",
        "guide",
        "document",
        "docs",
        "official",
    )
    RESOURCE_TERMS = (
        "pod",
        "pods",
        "파드",
        "deployment",
        "deployments",
        "service",
        "services",
        "route",
        "routes",
        "event",
        "events",
        "namespace",
        "namespaces",
        "네임스페이스",
    )
    DETAIL_TERMS = ("yaml", "manifest", "spec", "detail", "describe", "상세", "보여줘")
    LIVE_CONTROL_TERMS = (
        "보여줘",
        "목록",
        "상태",
        "현재",
        "실제 결과",
        "개수",
        "조회",
        "다시",
        "list",
        "show",
        "status",
        "current",
        "count",
        "yaml",
        "manifest",
        "spec",
    )
    FOLLOWUP_MARKERS = (
        "그거",
        "이거",
        "다시",
        "방금",
        "이어서",
        "again",
        "then",
        "that",
        "this",
    )
    COMMAND_MARKERS = ("명령어", "커맨드", "cli", "oc ", "kubectl")
    MIXED_MARKERS = (
        "같이",
        "함께",
        "실제 결과",
        "현재 결과",
        "문서 기준",
        "공식 문서 기준",
        "비교",
        "차이",
    )

    SYSTEM_PROMPT = """
You classify the user's question for a Kubernetes/OpenShift assistant and rewrite subqueries.

Return JSON only with this schema:
{
  "lane":"doc|live|mixed|needs_connection",
  "doc_query":"string",
  "live_query":"string"
}

Rules:
- doc: conceptual, procedural, manual/document grounded questions
- live: cluster state/resource inspection questions
- mixed: the user explicitly asks for both current cluster state and documentation guidance
- needs_connection: clearly live, but no active connection
- Be conservative: if unclear, choose doc
"""

    def __init__(self, llm_client: OpenAiCompatibleLlmClient | None = None) -> None:
        self.llm_client = llm_client

    async def classify(
        self,
        *,
        message: str,
        has_connection: bool,
        recent_turns: list[CopilotChatHistoryTurn],
    ) -> IntentDecision:
        rules_decision = self._classify_with_rules(
            message=message,
            has_connection=has_connection,
            recent_turns=recent_turns,
        )
        if self._should_prefer_rules(message=message, recent_turns=recent_turns, decision=rules_decision):
            return rules_decision

        if self.llm_client is not None and self.llm_client.is_enabled:
            llm_decision = await self._classify_with_llm(
                message=message,
                has_connection=has_connection,
                recent_turns=recent_turns,
            )
            if llm_decision is not None:
                return llm_decision
        return rules_decision

    async def _classify_with_llm(
        self,
        *,
        message: str,
        has_connection: bool,
        recent_turns: list[CopilotChatHistoryTurn],
    ) -> IntentDecision | None:
        if self.llm_client is None:
            return None

        history_lines = [
            f"- {turn.role}: {turn.text} (lane={turn.lane or '-'})"
            for turn in recent_turns[-4:]
            if str(turn.text or "").strip()
        ]
        user_prompt = "\n".join(
            [
                f"connection_available={str(has_connection).lower()}",
                "recent_turns:",
                *(history_lines or ["- none"]),
                f"user_message: {message.strip()}",
            ]
        )
        try:
            response = await self.llm_client.generate(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=220,
                temperature=0.0,
            )
            parsed = self._extract_json(response)
            lane = str((parsed or {}).get("lane") or "").strip()
            if lane not in {"doc", "live", "mixed", "needs_connection"}:
                return None
            if lane == "live" and not has_connection:
                lane = "needs_connection"
            doc_query = str((parsed or {}).get("doc_query") or "").strip()
            live_query = str((parsed or {}).get("live_query") or "").strip()
            if lane in {"doc", "mixed"} and not doc_query:
                doc_query = message.strip()
            if lane in {"live", "mixed", "needs_connection"} and not live_query:
                live_query = message.strip()
            return IntentDecision(lane=lane, doc_query=doc_query, live_query=live_query)
        except Exception:
            return None

    def _classify_with_rules(
        self,
        *,
        message: str,
        has_connection: bool,
        recent_turns: list[CopilotChatHistoryTurn],
    ) -> IntentDecision:
        lowered = str(message or "").casefold()

        asks_command = any(marker in lowered for marker in self.COMMAND_MARKERS)
        asks_mixed = any(marker in lowered for marker in self.MIXED_MARKERS)
        doc_signal = self._has_doc_signal(lowered)
        live_signal = self._has_live_signal(lowered)
        live_context_doc_followup = self._has_live_context_doc_followup(
            lowered=lowered,
            recent_turns=recent_turns,
        )

        if asks_command and asks_mixed and has_connection:
            return IntentDecision(lane="mixed", doc_query=message.strip(), live_query=message.strip())
        if asks_command and not asks_mixed:
            return IntentDecision(lane="doc", doc_query=message.strip())
        if doc_signal and live_context_doc_followup and has_connection:
            return IntentDecision(lane="mixed", doc_query=message.strip(), live_query=message.strip())
        if doc_signal and live_signal:
            return IntentDecision(lane="mixed", doc_query=message.strip(), live_query=message.strip())
        if doc_signal:
            return IntentDecision(lane="doc", doc_query=message.strip())
        if live_signal:
            return IntentDecision(
                lane="live" if has_connection else "needs_connection",
                live_query=message.strip(),
            )

        if self._looks_followup(lowered):
            last_lane = self._last_assistant_lane(recent_turns)
            if last_lane == "mixed":
                return IntentDecision(lane="mixed", doc_query=message.strip(), live_query=message.strip())
            if last_lane == "live":
                return IntentDecision(
                    lane="live" if has_connection else "needs_connection",
                    live_query=message.strip(),
                )
            if last_lane.startswith("doc"):
                return IntentDecision(lane="doc", doc_query=message.strip())

        return IntentDecision(lane="doc", doc_query=message.strip())

    def _has_doc_signal(self, lowered: str) -> bool:
        if any(marker in lowered for marker in self.COMMAND_MARKERS):
            return True
        return any(marker in lowered for marker in self.DOC_STYLE_MARKERS)

    def _has_live_signal(self, lowered: str) -> bool:
        if any(marker in lowered for marker in self.COMMAND_MARKERS) and not any(marker in lowered for marker in self.MIXED_MARKERS):
            return False
        has_resource_term = any(marker in lowered for marker in self.RESOURCE_TERMS)
        has_live_control = any(marker in lowered for marker in self.LIVE_CONTROL_TERMS)
        has_yaml_like = any(marker in lowered for marker in self.DETAIL_TERMS)
        has_name_like_token = any(
            ("-" in token or any(char.isdigit() for char in token))
            for token in lowered.replace("/", " ").split()
            if len(token) > 2
        )
        return has_live_control and (
            has_resource_term
            or "namespace" in lowered
            or "네임스페이스" in lowered
            or (has_yaml_like and has_name_like_token)
        )

    def _looks_followup(self, lowered: str) -> bool:
        return any(marker in lowered for marker in self.FOLLOWUP_MARKERS)

    def _should_prefer_rules(
        self,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        decision: IntentDecision,
    ) -> bool:
        lowered = str(message or "").casefold()
        if decision.lane in {"live", "mixed", "needs_connection"}:
            return True
        if any(marker in lowered for marker in self.COMMAND_MARKERS):
            return True
        if any(marker in lowered for marker in self.DETAIL_TERMS):
            return True
        if any(marker in lowered for marker in self.LIVE_CONTROL_TERMS) and any(marker in lowered for marker in self.RESOURCE_TERMS):
            return True
        if self._looks_followup(lowered) and self._last_assistant_lane(recent_turns) in {"live", "mixed"}:
            return True
        return False

    def _has_live_context_doc_followup(
        self,
        *,
        lowered: str,
        recent_turns: list[CopilotChatHistoryTurn],
    ) -> bool:
        if self._last_assistant_lane(recent_turns) not in {"live", "mixed"}:
            return False
        has_doc_anchor = any(marker in lowered for marker in ("문서", "공식", "기준", "정리", "체크리스트", "설명"))
        has_resource_anchor = any(marker in lowered for marker in self.RESOURCE_TERMS) or any(
            ("-" in token or any(char.isdigit() for char in token))
            for token in lowered.replace("/", " ").split()
            if len(token) > 2
        )
        return has_doc_anchor and has_resource_anchor

    @staticmethod
    def _last_assistant_lane(recent_turns: list[CopilotChatHistoryTurn]) -> str:
        for turn in reversed(recent_turns):
            if str(turn.role or "") == "assistant" and str(turn.lane or ""):
                lane = str(turn.lane or "").strip()
                if lane.startswith("doc"):
                    return lane
                if lane in {"live", "mixed"}:
                    return lane
        return ""

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

from __future__ import annotations

from dataclasses import dataclass

from apps.api.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient
from apps.api.rag.query.synonym_expansion import expand_acronyms


@dataclass(slots=True)
class QueryRouterDecision:
    lane: str
    search_query: str
    live_query: str = ""
    inherit_sources: bool = False
    reasoning_brief: str = ""


class QueryRouter:
    RESOURCE_MARKERS = (
        "pod",
        "pods",
        "파드",
        "deployment",
        "deployments",
        "service",
        "services",
        "route",
        "routes",
        "namespace",
        "namespaces",
        "네임스페이스",
        "event",
        "events",
        "yaml",
        "manifest",
        "spec",
    )
    LIVE_MARKERS = (
        "보여줘",
        "목록",
        "상태",
        "현재",
        "실제 결과",
        "live",
        "yaml",
        "manifest",
        "spec",
        "조회",
        "몇 개",
        "count",
        "list",
        "show",
        "current",
    )
    DOC_MARKERS = (
        "공식 문서",
        "문서 기준",
        "설명",
        "차이",
        "비교",
        "개념",
        "방법",
        "어떻게",
        "가이드",
        "체크포인트",
        "체크리스트",
        "권장사항",
        "명령어",
        "커맨드",
        "authentication",
        "authorization",
        "rbac",
        "route",
    )
    MIXED_MARKERS = (
        "같이",
        "함께",
        "실제 결과",
        "현재 결과",
        "공식 문서 기준",
        "문서 기준",
        "비교",
        "차이",
    )
    FOLLOWUP_MARKERS = (
        "다시",
        "이어서",
        "그거",
        "그 문서",
        "방금",
        "후속",
        "정리",
        "체크포인트",
        "체크리스트",
    )

    def __init__(self, *, llm_client: OpenAiCompatibleLlmClient | None = None) -> None:
        self.llm_client = llm_client

    async def decide(
        self,
        *,
        user_message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        has_connection: bool,
        last_lane: str,
        last_doc_sources: list[str],
    ) -> QueryRouterDecision:
        text = str(user_message or "").strip()
        lowered = text.casefold()

        asks_doc = any(marker in lowered for marker in self.DOC_MARKERS)
        asks_live = any(marker in lowered for marker in self.RESOURCE_MARKERS) and any(
            marker in lowered for marker in self.LIVE_MARKERS
        )
        asks_mixed = asks_live and any(marker in lowered for marker in self.MIXED_MARKERS)
        followup = any(marker in lowered for marker in self.FOLLOWUP_MARKERS)

        if asks_mixed:
            return QueryRouterDecision(
                lane="mixed" if has_connection else "doc",
                search_query=self._search_query(text),
                live_query=text,
                inherit_sources=bool(last_doc_sources),
                reasoning_brief="explicit mixed request",
            )

        if asks_live and not asks_doc:
            return QueryRouterDecision(
                lane="live" if has_connection else "needs_connection",
                search_query=text,
                live_query=text,
                reasoning_brief="explicit live request",
            )

        if asks_doc and not asks_live:
            return QueryRouterDecision(
                lane="doc",
                search_query=self._search_query(text),
                inherit_sources=followup and bool(last_doc_sources),
                reasoning_brief="explicit doc request",
            )

        if asks_doc and asks_live:
            return QueryRouterDecision(
                lane="mixed" if has_connection else "doc",
                search_query=self._search_query(text),
                live_query=text,
                inherit_sources=bool(last_doc_sources),
                reasoning_brief="doc + live markers",
            )

        if followup:
            if last_lane.startswith("doc"):
                return QueryRouterDecision(
                    lane="doc",
                    search_query=self._search_query(text),
                    inherit_sources=bool(last_doc_sources),
                    reasoning_brief="doc follow-up",
                )
            if last_lane == "live":
                return QueryRouterDecision(
                    lane="live" if has_connection else "needs_connection",
                    search_query=text,
                    live_query=text,
                    reasoning_brief="live follow-up",
                )
            if last_lane == "mixed":
                return QueryRouterDecision(
                    lane="mixed" if has_connection else "doc",
                    search_query=self._search_query(text),
                    live_query=text,
                    inherit_sources=bool(last_doc_sources),
                    reasoning_brief="mixed follow-up",
                )

        return QueryRouterDecision(
            lane="doc",
            search_query=self._search_query(text),
            reasoning_brief="default doc fallback",
        )

    @staticmethod
    def _search_query(text: str) -> str:
        enriched = text
        if "명령어" in text or "커맨드" in text:
            enriched = f"{enriched} command oc cli kubectl"
        if "route" in text.casefold() or "라우트" in text:
            enriched = f"{enriched} ingress route networking"
        if "authentication" in text.casefold() or "authorization" in text.casefold() or "rbac" in text.casefold():
            enriched = f"{enriched} authentication authorization rbac"
        return expand_acronyms(enriched.strip())

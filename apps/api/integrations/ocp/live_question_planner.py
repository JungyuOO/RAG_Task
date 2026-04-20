from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.api.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient
from apps.api.rag.query.chat_memory import build_chat_memory
from apps.api.rag.query.intent_agent import IntentAgent

_FOLLOWUP_MARKERS = (
    "that",
    "this",
    "those",
    "again",
    "more detail",
    "follow up",
    "그거",
    "그것",
    "이거",
    "이것",
    "다시",
    "방금",
    "이어서",
    "좀 더",
)


class LiveQuestionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Literal["resource_yaml", "resource_list", "overview", "doc_compare", "resource_events", "namespace_list", "resource_relations", "resource_health"]
    resource: str = ""
    resource_name: str = ""
    namespace: str = ""
    doc_topic: str = ""
    operations: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class LegacyRuleIntent:
    DETAIL_MARKERS = ("yaml", "manifest", "detail", "details", "describe", "spec", "show me", "상세")
    DOC_COMPARE_MARKERS = ("official", "docs", "documentation", "문서", "공식")
    DIFF_MARKERS = ("difference", "diff", "compare", "차이", "비교")
    RELATION_MARKERS = ("연결", "관계", "매칭", "mapping", "bound to", "connected to", "related service")
    HEALTH_MARKERS = ("왜", "뭐가 문제", "문제", "이상", "health", "healthy", "fix", "수정", "개선", "진단")
    STOP_TOKENS = {
        "get",
        "show",
        "the",
        "for",
        "of",
        "and",
        "show",
        "list",
        "overview",
        "summary",
        "current",
        "status",
        "count",
        "cluster",
        "namespace",
        "namespaces",
        "yaml",
        "manifest",
        "detail",
        "details",
        "describe",
        "spec",
        "pods",
        "pod",
        "deployments",
        "deployment",
        "services",
        "service",
        "routes",
        "route",
        "events",
        "event",
        "official",
        "docs",
        "documentation",
        "compare",
        "difference",
        "diff",
        "live",
        "문서",
        "공식",
        "보여줘",
        "알려줘",
        "조회",
        "목록",
    }

    @classmethod
    def plan(
        cls,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        last_live_resources: list[str],
        last_namespace: str,
        last_doc_sources: list[str],
    ) -> LiveQuestionPlan:
        del last_doc_sources
        lowered = str(message or "").casefold()

        if cls._is_namespace_request(lowered):
            return LiveQuestionPlan(
                intent="namespace_list",
                namespace="",
                operations=["list_namespaces"],
                confidence=0.96,
            )

        resource = cls._detect_resource(lowered)
        resource_name = cls._extract_resource_name(lowered, resource)
        wants_detail = any(marker in lowered for marker in cls.DETAIL_MARKERS)
        wants_compare = cls._wants_doc_compare(lowered)
        namespace = last_namespace or ""

        if wants_detail and resource:
            explicit_name = cls._extract_named_token(lowered)
            if explicit_name:
                return LiveQuestionPlan(
                    intent="resource_yaml",
                    resource=resource,
                    resource_name=explicit_name,
                    namespace=namespace,
                    operations=["get_resource_yaml"],
                    confidence=0.98,
                )

        if any(marker in lowered for marker in ("overview", "summary", "cluster", "개요", "요약")) and not resource:
            return LiveQuestionPlan(
                intent="overview",
                resource="",
                namespace=namespace,
                operations=["get_overview"],
                confidence=0.9,
            )

        if not resource:
            if any(marker in lowered for marker in ("event", "events", "warning", "이벤트")):
                resource = "events"
            else:
                resource = cls._resource_from_recent_assistant(recent_turns)

        if (cls._looks_followup(lowered) or wants_detail or wants_compare) and not resource_name and last_live_resources:
            resource_name = last_live_resources[0]

        if wants_compare:
            return LiveQuestionPlan(
                intent="doc_compare",
                resource=resource or "deployments",
                resource_name=resource_name,
                namespace=namespace,
                doc_topic=cls._build_doc_topic(message=message, resource=resource or "deployments", resource_name=resource_name),
                operations=["get_resource_yaml", "search_official_docs", "compare_live_with_docs"],
                confidence=0.82 if resource_name else 0.68,
            )

        if cls._wants_relations(lowered) and (resource or resource_name):
            return LiveQuestionPlan(
                intent="resource_relations",
                resource=resource or "pods",
                resource_name=resource_name,
                namespace=namespace,
                operations=["resolve_resource_relations"],
                confidence=0.78,
            )

        if cls._wants_health(lowered) and resource_name:
            return LiveQuestionPlan(
                intent="resource_health",
                resource=resource or "deployments",
                resource_name=resource_name,
                namespace=namespace,
                operations=["get_resource_health_summary", "search_official_docs", "compare_live_with_docs"],
                confidence=0.8,
            )

        if resource == "events":
            return LiveQuestionPlan(
                intent="resource_events",
                resource="events",
                resource_name=resource_name,
                namespace=namespace,
                operations=["list_resources"],
                confidence=0.9,
            )

        if wants_detail and resource_name:
            return LiveQuestionPlan(
                intent="resource_yaml",
                resource=resource or "pods",
                resource_name=resource_name,
                namespace=namespace,
                operations=["get_resource_yaml"],
                confidence=0.92,
            )

        if resource:
            return LiveQuestionPlan(
                intent="resource_list",
                resource=resource,
                resource_name=resource_name,
                namespace=namespace,
                operations=["list_resources"],
                confidence=0.88,
            )

        return LiveQuestionPlan(
            intent="overview",
            resource="",
            namespace=namespace,
            operations=["get_overview"],
            confidence=0.55,
        )

    @classmethod
    def _is_namespace_request(cls, lowered: str) -> bool:
        if not any(marker in lowered for marker in ("namespace", "namespaces", "네임스페이스")):
            return False
        return not any(
            resource in lowered
            for resource in ("pod", "pods", "deployment", "deployments", "service", "services", "route", "routes", "event", "events")
        )

    @classmethod
    def _wants_doc_compare(cls, lowered: str) -> bool:
        return any(marker in lowered for marker in cls.DOC_COMPARE_MARKERS) and any(
            marker in lowered for marker in cls.DIFF_MARKERS
        )

    @classmethod
    def _wants_relations(cls, lowered: str) -> bool:
        return any(marker in lowered for marker in cls.RELATION_MARKERS)

    @classmethod
    def _wants_health(cls, lowered: str) -> bool:
        return any(marker in lowered for marker in cls.HEALTH_MARKERS)

    @staticmethod
    def _looks_followup(lowered: str) -> bool:
        return any(marker in lowered for marker in _FOLLOWUP_MARKERS)

    @staticmethod
    def _detect_resource(lowered: str) -> str:
        if "event" in lowered or "warning" in lowered or "이벤트" in lowered:
            return "events"
        if "deployment" in lowered or "deployments" in lowered:
            return "deployments"
        if "service" in lowered or "services" in lowered:
            return "services"
        if "route" in lowered or "routes" in lowered:
            return "routes"
        if "pod" in lowered or "pods" in lowered or "파드" in lowered:
            return "pods"
        return ""

    @classmethod
    def _extract_resource_name(cls, lowered: str, resource: str) -> str:
        explicit_name = cls._extract_named_token(lowered)
        if explicit_name:
            return explicit_name

        detail_name_patterns = [
            r"([a-z0-9][a-z0-9._-]*[a-z0-9])\s+(?:yaml|manifest|spec)\b",
            r"([a-z0-9][a-z0-9._-]*[a-z0-9])\s+보여줘",
        ]
        for pattern in detail_name_patterns:
            matched = re.search(pattern, lowered)
            if matched:
                candidate = matched.group(1).strip()
                if candidate and candidate not in cls.STOP_TOKENS:
                    return candidate

        explicit_patterns = {
            "pods": [
                r"([a-z0-9][a-z0-9._-]*)\s+pod\b",
                r"([a-z0-9][a-z0-9._-]*)\s+pods\b",
                r"pod\s+([a-z0-9][a-z0-9._-]*)",
                r"pods\s+([a-z0-9][a-z0-9._-]*)",
            ],
            "deployments": [
                r"([a-z0-9][a-z0-9._-]*)\s+deployment\b",
                r"([a-z0-9][a-z0-9._-]*)\s+deployments\b",
                r"deployment\s+([a-z0-9][a-z0-9._-]*)",
                r"deployments\s+([a-z0-9][a-z0-9._-]*)",
            ],
            "services": [
                r"([a-z0-9][a-z0-9._-]*)\s+service\b",
                r"([a-z0-9][a-z0-9._-]*)\s+services\b",
                r"service\s+([a-z0-9][a-z0-9._-]*)",
                r"services\s+([a-z0-9][a-z0-9._-]*)",
            ],
            "routes": [
                r"([a-z0-9][a-z0-9._-]*)\s+route\b",
                r"([a-z0-9][a-z0-9._-]*)\s+routes\b",
                r"route\s+([a-z0-9][a-z0-9._-]*)",
                r"routes\s+([a-z0-9][a-z0-9._-]*)",
            ],
            "events": [
                r"([a-z0-9][a-z0-9._-]*)\s+event\b",
                r"([a-z0-9][a-z0-9._-]*)\s+events\b",
                r"event\s+([a-z0-9][a-z0-9._-]*)",
                r"events\s+([a-z0-9][a-z0-9._-]*)",
            ],
        }
        for pattern in explicit_patterns.get(resource, []):
            matched = re.search(pattern, lowered)
            if matched:
                candidate = matched.group(1).strip()
                if candidate and candidate not in cls.STOP_TOKENS:
                    return candidate

        quoted = re.findall(r"[`'\"]([^`'\"]+)[`'\"]", lowered)
        for candidate in quoted:
            normalized = candidate.strip()
            if normalized and normalized not in cls.STOP_TOKENS:
                return normalized

        tokens = re.findall(r"[a-z0-9][a-z0-9._-]*", lowered)
        ranked = [
            token
            for token in tokens
            if token not in cls.STOP_TOKENS and ("-" in token or any(char.isdigit() for char in token))
        ]
        return ranked[-1] if ranked else ""

    @staticmethod
    def _extract_named_token(lowered: str) -> str:
        tokens = re.findall(r"[a-z0-9][a-z0-9._-]*", lowered)
        candidates = [
            token
            for token in tokens
            if ("-" in token or any(char.isdigit() for char in token))
            and token not in {"yaml", "manifest", "spec"}
            and token not in LegacyRuleIntent.STOP_TOKENS
        ]
        return candidates[0] if candidates else ""

    @classmethod
    def _resource_from_recent_assistant(cls, recent_turns: list[CopilotChatHistoryTurn]) -> str:
        for turn in reversed(recent_turns):
            if str(turn.role or "") != "assistant":
                continue
            kind = cls._detect_resource(str(turn.text or "").casefold())
            if kind:
                return kind
        return ""

    @staticmethod
    def _build_doc_topic(*, message: str, resource: str, resource_name: str) -> str:
        resource_hint = f"{resource} {resource_name}".strip()
        return f"{message.strip()} {resource_hint} yaml manifest configuration".strip()


class LiveQuestionPlanner:
    SYSTEM_PROMPT = """
You plan deterministic live OpenShift/Kubernetes tool calls.

Return JSON only with this schema:
{
  "intent": "resource_yaml|resource_list|overview|doc_compare|resource_events|namespace_list|resource_relations|resource_health",
  "resource": "pods|deployments|services|routes|events|",
  "resource_name": "string",
  "namespace": "string",
  "doc_topic": "string",
  "operations": ["list_resources|get_resource_yaml|get_overview|list_namespaces|search_official_docs|compare_live_with_docs|resolve_resource_relations|get_resource_health_summary"],
  "confidence": 0.0
}

Rules:
- resource_yaml: user wants YAML/manifest/spec for a concrete live resource
- resource_list: user wants a live resource list
- overview: user wants cluster or namespace overview
- doc_compare: user wants current live resource state compared with official docs
- resource_events: user wants live events/warnings
- namespace_list: user wants namespaces
- resource_relations: user wants relations between resources like pod to service
- resource_health: user wants diagnosis or fix recommendations for a concrete resource
- Use recent_turns and last live memory for follow-up questions
- Keep operations deterministic and minimal
- Confidence must be 0.0 to 1.0
"""

    def __init__(self, *, llm_client: OpenAiCompatibleLlmClient | None = None, confidence_threshold: float = 0.65) -> None:
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    async def plan(
        self,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn] | None = None,
        last_live_resources: list[str] | None = None,
        last_namespace: str = "",
        last_doc_sources: list[str] | None = None,
    ) -> LiveQuestionPlan:
        turns = recent_turns or []
        memory = build_chat_memory(turns)
        resources = list(last_live_resources or memory.last_live_resource_names)
        namespace = last_namespace or memory.last_live_namespace
        doc_sources = list(last_doc_sources or memory.last_doc_source_paths)
        lowered = str(message or "").casefold()

        forced = self._forced_plan(
            lowered=lowered,
            message=message,
            last_live_resources=resources,
            last_namespace=namespace,
        )
        if forced is not None:
            return forced

        llm_plan = await self._plan_with_llm(
            message=message,
            recent_turns=turns,
            last_live_resources=resources,
            last_namespace=namespace,
            last_doc_sources=doc_sources,
        )
        if llm_plan is not None and llm_plan.confidence >= self.confidence_threshold:
            return self._postprocess_plan(
                llm_plan,
                message=message,
                recent_turns=turns,
                last_live_resources=resources,
                last_namespace=namespace,
            )

        fallback = LegacyRuleIntent.plan(
            message=message,
            recent_turns=turns,
            last_live_resources=resources,
            last_namespace=namespace,
            last_doc_sources=doc_sources,
        )
        return self._postprocess_plan(
            fallback,
            message=message,
            recent_turns=turns,
            last_live_resources=resources,
            last_namespace=namespace,
        )

    def _forced_plan(
        self,
        *,
        lowered: str,
        message: str,
        last_live_resources: list[str],
        last_namespace: str,
    ) -> LiveQuestionPlan | None:
        resource = LegacyRuleIntent._detect_resource(lowered)
        explicit_name = LegacyRuleIntent._extract_named_token(lowered)
        resource_name = explicit_name or LegacyRuleIntent._extract_resource_name(lowered, resource)
        wants_compare = LegacyRuleIntent._wants_doc_compare(lowered)
        wants_detail = any(marker in lowered for marker in LegacyRuleIntent.DETAIL_MARKERS)
        negative_detail = any(phrase in lowered for phrase in ("yaml 말고", "manifest 말고", "spec 말고", "상세 말고"))
        explicit_list = any(marker in lowered for marker in ("목록", "조회", "list", "show")) and (
            not wants_detail or negative_detail or "목록만" in lowered or "목록으로" in lowered
        )

        if wants_compare and (resource or resource_name):
            return LiveQuestionPlan(
                intent="doc_compare",
                resource=resource or "deployments",
                resource_name=resource_name or (last_live_resources[0] if last_live_resources else ""),
                namespace=last_namespace,
                doc_topic=LegacyRuleIntent._build_doc_topic(
                    message=message,
                    resource=resource or "deployments",
                    resource_name=resource_name or (last_live_resources[0] if last_live_resources else ""),
                ),
                operations=["get_resource_yaml", "search_official_docs", "compare_live_with_docs"],
                confidence=0.99,
            )

        if explicit_list and resource:
            return LiveQuestionPlan(
                intent="resource_list",
                resource=resource,
                resource_name="",
                namespace=last_namespace,
                operations=["list_resources"],
                confidence=0.99,
            )

        if wants_detail and (resource_name or last_live_resources):
            return LiveQuestionPlan(
                intent="resource_yaml",
                resource=resource or "pods",
                resource_name=resource_name or last_live_resources[0],
                namespace=last_namespace,
                operations=["get_resource_yaml"],
                confidence=0.99,
            )

        return None

    async def _plan_with_llm(
        self,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        last_live_resources: list[str],
        last_namespace: str,
        last_doc_sources: list[str],
    ) -> LiveQuestionPlan | None:
        if self.llm_client is None or not self.llm_client.is_enabled:
            return None

        history_lines = [
            f"- {turn.role}: {turn.text} (lane={turn.lane or '-'}, namespace={turn.namespace or '-'}, resources={turn.resource_names or []})"
            for turn in recent_turns[-4:]
            if str(turn.text or "").strip()
        ]
        prompt = "\n".join(
            [
                f"user_message: {message.strip()}",
                f"last_namespace: {last_namespace or '-'}",
                f"last_live_resources: {last_live_resources}",
                f"last_doc_sources: {last_doc_sources}",
                "recent_turns:",
                *(history_lines or ["- none"]),
            ]
        )
        try:
            raw = await self.llm_client.generate(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=260,
                temperature=0.0,
                purpose="router",
            )
            parsed = IntentAgent._extract_json(raw) or {}
            return self._coerce_llm_plan(parsed)
        except Exception:
            return None

    def _coerce_llm_plan(self, payload: dict) -> LiveQuestionPlan | None:
        if not isinstance(payload, dict):
            return None
        intent = str(payload.get("intent") or "").strip()
        if intent not in {"resource_yaml", "resource_list", "overview", "doc_compare", "resource_events", "namespace_list", "resource_relations", "resource_health"}:
            return None
        operations = payload.get("operations")
        if not isinstance(operations, list) or not operations:
            operations = self._default_operations(intent)
        try:
            return LiveQuestionPlan.model_validate(
                {
                    "intent": intent,
                    "resource": self._normalize_resource(str(payload.get("resource") or "").strip()),
                    "resource_name": str(payload.get("resource_name") or "").strip(),
                    "namespace": str(payload.get("namespace") or "").strip(),
                    "doc_topic": str(payload.get("doc_topic") or "").strip(),
                    "operations": [str(operation).strip() for operation in operations if str(operation).strip()],
                    "confidence": max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
                }
            )
        except (ValidationError, TypeError, ValueError):
            return None

    def _postprocess_plan(
        self,
        plan: LiveQuestionPlan,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        last_live_resources: list[str],
        last_namespace: str,
    ) -> LiveQuestionPlan:
        normalized = plan.model_copy(deep=True)
        normalized.resource = self._normalize_resource(normalized.resource)
        if not normalized.namespace:
            normalized.namespace = last_namespace
        if not normalized.operations:
            normalized.operations = self._default_operations(normalized.intent)

        lowered = str(message or "").casefold()
        if normalized.intent in {"resource_yaml", "doc_compare"} and not normalized.resource_name and last_live_resources:
            if any(marker in lowered for marker in _FOLLOWUP_MARKERS) or any(
                marker in lowered for marker in LegacyRuleIntent.DETAIL_MARKERS + LegacyRuleIntent.DIFF_MARKERS
            ):
                normalized.resource_name = last_live_resources[0]
        if not normalized.resource:
            normalized.resource = LegacyRuleIntent._detect_resource(lowered) or LegacyRuleIntent._resource_from_recent_assistant(recent_turns)
        if normalized.intent == "resource_events":
            normalized.resource = "events"
            normalized.operations = ["list_resources"]
        if normalized.intent == "namespace_list":
            normalized.resource = ""
            normalized.namespace = ""
            normalized.operations = ["list_namespaces"]
        if normalized.intent == "overview":
            normalized.resource = ""
            normalized.operations = ["get_overview"]
        if normalized.intent == "resource_list" and not normalized.resource:
            normalized.resource = "pods"
        if normalized.intent == "resource_relations":
            normalized.operations = ["resolve_resource_relations"]
            if not normalized.resource:
                normalized.resource = "pods"
        if normalized.intent == "resource_health":
            normalized.operations = ["get_resource_health_summary", "search_official_docs", "compare_live_with_docs"]
            if not normalized.resource:
                normalized.resource = "deployments"
        if normalized.intent == "resource_yaml":
            normalized.operations = ["get_resource_yaml"]
        if normalized.intent == "doc_compare":
            normalized.operations = ["get_resource_yaml", "search_official_docs", "compare_live_with_docs"]
            if not normalized.doc_topic:
                normalized.doc_topic = LegacyRuleIntent._build_doc_topic(
                    message=message,
                    resource=normalized.resource or "deployments",
                    resource_name=normalized.resource_name,
                )
        return normalized

    @staticmethod
    def _normalize_resource(resource: str) -> str:
        mapping = {
            "pod": "pods",
            "pods": "pods",
            "deployment": "deployments",
            "deployments": "deployments",
            "service": "services",
            "services": "services",
            "route": "routes",
            "routes": "routes",
            "event": "events",
            "events": "events",
        }
        return mapping.get(str(resource or "").casefold(), "")

    @staticmethod
    def _default_operations(intent: str) -> list[str]:
        if intent == "namespace_list":
            return ["list_namespaces"]
        if intent == "overview":
            return ["get_overview"]
        if intent == "resource_yaml":
            return ["get_resource_yaml"]
        if intent == "doc_compare":
            return ["get_resource_yaml", "search_official_docs", "compare_live_with_docs"]
        if intent == "resource_relations":
            return ["resolve_resource_relations"]
        if intent == "resource_health":
            return ["get_resource_health_summary", "search_official_docs", "compare_live_with_docs"]
        return ["list_resources"]


from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

from app.rag.utils import normalize_query_keywords, normalize_text


@dataclass(slots=True)
class OcpQueryPlan:
    mode: str
    resources: list[str] = field(default_factory=list)
    namespace: str = ""
    target_name: str = ""
    candidate_names: list[str] = field(default_factory=list)
    pattern: str = ""
    warning_only: bool = False
    status_check: bool = False
    followup: bool = False
    parse_strategy: str = "rule_fastpath"
    confidence: float = 0.0

    @property
    def resource(self) -> str:
        return self.resources[0] if self.resources else ""

    def to_dict(self) -> dict:
        return asdict(self)


class RuleFirstOcpPlanner:
    ALL_RESOURCES = ("pods", "deployments", "services", "routes", "events")
    RESOURCE_ALIASES = {
        "pods": ("pod", "pods", "파드"),
        "deployments": ("deployment", "deployments", "디플로이먼트", "배포", "deploy"),
        "services": ("service", "services", "서비스"),
        "routes": ("route", "routes", "라우트"),
        "events": ("event", "events", "이벤트"),
    }
    EXPLANATION_MARKERS = ("설명", "차이", "difference", "가이드", "guide", "절차", "what is", "how do", "이란", "란 무엇")
    GUIDE_MARKERS = ("명령어", "커맨드", "cli", "kubectl", "oc ", "방법", "생성할 때", "확인해야", "기본적으로", "보통", "관계")
    OPERATION_MARKERS = (
        "현재", "지금", "몇 개", "몇개", "개수", "갯수", "목록", "보여", "show", "list", "상태", "running",
        "warning", "yaml", "manifest", "연결", "host", "target", "요약", "같이", "함께", "동시에",
    )
    LIVE_STATUS_MARKERS = (
        "현재", "지금", "실제", "결과", "상태", "같이", "함께", "동시에",
        "보여줘", "보여", "요약", "몇 개", "몇개", "warning", "show", "list", "count",
    )
    FOLLOWUP_MARKERS = ("그중", "그 ", "그거", "그 yaml", "그 pod", "그 service", "해당", "방금", "아까", "that", "those", "it")
    YAML_MARKERS = ("yaml", "manifest")
    CONNECTION_MARKERS = ("연결", "연결돼", "연결된", "host", "target", "route")
    WARNING_MARKERS = ("warning", "경고")
    STATUS_CHECK_MARKERS = ("정상", "ready", "running", "상태", "health")
    COMPLEX_QUERY_MARKERS = ("같이", "함께", "동시에", "and", "plus", "비교")
    LIVE_RUNTIME_HINTS = ("ocp", "api", "namespace", "네임스페이스", "실제", "현재", "지금", "결과")
    DOC_TO_LIVE_BRIDGE_HINTS = ("어떻게", "확인해", "확인하지", "명령어", "command", "cli", "보려면")
    STOPWORDS = {
        "현재", "지금", "상태", "요약", "정리", "보여줘", "보여", "알려줘", "알려", "뭐야", "무엇", "리소스", "요소",
        "그중", "그거", "해당", "다시", "이름", "관련", "있는", "되어있는", "되어", "된", "기준", "namespace", "네임스페이스",
        "그럼", "내", "쪽", "ocp", "api", "어떻게",
        "warning", "이벤트", "event", "events", "yaml", "manifest", "show", "list", "count", "many", "how",
        "route", "routes", "service", "services", "pod", "pods", "deployment", "deployments",
        "몇", "몇개", "몇개야", "몇개지", "몇 개", "개", "개야", "갯수", "개수", "같이", "함께", "동시에", "그리고",
        "확인", "확인하", "보려면", "무슨", "어떤", "명령어", "커맨드", "명령어랑", "결과", "실제",
    }

    def __init__(self, *, ocp_api_client: Any, llm_client: Any | None = None) -> None:
        self.ocp_api_client = ocp_api_client
        self.llm_client = llm_client

    async def build_plan(self, user_message: str, topic_state: dict | None) -> OcpQueryPlan | None:
        message = normalize_text(user_message)
        lowered = message.casefold()
        topic_state = topic_state or {}
        if not message:
            return None
        if self._is_explanation_request(lowered) and not self.is_mixed_request(message, topic_state):
            return None
        rule_plan = self._build_rule_plan(message, topic_state)
        if rule_plan is None or self._should_use_rule_fastpath(rule_plan, lowered):
            return rule_plan
        fallback = await self._build_agent_fallback_plan(message, topic_state, rule_plan)
        return fallback or rule_plan

    def is_mixed_request(self, user_message: str, topic_state: dict | None) -> bool:
        message = normalize_text(user_message)
        lowered = message.casefold()
        topic_state = topic_state or {}
        if not message:
            return False
        has_live_context = bool(topic_state.get("last_ocp_result_items") or topic_state.get("last_ocp_resource"))
        if self._mentions_official_doc(lowered) and has_live_context:
            return True
        if not self._is_explanation_request(lowered):
            return False
        rule_plan = self._build_rule_plan(message, topic_state)
        if rule_plan is None:
            return False
        return self._has_live_runtime_signal(message, topic_state, rule_plan)

    def _build_rule_plan(self, user_message: str, topic_state: dict) -> OcpQueryPlan | None:
        lowered = user_message.casefold()
        explicit_resources = self._detect_resources(lowered)
        followup = any(marker in lowered for marker in self.FOLLOWUP_MARKERS)
        has_status_signal = any(marker in lowered for marker in self.OPERATION_MARKERS)
        last_resource = str(topic_state.get("last_ocp_resource") or "")
        inherited_resources = [
            str(value).strip().lower()
            for value in (topic_state.get("last_explicit_resources") or [])
            if value
        ]
        live_runtime_followup = any(marker in lowered for marker in self.LIVE_RUNTIME_HINTS)
        if not explicit_resources and not followup and not has_status_signal and not last_resource:
            return None
        namespace = self._extract_namespace(user_message) or str(topic_state.get("last_namespace") or "") or getattr(self.ocp_api_client, "default_namespace", "")
        resources = explicit_resources or ([last_resource] if last_resource else [])
        if not resources and has_status_signal and inherited_resources and live_runtime_followup:
            resource = inherited_resources[0]
            if resource and not resource.endswith("s"):
                resource = f"{resource}s"
            resources = [resource] if resource else []
        pattern = self._extract_pattern(user_message, namespace, resources)
        if followup and not pattern:
            pattern = str(topic_state.get("last_ocp_filter_keyword") or "")
        has_ocp_runtime_context = bool(last_resource or topic_state.get("last_ocp_result_items"))
        if self._is_yaml_request(lowered) and not resources and not last_resource:
            return None
        if self._is_yaml_request(lowered) and not has_ocp_runtime_context and not live_runtime_followup:
            return None
        if self._is_yaml_request(lowered):
            resources = [resource for resource in (resources or [str((topic_state.get("last_ocp_result_items") or [{}])[0].get("resource") or "")]) if resource]
            resource = resources[0] if resources else ""
            candidate_names = self._candidate_names(topic_state, resource)
            target_name = self._resolve_target_name(user_message, resource, candidate_names)
            return OcpQueryPlan("yaml", resources[:1], namespace, target_name, candidate_names, pattern, followup=followup, confidence=0.94 if resources else 0.72)
        if self._is_relationship_request(lowered):
            resource = "services" if not resources or resources[0] in {"services", "routes"} else resources[0]
            candidate_names = self._candidate_names(topic_state, resource)
            target_name = self._resolve_target_name(user_message, resource, candidate_names)
            return OcpQueryPlan("relationship", [resource], namespace, target_name, candidate_names, pattern, followup=followup, confidence=0.86 if resource else 0.68)
        warning_only = any(marker in lowered for marker in self.WARNING_MARKERS)
        status_check = any(marker in lowered for marker in self.STATUS_CHECK_MARKERS)
        if not resources and not warning_only and (pattern or status_check):
            resources = ["pods"]
        complex_query = len(resources) > 1 or self._has_complex_query_marker(lowered)
        simple_resource_query = len(resources) == 1 and not complex_query
        return OcpQueryPlan(
            "summary",
            self._normalize_resource_list(resources),
            namespace,
            "",
            [],
            pattern=pattern,
            warning_only=warning_only,
            status_check=status_check,
            followup=followup,
            confidence=0.92 if simple_resource_query else 0.64,
        )

    def _should_use_rule_fastpath(self, plan: OcpQueryPlan, lowered: str) -> bool:
        if plan.mode in {"yaml", "relationship"}:
            return bool(plan.resource and (plan.target_name or plan.candidate_names or plan.mode != "yaml"))
        return len(plan.resources) <= 1 and not self._has_complex_query_marker(lowered) and plan.confidence >= 0.8

    async def _build_agent_fallback_plan(self, user_message: str, topic_state: dict, rule_plan: OcpQueryPlan) -> OcpQueryPlan | None:
        if self.llm_client is None:
            return None
        prompt = (
            "You are an OpenShift read-only query planner. Return JSON only. "
            f"Choose mode from summary|yaml|relationship and resources only from {', '.join(self.ALL_RESOURCES)}. "
            "Never answer the question.\n"
            '{"mode":"summary","resources":["pods"],"namespace":"","target_name":"","pattern":"","warning_only":false,"status_check":false,"followup":false,"confidence":0.0}\n'
            f"user_message={user_message}\n"
            f"recent_context={json.dumps({'last_namespace': topic_state.get('last_namespace', ''), 'last_ocp_resource': topic_state.get('last_ocp_resource', ''), 'last_ocp_resource_names': topic_state.get('last_ocp_resource_names', [])[:10], 'last_ocp_filter_keyword': topic_state.get('last_ocp_filter_keyword', '')}, ensure_ascii=False)}\n"
            f"rule_plan={json.dumps(rule_plan.to_dict(), ensure_ascii=False)}"
        )
        try:
            raw = await self.llm_client.generate([{"role": "user", "content": prompt}], max_tokens=320)
            data = json.loads(self._extract_json_object(raw))
        except Exception:
            return None
        mode = str(data.get("mode") or rule_plan.mode).strip().lower()
        if mode not in {"summary", "yaml", "relationship"}:
            return None
        resources = self._normalize_resource_list(data.get("resources") or rule_plan.resources)
        plan = OcpQueryPlan(
            mode,
            resources or list(rule_plan.resources),
            str(data.get("namespace") or rule_plan.namespace or topic_state.get("last_namespace") or ""),
            str(data.get("target_name") or rule_plan.target_name or ""),
            list(data.get("candidate_names") or rule_plan.candidate_names or []),
            str(data.get("pattern") or rule_plan.pattern or ""),
            bool(data.get("warning_only", rule_plan.warning_only)),
            bool(data.get("status_check", rule_plan.status_check)),
            bool(data.get("followup", rule_plan.followup)),
            "agent_fallback",
            max(float(data.get("confidence", rule_plan.confidence) or 0.0), rule_plan.confidence),
        )
        if plan.mode == "yaml" and not plan.target_name:
            candidate_names = self._candidate_names(topic_state, plan.resource)
            plan.candidate_names = plan.candidate_names or candidate_names
            plan.target_name = self._resolve_target_name(user_message, plan.resource, plan.candidate_names)
        if plan.mode == "relationship":
            if not plan.resources:
                plan.resources = ["services"]
            if not plan.target_name:
                candidate_names = self._candidate_names(topic_state, "services")
                plan.candidate_names = plan.candidate_names or candidate_names
                plan.target_name = self._resolve_target_name(user_message, "services", plan.candidate_names)
        return plan

    @staticmethod
    def _extract_json_object(raw: str) -> str:
        text = str(raw or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("json object not found")
        return text[start : end + 1]

    def _is_explanation_request(self, lowered: str) -> bool:
        if any(marker in lowered for marker in self.EXPLANATION_MARKERS) and not any(marker in lowered for marker in self.OPERATION_MARKERS):
            return True
        if any(marker in lowered for marker in self.GUIDE_MARKERS):
            return True
        return False

    def _has_live_runtime_signal(self, user_message: str, topic_state: dict, rule_plan: OcpQueryPlan | None = None) -> bool:
        lowered = user_message.casefold()
        rule_plan = rule_plan or self._build_rule_plan(user_message, topic_state)
        if rule_plan is None:
            return False
        if any(marker in lowered for marker in self.LIVE_STATUS_MARKERS):
            return True
        if rule_plan.followup and bool(topic_state.get("last_ocp_result_items") or topic_state.get("last_ocp_resource")):
            return True
        return False

    def _has_complex_query_marker(self, lowered: str) -> bool:
        if any(marker in lowered for marker in ("같이", "함께", "동시에", "비교")):
            return True
        return bool(re.search(r"\b(?:and|plus)\b", lowered))

    def _mentions_official_doc(self, lowered: str) -> bool:
        return any(marker in lowered for marker in ("공식 문서", "공식 docs", "official docs", "official document", "공식 가이드"))

    def _detect_resources(self, lowered: str) -> list[str]:
        matched: list[str] = []
        for resource, aliases in self.RESOURCE_ALIASES.items():
            if any(alias in lowered for alias in aliases) and resource not in matched:
                matched.append(resource)
        return matched

    def _extract_namespace(self, message: str) -> str:
        for pattern in (r"\b([a-z0-9][a-z0-9-]*)\s*namespace\b", r"\b([a-z0-9][a-z0-9-]*)\s*네임스페이스\b"):
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_pattern(self, message: str, namespace: str, resources: list[str]) -> str:
        keywords = normalize_query_keywords(message)
        ignore = set(self.STOPWORDS)
        generic_prefixes = ("확인", "보여", "알려", "어떻", "지금", "현재", "그럼")
        if namespace:
            ignore.add(namespace.casefold())
        for resource in resources:
            ignore.add(resource.casefold())
            ignore.add((resource[:-1] if resource.endswith("s") else resource).casefold())
        for aliases in self.RESOURCE_ALIASES.values():
            ignore.update(alias.casefold() for alias in aliases)
        for token in keywords:
            lowered = token.casefold()
            if lowered.startswith(generic_prefixes):
                continue
            if lowered not in ignore and len(token) >= 2 and not self._looks_like_count_token(token):
                return token
        return ""

    @staticmethod
    def _looks_like_count_token(token: str) -> bool:
        lowered = str(token or "").casefold().strip()
        return bool(lowered) and (lowered in {"몇", "몇개", "몇개야", "몇개지", "개", "개야", "count"} or lowered.endswith("개야") or lowered.endswith("개지"))

    def _candidate_names(self, topic_state: dict, resource: str) -> list[str]:
        names = [str(value) for value in topic_state.get("last_ocp_resource_names", []) if value]
        items = topic_state.get("last_ocp_result_items") or []
        filtered = [item for item in items if str(item.get("resource") or "") == resource] if resource else items
        unique_names: list[str] = []
        if filtered:
            for item in filtered:
                name = str(item.get("name") or "").strip()
                if name and name not in unique_names:
                    unique_names.append(name)
        if unique_names:
            return unique_names[:8]
        deduped: list[str] = []
        for name in names:
            if name and name not in deduped:
                deduped.append(name)
        return deduped[:8]

    def _resolve_target_name(self, message: str, resource: str, candidate_names: list[str]) -> str:
        explicit = self._extract_explicit_target_name(message, resource, candidate_names)
        if explicit:
            return explicit
        return candidate_names[0] if len(candidate_names) == 1 else ""

    def _extract_explicit_target_name(self, message: str, resource: str, candidate_names: list[str]) -> str:
        lowered_message = normalize_text(message).casefold()
        for candidate in sorted((str(value).strip() for value in candidate_names if value), key=len, reverse=True):
            if candidate and candidate.casefold() in lowered_message:
                return candidate

        ignore = set(self.STOPWORDS)
        ignore.update({str(resource or "").casefold(), str(resource[:-1] if resource.endswith("s") else resource).casefold()})
        for aliases in self.RESOURCE_ALIASES.values():
            ignore.update(alias.casefold() for alias in aliases)

        token_pattern = re.compile(r"\b[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\b", flags=re.IGNORECASE)
        for token in token_pattern.findall(message):
            lowered = token.casefold()
            if lowered in ignore or len(token) < 3:
                continue
            if not ("-" in token or "." in token or any(char.isdigit() for char in token) or len(token) >= 8):
                continue
            return token
        return ""

    def _is_yaml_request(self, lowered: str) -> bool:
        return any(marker in lowered for marker in self.YAML_MARKERS)

    def _is_relationship_request(self, lowered: str) -> bool:
        return ("route" in lowered or "라우트" in lowered) and (("service" in lowered or "서비스" in lowered) or any(marker in lowered for marker in self.CONNECTION_MARKERS))

    def _normalize_resource_list(self, resources: Any) -> list[str]:
        normalized: list[str] = []
        for value in resources or []:
            resource = str(value or "").strip().lower()
            if resource in self.ALL_RESOURCES and resource not in normalized:
                normalized.append(resource)
        return normalized


class OcpChatService:
    ALL_RESOURCES = ("pods", "deployments", "services", "routes", "events")
    DOC_COMMAND_HINTS = ("명령어", "커맨드", "command", "commands", "cli", "kubectl", "oc ", "보려면", "쳐야", "알려줘")
    LIVE_ONLY_HINTS = ("현재 ocp", "현재 상태", "실제 결과", "같이 알려", "같이 보여", "namespace에서는", "네임스페이스에서는")
    LIVE_CONTEXT_MARKERS = ("지금", "현재", "실제", "결과", "pandas", "warning", "이벤트")

    def __init__(self, *, ocp_api_client: Any, session_repository: Any, answer_service: Any, llm_client: Any | None = None) -> None:
        self.ocp_api_client = ocp_api_client
        self.session_repository = session_repository
        self.answer_service = answer_service
        self.llm_client = llm_client
        self.planner = RuleFirstOcpPlanner(ocp_api_client=ocp_api_client, llm_client=llm_client)

    def should_handle(self, session_id: str, user_message: str) -> bool:
        return self.detect_query_mode(session_id, user_message) == "ocp"

    def detect_query_mode(self, session_id: str, user_message: str) -> str:
        topic_state = self.session_repository.topic_state(session_id)
        lowered = normalize_text(user_message).casefold()
        if not lowered:
            return "document"
        has_doc_command_hint = any(marker in lowered for marker in self.DOC_COMMAND_HINTS)
        has_live_context = bool(topic_state.get("last_ocp_result_items") or topic_state.get("last_ocp_resource"))
        has_live_followup = any(marker in lowered for marker in self.planner.LIVE_STATUS_MARKERS) and bool(topic_state.get("last_explicit_resources"))
        has_doc_to_live_bridge = has_live_followup and any(marker in lowered for marker in self.planner.DOC_TO_LIVE_BRIDGE_HINTS)
        is_mixed_request = self.planner.is_mixed_request(user_message, topic_state)
        if has_doc_to_live_bridge and not topic_state.get("last_ocp_result_items") and not topic_state.get("last_ocp_resource"):
            return "mixed"
        if self.planner._is_yaml_request(lowered) and has_live_context and not is_mixed_request:
            return "ocp"
        if has_doc_command_hint and not has_live_context and not any(marker in lowered for marker in self.LIVE_ONLY_HINTS) and not any(marker in lowered for marker in self.LIVE_CONTEXT_MARKERS):
            return "document"
        if is_mixed_request:
            return "mixed"
        if self.planner._is_explanation_request(lowered):
            return "document"
        return "ocp" if self.planner._build_rule_plan(user_message, topic_state) is not None else "document"

    async def _build_plan(self, session_id: str, user_message: str) -> OcpQueryPlan | None:
        return await self.planner.build_plan(user_message, self.session_repository.topic_state(session_id))

    async def stream(self, session_id: str, user_message: str, *, append_user_turn: bool = True) -> AsyncIterator[dict]:
        plan = await self._build_plan(session_id, user_message)
        if plan is None:
            return
        if append_user_turn:
            self.session_repository.add_turn(session_id, "user", user_message)
        yield {"type": "status", "stage": "querying_ocp", "message": "OpenShift 상태를 조회하는 중입니다."}
        if not self.ocp_api_client.enabled:
            async for event in self._yield_terminal(session_id, user_message, "ocp_unavailable", "OCP API 클라이언트가 설정되지 않아 현재 클러스터 상태를 조회할 수 없습니다.", plan, [], 0.0):
                yield event
            return
        try:
            result = await self._execute_plan(plan)
        except Exception as exc:
            async for event in self._yield_terminal(session_id, user_message, "ocp_error", f"OCP API 조회 중 오류가 발생했습니다: {exc}", plan, [], 0.0, error="ocp_query_failed"):
                yield event
            return
        async for event in self._yield_terminal(session_id, user_message, result["answer_route"], result["answer"], plan, result["ocp_context"]["last_result_items"], max(plan.confidence, 0.5), explicit_context=result["ocp_context"]):
            yield event

    async def _yield_terminal(
        self,
        session_id: str,
        user_message: str,
        answer_route: str,
        answer: str,
        plan: OcpQueryPlan,
        result_items: list[dict],
        top_score: float,
        *,
        explicit_context: dict | None = None,
        error: str | None = None,
    ) -> AsyncIterator[dict]:
        final_payload = self._build_context_payload(
            query=user_message,
            answer_route=answer_route,
            query_interpretation=self._build_query_interpretation(plan),
            ocp_context=explicit_context or self._build_ocp_context(plan, result_items),
            top_score=top_score,
        )
        yield {"type": "context", **self.answer_service.public_context_payload(final_payload)}
        token_event = {"type": "token", "content": answer, "cached": False}
        if error:
            token_event["error"] = error
        yield token_event
        self.session_repository.add_turn(session_id, "assistant", answer, metadata=final_payload)
        yield {"type": "done", "cached": False}

    def _build_context_payload(self, *, query: str, answer_route: str, query_interpretation: dict, ocp_context: dict, top_score: float) -> dict:
        payload = self.answer_service.build_context_payload(query, "ocp", top_score, None, [], [], [], [], preview_finalized=True)
        payload["answer_route"] = answer_route
        payload["query_interpretation"] = query_interpretation
        payload["ocp_context"] = ocp_context
        return payload

    def _build_query_interpretation(self, plan: OcpQueryPlan) -> dict:
        actions: list[str] = []
        if plan.mode == "relationship":
            actions.append("relationship")
        if plan.mode == "summary" and plan.status_check:
            actions.append("status_check")
        if plan.pattern:
            actions.append("filter")
        if plan.warning_only:
            actions.append("warning_only")
        return {
            "intent": "ocp_yaml" if plan.mode == "yaml" else "ocp_status",
            "resources": list(plan.resources),
            "actions": actions,
            "format_constraints": ["yaml"] if plan.mode == "yaml" else [],
            "response_shape": "yaml" if plan.mode == "yaml" else "status",
            "namespace": plan.namespace,
            "parse_strategy": plan.parse_strategy,
            "confidence": round(plan.confidence, 3),
        }

    def _build_ocp_context(self, plan: OcpQueryPlan, result_items: list[dict]) -> dict:
        names = [str(item.get("name") or "") for item in result_items if item.get("name")]
        unique_names: list[str] = []
        for name in names:
            if name not in unique_names:
                unique_names.append(name)
        return {
            "namespace": plan.namespace,
            "last_resource": plan.resource,
            "last_resource_names": unique_names[:12],
            "last_result_items": result_items[:20],
            "last_filter_keyword": plan.pattern,
            "parse_strategy": plan.parse_strategy,
            "confidence": round(plan.confidence, 3),
        }

    async def _execute_plan(self, plan: OcpQueryPlan) -> dict:
        if plan.mode == "yaml":
            return await self._execute_yaml(plan)
        if plan.mode == "relationship":
            return await self._execute_relationship(plan)
        return await self._execute_summary(plan)

    async def _execute_summary(self, plan: OcpQueryPlan) -> dict:
        resources = list(plan.resources) or list(self.ALL_RESOURCES)
        if len(resources) == 1:
            resource = resources[0]
            payload = await self.ocp_api_client.list_resources(resource, namespace=plan.namespace or None)
            items = self._filter_items(resource, payload.get("items", []), pattern=plan.pattern, warning_only=plan.warning_only)
            result_items = [self._serialize_result_item(resource, payload.get("namespace", plan.namespace), item) for item in items]
            adjusted_plan = OcpQueryPlan(**{**plan.to_dict(), "namespace": payload.get("namespace", plan.namespace)})
            answer = await self._compose_grounded_ocp_answer(
                plan=adjusted_plan,
                namespace=payload.get("namespace", plan.namespace),
                grouped_items={resource: items},
                fallback=self._format_single_resource_answer(resource, payload.get("namespace", plan.namespace), items, pattern=plan.pattern, status_check=plan.status_check),
            )
            return {"answer_route": "ocp_status", "answer": answer, "ocp_context": self._build_ocp_context(adjusted_plan, result_items)}
        grouped_payloads = []
        for resource in resources:
            payload = await self.ocp_api_client.list_resources(resource, namespace=plan.namespace or None)
            items = self._filter_items(resource, payload.get("items", []), pattern=plan.pattern, warning_only=plan.warning_only)
            grouped_payloads.append((resource, payload.get("namespace", plan.namespace), items))
        result_items: list[dict] = []
        grouped_items: dict[str, list[dict]] = {}
        resolved_namespace = plan.namespace
        for resource_name, namespace, items in grouped_payloads:
            resolved_namespace = namespace or resolved_namespace
            grouped_items[resource_name] = items
            result_items.extend(self._serialize_result_item(resource_name, namespace, item) for item in items)
        adjusted_plan = OcpQueryPlan(**{**plan.to_dict(), "namespace": resolved_namespace})
        answer = await self._compose_grounded_ocp_answer(
            plan=adjusted_plan,
            namespace=resolved_namespace,
            grouped_items=grouped_items,
            fallback=self._format_aggregate_answer(grouped_payloads, pattern=plan.pattern),
        )
        return {"answer_route": "ocp_status", "answer": answer, "ocp_context": self._build_ocp_context(adjusted_plan, result_items)}

    async def _execute_yaml(self, plan: OcpQueryPlan) -> dict:
        if not plan.resource:
            return {"answer_route": "ocp_yaml", "answer": "YAML을 조회할 리소스 종류를 먼저 지정해 주세요.", "ocp_context": self._build_ocp_context(plan, [])}
        if not plan.target_name:
            if plan.candidate_names:
                examples = ", ".join(plan.candidate_names[:3])
                return {
                    "answer_route": "ocp_yaml",
                    "answer": f"현재 조회된 {plan.resource} 중 어떤 것을 말씀하시는지 먼저 알려 주세요. 예: {examples}",
                    "ocp_context": self._build_ocp_context(plan, []),
                }
            return {"answer_route": "ocp_yaml", "answer": f"{plan.resource} YAML을 조회할 대상 이름을 먼저 알려 주세요.", "ocp_context": self._build_ocp_context(plan, [])}
        payload = await self.ocp_api_client.get_resource_yaml(plan.resource, name=plan.target_name, namespace=plan.namespace or None)
        yaml_text = self._to_yaml(payload.get("object") or {})
        lines = yaml_text.splitlines()
        visible = "\n".join(lines[:140]) if len(lines) > 140 else yaml_text
        answer = f"{payload['namespace']} namespace의 {payload['resource']} `{payload['name']}` YAML입니다.\n\n```yaml\n{visible}\n```"
        if len(lines) > 140:
            answer += "\n\n응답 길이 때문에 앞부분만 표시했습니다."
        result_items = [self._serialize_result_item(payload["resource"], payload["namespace"], {"name": payload["name"], "kind": payload["resource"][:-1].title()})]
        adjusted_plan = OcpQueryPlan(**{**plan.to_dict(), "namespace": payload["namespace"], "resources": [payload["resource"]]})
        return {"answer_route": "ocp_yaml", "answer": answer, "ocp_context": self._build_ocp_context(adjusted_plan, result_items)}

    async def _execute_relationship(self, plan: OcpQueryPlan) -> dict:
        if not plan.target_name:
            return {"answer_route": "ocp_relationship", "answer": "Route 연결을 확인할 Service 이름을 먼저 알려 주세요.", "ocp_context": self._build_ocp_context(plan, [])}
        payload = await self.ocp_api_client.list_resources("routes", namespace=plan.namespace or None)
        routes = [item for item in payload.get("items", []) if str(item.get("to") or "").casefold() == plan.target_name.casefold()]
        result_items = [self._serialize_result_item("routes", payload["namespace"], item) for item in routes]
        adjusted_plan = OcpQueryPlan(**{**plan.to_dict(), "namespace": payload["namespace"], "resources": ["services"]})
        if not routes:
            fallback = f"{payload['namespace']} namespace에서 Service `{plan.target_name}`와 연결된 Route를 찾지 못했습니다."
        else:
            lines = [f"- {item['name']} (host={item.get('host') or '-'}, target={item.get('to') or '-'})" for item in routes]
            fallback = f"{payload['namespace']} namespace에서 Service `{plan.target_name}`와 연결된 Route는 {len(routes)}개입니다.\n\n" + "\n".join(lines[:8])
        answer = await self._compose_grounded_ocp_answer(
            plan=adjusted_plan,
            namespace=payload["namespace"],
            grouped_items={"routes": routes},
            fallback=fallback,
        )
        return {"answer_route": "ocp_relationship", "answer": answer, "ocp_context": self._build_ocp_context(adjusted_plan, result_items)}

    async def _compose_grounded_ocp_answer(
        self,
        *,
        plan: OcpQueryPlan,
        namespace: str,
        grouped_items: dict[str, list[dict]],
        fallback: str,
    ) -> str:
        if self.llm_client is None:
            return fallback
        if not (plan.parse_strategy == "agent_fallback" or len(plan.resources) > 1 or plan.followup or plan.mode == "relationship"):
            return fallback
        summary_payload = {
            "namespace": namespace,
            "mode": plan.mode,
            "resources": plan.resources,
            "pattern": plan.pattern,
            "warning_only": plan.warning_only,
            "status_check": plan.status_check,
            "items": {
                resource: items[:8]
                for resource, items in grouped_items.items()
            },
        }
        prompt = (
            "You are an OpenShift assistant. Answer only from the provided OCP API facts.\n"
            "Do not invent resources, states, or remediation steps not grounded in the facts.\n"
            "If facts are missing, say so briefly.\n"
            "Write concise Korean operator-facing prose.\n"
            f"Plan: {json.dumps(plan.to_dict(), ensure_ascii=False)}\n"
            f"OCP facts: {json.dumps(summary_payload, ensure_ascii=False)}\n"
            f"Fallback answer: {fallback}\n"
        )
        try:
            answer = (await self.llm_client.generate([{"role": "user", "content": prompt}], max_tokens=420)).strip()
        except Exception:
            return fallback
        return answer or fallback

    def _filter_items(self, resource: str, items: list[dict], *, pattern: str, warning_only: bool) -> list[dict]:
        filtered = list(items or [])
        if warning_only and resource == "events":
            filtered = [item for item in filtered if str(item.get("type") or "").casefold() == "warning"]
        if pattern:
            needle = pattern.casefold()
            filtered = [item for item in filtered if needle in " ".join(str(item.get(field) or "") for field in ("name", "kind", "phase", "type", "host", "to", "cluster_ip", "node_name")).casefold()]
        return filtered

    def _format_aggregate_answer(self, grouped_payloads: list[tuple[str, str, list[dict]]], *, pattern: str) -> str:
        namespace = next((ns for _resource, ns, _items in grouped_payloads if ns), "")
        total = sum(len(items) for _resource, _namespace, items in grouped_payloads)
        if total == 0:
            return f"{namespace} namespace에서 `{pattern}`와 일치하는 리소스를 찾지 못했습니다." if pattern else f"{namespace} namespace에서 확인된 리소스가 없습니다."
        head = f"{namespace} namespace 기준으로 확인한 결과입니다."
        if pattern:
            head = f"{namespace} namespace에서 `{pattern}`와 일치하는 리소스 기준 결과입니다."
        lines = [head]
        for resource, _ns, items in grouped_payloads:
            if not items:
                continue
            lines.append(f"- {self._resource_label(resource)}: {len(items)}개")
            lines.extend(f"  {line}" for line in self._format_items(resource, items, limit=4))
        return "\n".join(lines)

    def _format_single_resource_answer(self, resource: str, namespace: str, items: list[dict], *, pattern: str, status_check: bool) -> str:
        label = self._resource_label(resource)
        if not items:
            return f"{namespace} namespace에서 `{pattern}`와 일치하는 {label}을(를) 찾지 못했습니다." if pattern else f"{namespace} namespace에서 확인된 {label}이(가) 없습니다."
        lines: list[str] = []
        if status_check and len(items) == 1:
            lines.append(self._status_line(resource, items[0]))
        elif pattern:
            lines.append(f"{namespace} namespace에서 `{pattern}`와 일치하는 {label}은(는) {len(items)}개입니다.")
        else:
            lines.append(f"{namespace} namespace의 {label}은(는) {len(items)}개입니다.")
        lines.extend(self._format_items(resource, items, limit=8))
        return "\n".join(lines)

    def _format_items(self, resource: str, items: list[dict], *, limit: int) -> list[str]:
        formatted: list[str] = []
        for item in items[:limit]:
            if resource == "pods":
                formatted.append(f"- {item['name']} (phase={item.get('phase') or '-'}, node={item.get('node_name') or '-'})")
            elif resource == "deployments":
                formatted.append(f"- {item['name']} ({item.get('ready_replicas', 0)}/{item.get('replicas', 0)} ready)")
            elif resource == "services":
                formatted.append(f"- {item['name']} (type={item.get('type') or '-'}, clusterIP={item.get('cluster_ip') or '-'})")
            elif resource == "routes":
                formatted.append(f"- {item['name']} (host={item.get('host') or '-'}, target={item.get('to') or '-'})")
            else:
                formatted.append(f"- {item['name']} ({item.get('type') or '-'} {item.get('phase') or '-'} → {item.get('to') or '-'})")
        return formatted

    def _status_line(self, resource: str, item: dict) -> str:
        if resource == "deployments":
            ready = int(item.get("ready_replicas") or 0)
            replicas = int(item.get("replicas") or 0)
            return f"Deployment `{item['name']}`는 정상으로 보입니다. ({ready}/{replicas} ready)" if replicas > 0 and ready >= replicas else f"Deployment `{item['name']}`는 확인이 필요합니다. ({ready}/{replicas} ready)"
        if resource == "pods":
            phase = str(item.get("phase") or "")
            return f"Pod `{item['name']}`는 Running 상태입니다." if phase.casefold() == "running" else f"Pod `{item['name']}`는 `{phase or 'Unknown'}` 상태입니다."
        return f"{self._resource_label(resource)} `{item['name']}` 상태를 확인했습니다."

    def _serialize_result_item(self, resource: str, namespace: str, item: dict) -> dict:
        return {"resource": resource, "name": str(item.get("name") or ""), "namespace": str(item.get("namespace") or namespace or ""), "kind": str(item.get("kind") or "")}

    def _resource_label(self, resource: str) -> str:
        return {"pods": "Pod", "deployments": "Deployment", "services": "Service", "routes": "Route", "events": "Event"}.get(resource, resource)

    def _to_yaml(self, value: Any, indent: int = 0) -> str:
        prefix = "  " * indent
        if isinstance(value, dict):
            lines: list[str] = []
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._to_yaml(child, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {self._format_scalar(child)}")
            return "\n".join(lines)
        if isinstance(value, list):
            lines = []
            for child in value:
                if isinstance(child, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(self._to_yaml(child, indent + 1))
                else:
                    lines.append(f"{prefix}- {self._format_scalar(child)}")
            return "\n".join(lines)
        return f"{prefix}{self._format_scalar(value)}"

    @staticmethod
    def _format_scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        if not text:
            return '""'
        if re.search(r"[:#\-\n\t]", text):
            return f"\"{text.replace('\"', '\\\"')}\""
        return text

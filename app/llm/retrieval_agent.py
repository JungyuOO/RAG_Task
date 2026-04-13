from __future__ import annotations

import json
import re

from app.llm.base_agent import BaseAgent
from app.rag.utils import normalize_domain_terms, normalize_query_keywords, normalize_text


RETRIEVAL_SYSTEM_PROMPT = """
You optimize search queries for a document-grounded RAG system.
Return only JSON with:
- expanded_query
- alternatives
- translated_keywords
- target_versions
- multi_source
- resources
- actions
- format_constraints
- response_shape
"""


class RetrievalAgent(BaseAgent):
    FOLLOWUP_REFERENCE_MARKERS = (
        "그 ",
        "그때",
        "그 흐름",
        "그다음",
        "그 다음",
        "방금",
        "이어서",
        "이번에는",
        "다시",
        "같은 주제",
        "같은 내용",
        "앞에서",
        "앞서",
        "that",
        "those",
        "again",
        "continue",
        "follow-up",
        "same",
    )
    COMPARE_MARKERS = (
        "비교",
        "차이",
        "compare",
        "difference",
    )
    DOCUMENT_QUERY_HINTS = (
        "설명",
        "정리",
        "비교",
        "차이",
        "예시",
        "코드",
        "yaml",
        "문서",
        "페이지",
        "출처",
        "무엇",
        "뭐야",
        "어떻게",
        "what",
        "how",
        "why",
        "compare",
        "difference",
        "explain",
    )
    RESOURCE_CANONICALS = (
        "pod",
        "namespace",
        "project",
        "deployment",
        "service",
        "route",
        "ingress",
        "storageclass",
        "node",
        "configmap",
        "secret",
        "statefulset",
        "daemonset",
        "pv",
        "pvc",
    )
    SEARCH_STOPWORDS = {
        "설명",
        "정의",
        "개념",
        "관계",
        "무엇",
        "뭐야",
        "대해",
        "알려줘",
        "해주세요",
        "해줘",
        "what",
        "is",
        "explain",
        "overview",
    }
    OFFICIAL_DOC_MARKERS = (
        "공식 문서",
        "공식 docs",
        "red hat docs",
        "redhat docs",
        "official docs",
        "official document",
        "공식 가이드",
    )
    CUSTOMER_DOC_MARKERS = (
        "고객사",
        "우리 고객사",
        "운영 매뉴얼",
        "운영 메뉴얼",
        "운영 지침서",
        "개발 매뉴얼",
        "개발 메뉴얼",
        "운영 가이드",
        "사내",
        "우리 코드",
        "고객사 코드",
    )
    CLI_HINT_MARKERS = (
        "명령어",
        "커맨드",
        "command",
        "cli",
        "kubectl",
        "oc ",
        "사용법",
        "어떻게 써",
        "어떻게 사용",
        "어떤 명령",
    )
    GENERIC_CLI_ANCHORS = ("oc", "cli", "command", "example")
    GENERIC_YAML_ANCHORS = ("yaml", "manifest", "oc", "example")
    GENERIC_COMMAND_TOKENS = {
        "pod", "pods", "namespace", "project", "projects", "yaml", "manifest", "cli", "command", "commands",
        "명령어", "커맨드", "확인", "봐", "보여", "보려면", "뭐야", "무슨", "어떤", "상태", "현재", "결과",
        "그거", "그", "그쪽", "해당", "이거", "this", "that", "those", "it",
        "show", "list", "get", "status", "resource", "resources", "oc", "kubectl",
    }
    GENERIC_COMMAND_PREFIXES = ("확인", "보려", "명령", "커맨", "상태", "결과", "현재", "어떤", "무슨")

    def __init__(self, llm_client) -> None:
        super().__init__(llm_client, system_prompt=RETRIEVAL_SYSTEM_PROMPT)

    async def expand(self, user_message: str, intent_result: dict, available_sources: list) -> dict:
        normalized_message = normalize_domain_terms(user_message)
        if self._should_use_fast_path(normalized_message, intent_result):
            result = self._fast_path_expand(normalized_message, available_sources)
        else:
            result = await self.call(
                normalized_message,
                intent_result=json.dumps(intent_result, ensure_ascii=False),
                available_sources=json.dumps(available_sources, ensure_ascii=False),
            )

        if "expanded_query" not in result:
            result["expanded_query"] = normalized_message
        if "alternatives" not in result or not isinstance(result["alternatives"], list):
            result["alternatives"] = []
        if "translated_keywords" not in result or not isinstance(result["translated_keywords"], list):
            result["translated_keywords"] = []
        if "target_versions" not in result or not isinstance(result["target_versions"], list):
            result["target_versions"] = self._detect_versions(normalized_message, available_sources)
        if "multi_source" not in result:
            result["multi_source"] = self._detect_multi_source(normalized_message)
        if not result["target_versions"]:
            result["target_versions"] = self._detect_versions(result["expanded_query"], available_sources)

        if "resources" not in result or not isinstance(result.get("resources"), list):
            result["resources"] = []
        if "actions" not in result or not isinstance(result.get("actions"), list):
            result["actions"] = []
        if "format_constraints" not in result or not isinstance(result.get("format_constraints"), list):
            result["format_constraints"] = []
        if "response_shape" not in result or not isinstance(result.get("response_shape"), str):
            result["response_shape"] = ""

        result.setdefault("refined_query", result["expanded_query"])
        result.setdefault("alternative_queries", result["alternatives"])
        result.setdefault("search_keywords", list(intent_result.get("keywords", [])))
        return result

    def interpret(self, user_message: str, query_result: dict | None = None, topic_state: dict | None = None) -> dict:
        normalized_message = normalize_text(normalize_domain_terms(user_message)).lower()
        query_result = query_result or {}
        topic_state = topic_state or {}
        normalized_keywords = normalize_query_keywords(normalized_message, query_result.get("search_keywords", []))

        resources = [
            str(r).lower()
            for r in query_result.get("resources", [])
            if r
            and not str(r).lower().startswith(("http://", "https://"))
            and "/" not in str(r)
            and len(str(r)) <= 64
        ]
        actions = [str(a).lower() for a in query_result.get("actions", []) if a]
        format_constraints = [str(f).lower() for f in query_result.get("format_constraints", []) if f]
        response_shape = str(query_result.get("response_shape", "") or "").lower().strip()

        if not resources:
            resources = self._extract_resources(normalized_message)
        if not actions:
            actions = self._extract_actions(normalized_message)
        if not format_constraints:
            format_constraints = self._extract_format_constraints(normalized_message)
        if not resources:
            resources = self._inherit_resources_from_topic(normalized_message, normalized_keywords, topic_state)

        needs_multiturn_state = any(marker in normalized_message for marker in ("다음", "계속", "step", "1단계", "2단계", "3단계")) or self._has_followup_reference(normalized_message)

        if not response_shape:
            response_shape = self._fallback_response_shape(format_constraints, actions)
        intent = self._determine_intent(response_shape, format_constraints, actions)
        is_document_query = self._is_document_query(normalized_message, resources, actions, format_constraints, topic_state)
        document_group_preference = self._detect_document_group_preference(normalized_message, topic_state)
        if document_group_preference == "auto":
            document_group_preference = str(
                topic_state.get("last_document_group_preference")
                or topic_state.get("active_document_group")
                or "auto"
            )
        generic_command_query = self._is_generic_command_query(
            normalized_message,
            normalized_keywords=normalized_keywords,
            resources=resources,
            format_constraints=format_constraints,
        )

        return {
            "intent": intent,
            "is_document_query": is_document_query,
            "target_versions": [str(value).strip() for value in query_result.get("target_versions", []) if value],
            "document_group_preference": document_group_preference,
            "resources": resources,
            "actions": actions,
            "format_constraints": format_constraints,
            "response_shape": response_shape,
            "normalized_keywords": normalized_keywords,
            "needs_multiturn_state": needs_multiturn_state,
            "generic_command_query": generic_command_query,
        }

    def _is_generic_command_query(
        self,
        normalized_message: str,
        *,
        normalized_keywords: list[str],
        resources: list[str],
        format_constraints: list[str],
    ) -> bool:
        if not ({"cli", "yaml"} & {value.casefold() for value in format_constraints}):
            return False
        resource_tokens = {resource.casefold() for resource in resources if resource}
        resource_tokens.update({f"{resource}s" for resource in list(resource_tokens)})
        lowered = normalized_message.casefold()
        if any(marker in lowered for marker in ("pandas", "cert-manager", "workshop", "operator", "openshift-")):
            return False
        meaningful = []
        for token in normalized_keywords:
            normalized = str(token).casefold().strip()
            if not normalized or len(normalized) < 2:
                continue
            if (
                normalized in self.GENERIC_COMMAND_TOKENS
                or normalized in resource_tokens
                or any(normalized.startswith(prefix) for prefix in self.GENERIC_COMMAND_PREFIXES)
            ):
                continue
            meaningful.append(normalized)
        return not meaningful

    def _should_use_fast_path(self, normalized_message: str, intent_result: dict) -> bool:
        if str(intent_result.get("intent", "")).casefold() != "rag":
            return False
        lowered = normalized_message.casefold()
        compare_requested = any(marker in lowered for marker in self.COMPARE_MARKERS)
        wants_official = self._mentions_official_doc(lowered)
        wants_customer = self._mentions_customer_doc(lowered)
        if compare_requested and not (wants_official and wants_customer):
            return False
        if self._detect_multi_source(lowered) and wants_official and wants_customer:
            return False
        return True

    def _fast_path_expand(self, normalized_message: str, available_sources: list) -> dict:
        keywords = normalize_query_keywords(normalized_message)
        resources = self._extract_resources(normalized_message)
        actions = self._extract_actions(normalized_message)
        if not actions and resources:
            actions = ["explain"]
        format_constraints = self._extract_format_constraints(normalized_message)
        response_shape = self._fallback_response_shape(format_constraints, actions)
        search_terms = [keyword for keyword in keywords if keyword not in self.SEARCH_STOPWORDS]
        intent_anchors = self._build_intent_anchors(
            normalized_message,
            resources=resources,
            format_constraints=format_constraints,
            normalized_keywords=keywords,
        )
        expanded_terms = search_terms[:8] + [anchor for anchor in intent_anchors if anchor not in search_terms]
        expanded_query = " ".join(expanded_terms).strip() if expanded_terms else normalized_message
        translated_keywords = [keyword for keyword in keywords if keyword.isascii()][:4]
        lowered = normalized_message.casefold()
        multi_source = self._detect_multi_source(normalized_message) or (
            any(marker in lowered for marker in self.COMPARE_MARKERS)
            and self._mentions_official_doc(lowered)
            and self._mentions_customer_doc(lowered)
        )
        return {
            "expanded_query": expanded_query,
            "alternatives": [],
            "translated_keywords": translated_keywords,
            "target_versions": self._detect_versions(normalized_message, available_sources),
            "multi_source": multi_source,
            "resources": resources,
            "actions": actions,
            "format_constraints": format_constraints,
            "response_shape": response_shape,
        }

    def _build_intent_anchors(
        self,
        normalized_message: str,
        *,
        resources: list[str],
        format_constraints: list[str],
        normalized_keywords: list[str],
    ) -> list[str]:
        lowered = normalized_message.casefold()
        anchors: list[str] = []

        def _append(values: tuple[str, ...]) -> None:
            for value in values:
                if value not in anchors:
                    anchors.append(value)

        if "cli" in format_constraints or "yaml" in format_constraints:
            _append(self.GENERIC_CLI_ANCHORS)
            if resources:
                _append(tuple(f"{resource}s" for resource in resources if resource))
            if "namespace" in lowered or "project" in lowered or "네임스페이스" in lowered or "프로젝트" in lowered:
                _append(("namespace", "project"))
            if "yaml" in format_constraints:
                _append(self.GENERIC_YAML_ANCHORS)
                _append(("-o", "yaml"))
            custom_keywords = [
                keyword
                for keyword in normalized_keywords
                if len(keyword) >= 3 and keyword not in self.SEARCH_STOPWORDS and keyword not in {"yaml", "namespace", "project"}
            ]
            if custom_keywords and not resources:
                _append(("oc", "get"))
        return anchors

    def _extract_resources(self, normalized_message: str) -> list[str]:
        lowered = normalized_message.casefold()
        resources: list[str] = []
        for canonical in self.RESOURCE_CANONICALS:
            if canonical in lowered and canonical not in resources:
                resources.append(canonical)
        if "네임스페이스" in lowered and "namespace" not in resources:
            resources.append("namespace")
        if "프로젝트" in lowered and "project" not in resources:
            resources.append("project")
        if "배포" in lowered and "deployment" not in resources:
            resources.append("deployment")
        if any(marker in lowered for marker in ("연결", "selector", "내부 접근", "연결 구조", "리소스 연결", "서비스 연결")) and "service" not in resources:
            resources.append("service")
        if any(marker in lowered for marker in ("노출", "route", "외부 접근", "연결 구조", "리소스 연결", "외부 노출")) and "route" not in resources:
            resources.append("route")
        if any(marker in lowered for marker in ("storageclass", "storage class", "스토리지클래스", "스토리지 클래스")) and "storageclass" not in resources:
            resources.append("storageclass")
        if any(marker in lowered for marker in ("pv", "persistentvolume", "퍼시스턴트볼륨")) and "pv" not in resources:
            resources.append("pv")
        if any(marker in lowered for marker in ("pvc", "persistentvolumeclaim", "퍼시스턴트볼륨클레임")) and "pvc" not in resources:
            resources.append("pvc")
        return resources

    @staticmethod
    def _extract_actions(normalized_message: str) -> list[str]:
        lowered = normalized_message.casefold()
        actions: list[str] = []
        if any(marker in lowered for marker in ("설명", "정의", "개념", "뭐야", "무엇", "what", "explain")):
            actions.append("explain")
        if any(marker in lowered for marker in ("비교", "차이", "compare", "difference")):
            actions.append("compare")
        if any(marker in lowered for marker in ("생성", "만들", "작성", "create")):
            actions.append("create")
        return actions

    @staticmethod
    def _extract_format_constraints(normalized_message: str) -> list[str]:
        lowered = normalized_message.casefold()
        formats: list[str] = []
        if "yaml" in lowered or "manifest" in lowered:
            formats.append("yaml")
        resource_mentions = any(canonical in lowered for canonical in RetrievalAgent.RESOURCE_CANONICALS)
        cli_syntax_hint = bool(re.search(r"\b(?:oc|kubectl)\b", lowered))
        cli_usage_hint = any(marker in lowered for marker in RetrievalAgent.CLI_HINT_MARKERS)
        if "cli" in lowered or "command" in lowered or "명령어" in lowered or cli_syntax_hint or cli_usage_hint:
            formats.append("cli")
        elif resource_mentions and any(marker in lowered for marker in ("어떻게", "사용", "써", "치면", "입력")):
            formats.append("cli")
        if "table" in lowered or "표" in lowered:
            formats.append("table")
        return formats

    def _detect_versions(self, text: str, available_sources: list) -> list[str]:
        del available_sources
        versions = set(re.findall(r"\b(\d+\.\d+)\b", text or ""))
        return sorted(versions)

    @staticmethod
    def _sanitize_target_versions(values: list[str] | None) -> list[str]:
        sanitized: list[str] = []
        for value in values or []:
            text = str(value).strip()
            if re.fullmatch(r"\d+\.\d+", text) and text not in sanitized:
                sanitized.append(text)
        return sanitized

    def _sanitize_resources(self, values: list[str] | None) -> list[str]:
        sanitized: list[str] = []
        for value in values or []:
            text = str(value).strip().lower()
            if text in self.RESOURCE_CANONICALS and text not in sanitized:
                sanitized.append(text)
        return sanitized

    def _detect_multi_source(self, text: str) -> bool:
        normalized = (text or "").lower()
        import re as _re
        if _re.search(r"[\w가-힣]+와\s+[\w가-힣]+", normalized):
            return True
        simple_markers = ("그리고", "and", "비교")
        if any(marker in normalized for marker in simple_markers):
            return True
        if "," in normalized and self._extract_resources(text) and len(self._extract_resources(text)) >= 2:
            return True
        return False

    def _detect_document_group_preference(self, normalized_message: str, topic_state: dict | None = None) -> str:
        lowered = normalized_message.casefold()
        topic_state = topic_state or {}
        wants_official = self._mentions_official_doc(lowered)
        wants_customer = self._mentions_customer_doc(lowered)
        prior_group = str(topic_state.get("last_document_group_preference") or topic_state.get("active_document_group") or "auto")
        compare_requested = any(marker in lowered for marker in self.COMPARE_MARKERS)
        if self._detect_multi_source(lowered) and wants_official and wants_customer:
            return "mixed"
        if compare_requested and wants_official and wants_customer:
            return "mixed"
        if wants_official and wants_customer:
            return "mixed"
        if wants_customer:
            return "customer_generated"
        if wants_official:
            return "official_ocp"
        if prior_group in {"official_ocp", "customer_generated"} and self._has_followup_reference(lowered):
            return prior_group
        return "official_ocp"

    def _mentions_official_doc(self, lowered: str) -> bool:
        return any(marker in lowered for marker in self.OFFICIAL_DOC_MARKERS) or (
            "공식" in lowered and any(token in lowered for token in ("문서", "가이드", "기준", "방식"))
        )

    def _mentions_customer_doc(self, lowered: str) -> bool:
        return any(marker in lowered for marker in self.CUSTOMER_DOC_MARKERS) or (
            "고객사" in lowered
        ) or (
            any(token in lowered for token in ("운영", "내부"))
            and any(token in lowered for token in ("메뉴얼", "매뉴얼", "문서", "가이드", "기준", "방식"))
        )

    def _has_followup_reference(self, lowered: str) -> bool:
        return any(marker in lowered for marker in self.FOLLOWUP_REFERENCE_MARKERS)

    def _mentions_official_doc(self, lowered: str) -> bool:
        explicit_markers = (
            "공식 문서",
            "공식 docs",
            "official docs",
            "official document",
            "official doc",
            "red hat docs",
            "redhat docs",
            "레드햇 공식 문서",
            "ocp 공식 문서",
            "ocp 문서",
            "공식 가이드",
        )
        return any(marker in lowered for marker in explicit_markers) or (
            "공식" in lowered and any(token in lowered for token in ("문서", "가이드", "기준", "설명"))
        )

    def _mentions_customer_doc(self, lowered: str) -> bool:
        explicit_markers = (
            "고객사",
            "고객 문서",
            "고객사 문서",
            "고객사 메뉴얼",
            "고객사 매뉴얼",
            "우리 매뉴얼",
            "우리 메뉴얼",
            "운영 매뉴얼",
            "운영 메뉴얼",
            "내부 문서",
            "내부 가이드",
            "사내 문서",
            "사내 가이드",
            "customer guide",
            "customer manual",
        )
        return any(marker in lowered for marker in explicit_markers)

    def _inherit_resources_from_topic(self, normalized_message: str, normalized_keywords: list[str], topic_state: dict) -> list[str]:
        active_slot = topic_state.get("active_slot") or {}
        slot_resources = [str(value).lower().strip() for value in active_slot.get("resources", []) or [] if value]
        slot_code_resource_kind = str(active_slot.get("code_resource_kind") or "").lower().strip()
        slot_example_anchor = active_slot.get("example_anchor") or {}
        if not (
            slot_resources
            or slot_code_resource_kind
            or (slot_example_anchor or {}).get("resource_kind")
            or topic_state.get("last_explicit_resources")
            or topic_state.get("last_code_resource_kind")
            or (topic_state.get("last_example_anchor") or {}).get("resource_kind")
        ):
            return []
        referential_markers = ("그거", "그건", "그 yaml", "그 코드", "그 예시", "다시", "that", "this", "those", "it", "again", "also")
        code_markers = ("yaml", "manifest", "code", "example", "sample", "demo", "예시", "코드")
        procedure_markers = ("단계", "절차", "순서", "step")
        should_inherit = (
            any(marker in normalized_message for marker in referential_markers)
            or any(marker in normalized_message for marker in code_markers)
            or any(marker in normalized_message for marker in procedure_markers)
            or bool({"yaml", "yml", "manifest", "code", "example", "sample", "demo", "예시", "코드"} & set(normalized_keywords))
        )
        if not should_inherit:
            return []
        resources: list[str] = []
        anchor = slot_example_anchor or topic_state.get("last_example_anchor") or {}
        anchor_resource = str(anchor.get("resource_kind") or "").lower().strip()
        if anchor_resource:
            resources.append(anchor_resource)
        for resource in [*slot_resources, *(topic_state.get("last_explicit_resources", []) or [])]:
            normalized_resource = str(resource).lower().strip()
            if normalized_resource and normalized_resource not in resources:
                resources.append(normalized_resource)
        last_code_resource_kind = slot_code_resource_kind or str(topic_state.get("last_code_resource_kind") or "").lower().strip()
        if last_code_resource_kind and last_code_resource_kind not in resources:
            resources.append(last_code_resource_kind)
        return resources

    def _fallback_response_shape(self, format_constraints: list[str], actions: list[str]) -> str:
        if "table" in format_constraints:
            return "table"
        if "yaml" in format_constraints or "cli" in format_constraints:
            return "code"
        if "compare" in actions:
            return "comparison"
        return "text"

    def _determine_intent(self, response_shape: str, format_constraints: list[str], actions: list[str]) -> str:
        if response_shape == "table":
            return "table"
        if response_shape == "code":
            if "yaml" in format_constraints:
                return "yaml_example"
            if "cli" in format_constraints:
                return "cli_example"
            return "code_example"
        if response_shape == "procedure":
            return "procedure_followup"
        if "compare" in actions:
            return "compare"
        return "explain"

    def _is_document_query(self, normalized_message: str, resources: list[str], actions: list[str], format_constraints: list[str], topic_state: dict) -> bool:
        if resources or actions or format_constraints:
            return True
        if any(marker in normalized_message for marker in self.DOCUMENT_QUERY_HINTS):
            return True
        if topic_state.get("selected_sources") and self._has_followup_reference(normalized_message):
            return True
        if topic_state.get("active_topic") or topic_state.get("selected_sources"):
            if len(normalized_message) <= 64:
                return True
        return False

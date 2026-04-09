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
        "deployment",
        "service",
        "route",
        "ingress",
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
        }

    def _should_use_fast_path(self, normalized_message: str, intent_result: dict) -> bool:
        if str(intent_result.get("intent", "")).casefold() != "rag":
            return False
        lowered = normalized_message.casefold()
        compare_requested = any(marker in lowered for marker in self.COMPARE_MARKERS)
        wants_official = self._mentions_official_doc(lowered)
        wants_customer = self._mentions_customer_doc(lowered)
        if self._detect_multi_source(lowered) and not (compare_requested and wants_official and wants_customer):
            return False
        if compare_requested and not (wants_official and wants_customer):
            return False
        if self._extract_resources(normalized_message):
            return True
        explain_markers = ("설명", "정의", "개념", "무엇", "뭐야", "what", "explain", "overview")
        return any(marker in lowered for marker in explain_markers) or (compare_requested and wants_official and wants_customer)

    def _fast_path_expand(self, normalized_message: str, available_sources: list) -> dict:
        keywords = normalize_query_keywords(normalized_message)
        resources = self._extract_resources(normalized_message)
        actions = self._extract_actions(normalized_message)
        if not actions and resources:
            actions = ["explain"]
        format_constraints = self._extract_format_constraints(normalized_message)
        response_shape = self._fallback_response_shape(format_constraints, actions)
        search_terms = [keyword for keyword in keywords if keyword not in self.SEARCH_STOPWORDS]
        expanded_query = " ".join(search_terms[:8]) if search_terms else normalized_message
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

    def _extract_resources(self, normalized_message: str) -> list[str]:
        lowered = normalized_message.casefold()
        resources: list[str] = []
        for canonical in self.RESOURCE_CANONICALS:
            if canonical in lowered and canonical not in resources:
                resources.append(canonical)
        if "배포" in lowered and "deployment" not in resources:
            resources.append("deployment")
        if any(marker in lowered for marker in ("연결", "selector", "내부 접근", "연결 구조", "리소스 연결", "서비스 연결")) and "service" not in resources:
            resources.append("service")
        if any(marker in lowered for marker in ("노출", "route", "외부 접근", "연결 구조", "리소스 연결", "외부 노출")) and "route" not in resources:
            resources.append("route")
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
        if "cli" in lowered or "command" in lowered or "명령어" in lowered:
            formats.append("cli")
        if "table" in lowered or "표" in lowered:
            formats.append("table")
        return formats

    def _detect_versions(self, text: str, available_sources: list) -> list[str]:
        versions = set(re.findall(r"\b(\d+\.\d+)\b", text or ""))
        if versions:
            return sorted(versions)
        source_versions = []
        for item in available_sources or []:
            if isinstance(item, dict):
                version = str(item.get("version", "") or "").strip()
                if version:
                    source_versions.append(version)
        return sorted(set(source_versions))[:1] if source_versions else []

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
        return "auto"

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

    def _inherit_resources_from_topic(self, normalized_message: str, normalized_keywords: list[str], topic_state: dict) -> list[str]:
        if not (topic_state.get("last_explicit_resources") or topic_state.get("last_code_resource_kind") or (topic_state.get("last_example_anchor") or {}).get("resource_kind")):
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
        anchor = topic_state.get("last_example_anchor") or {}
        anchor_resource = str(anchor.get("resource_kind") or "").lower().strip()
        if anchor_resource:
            resources.append(anchor_resource)
        for resource in topic_state.get("last_explicit_resources", []) or []:
            normalized_resource = str(resource).lower().strip()
            if normalized_resource and normalized_resource not in resources:
                resources.append(normalized_resource)
        last_code_resource_kind = str(topic_state.get("last_code_resource_kind") or "").lower().strip()
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

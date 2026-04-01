from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.rag.utils import normalize_query_keywords, normalize_text


@dataclass(slots=True)
class QueryInterpretation:
    intent: str
    is_document_query: bool
    resources: list[str]
    actions: list[str]
    format_constraints: list[str]
    response_shape: str
    normalized_keywords: list[str]
    needs_multiturn_state: bool

    def to_dict(self) -> dict:
        return asdict(self)


class QueryInterpreter:
    RESOURCE_MARKERS = {
        "configmap": ("configmap",),
        "secret": ("secret",),
        "pod": ("pod",),
        "deployment": ("deployment",),
        "service": ("service",),
        "pv": ("pv", "persistentvolume"),
        "pvc": ("pvc", "persistentvolumeclaim"),
        "rbac": ("rbac",),
        "rolebinding": ("rolebinding",),
        "clusterrole": ("clusterrole",),
        "clusterrolebinding": ("clusterrolebinding",),
        "scc": ("scc",),
        "route": ("route",),
        "ingress": ("ingress",),
        "storageclass": ("storageclass",),
        "namespace": ("namespace",),
        "openshift": ("openshift",),
        "kubernetes": ("kubernetes",),
        "argocd": ("argocd", "argo cd", "argo-cd"),
        "tekton": ("tekton",),
        "daemonset": ("daemonset",),
        "statefulset": ("statefulset",),
        "pipeline": ("pipeline",),
        "cicd": ("ci/cd", "cicd"),
        "operator": ("operator",),
    }
    ACTION_MARKERS = {
        "create": ("생성", "만들", "작성", "create"),
        "compare": ("차이", "비교", "compare", "difference", "diff", "vs", "versus"),
        "explain": ("설명", "정리", "의미", "explain", "what", "why"),
        "mount": ("마운트", "mount"),
        "inject": ("주입", "inject"),
        "apply": ("적용", "apply"),
        "delete": ("삭제", "지우", "delete", "remove"),
        "list": ("목록", "종류", "리스트", "list"),
    }
    FORMAT_MARKERS = {
        "yaml": ("yaml", "yml", "manifest", "매니페스트"),
        "cli": ("cli", "kubectl", "oc ", "oc\n", "command", "명령어"),
        "table": ("표", "table"),
    }
    REFERENTIAL_MARKERS = (
        "그거",
        "그건",
        "그중",
        "그 yaml",
        "그 코드",
        "그 예시",
        "그럼",
        "다시",
        "바꿔",
        "로도",
        "that",
        "this",
        "those",
        "again",
        "also",
    )
    MULTITURN_MARKERS = ("다음", "계속", "step", "단계", "1단계", "2단계", "3단계")
    CODE_MARKERS = ("yaml", "manifest", "code", "example", "sample", "demo", "코드", "예시", "샘플")
    PROCEDURE_MARKERS = ("단계", "절차", "순서", "step")
    TABLE_MARKERS = ("표", "table")
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
        "뭐",
        "왜",
        "어떻게",
        "알려줘",
        "보여줘",
        "what",
        "how",
        "why",
        "compare",
        "difference",
        "explain",
    )

    @staticmethod
    def _marker_in_text(normalized_message: str, marker: str) -> bool:
        normalized_marker = str(marker or "").strip().lower()
        if not normalized_marker:
            return False
        if " " in normalized_marker or "/" in normalized_marker:
            return normalized_marker in normalized_message
        if re.fullmatch(r"[a-z0-9_-]+", normalized_marker):
            pattern = rf"(?<![a-z0-9]){re.escape(normalized_marker)}(?![a-z0-9])"
            return re.search(pattern, normalized_message) is not None
        return normalized_marker in normalized_message

    def interpret(
        self,
        user_message: str,
        query_result: dict | None = None,
        topic_state: dict | None = None,
    ) -> QueryInterpretation:
        normalized_message = normalize_text(user_message).lower()
        query_result = query_result or {}
        topic_state = topic_state or {}

        normalized_keywords = normalize_query_keywords(
            user_message,
            query_result.get("search_keywords", []),
        )
        resources = self._extract_resources(normalized_message, normalized_keywords, topic_state)
        actions = self._extract_actions(normalized_message, normalized_keywords)
        format_constraints = self._extract_formats(normalized_message, normalized_keywords)
        needs_multiturn_state = any(marker in normalized_message for marker in self.MULTITURN_MARKERS)
        response_shape = self._determine_response_shape(normalized_message, format_constraints, actions)
        intent = self._determine_intent(response_shape, format_constraints, actions)
        is_document_query = self._is_document_query(normalized_message, resources, actions, format_constraints, topic_state)

        return QueryInterpretation(
            intent=intent,
            is_document_query=is_document_query,
            resources=resources,
            actions=actions,
            format_constraints=format_constraints,
            response_shape=response_shape,
            normalized_keywords=normalized_keywords,
            needs_multiturn_state=needs_multiturn_state,
        )

    def _extract_resources(self, normalized_message: str, normalized_keywords: list[str], topic_state: dict) -> list[str]:
        resources: list[str] = []
        for name, markers in self.RESOURCE_MARKERS.items():
            if any(self._marker_in_text(normalized_message, marker) for marker in markers) or any(marker in normalized_keywords for marker in markers):
                resources.append(name)

        active_entities = [str(value).lower() for value in topic_state.get("active_entities", []) if value]
        known_resources = set(self.RESOURCE_MARKERS.keys())
        for entity in active_entities:
            if entity not in known_resources:
                continue
            if entity in normalized_keywords and entity not in resources:
                resources.append(entity)
        if not resources and self._should_inherit_resources(normalized_message, normalized_keywords, topic_state):
            anchor = topic_state.get("last_example_anchor") or {}
            anchor_resource = str(anchor.get("resource_kind") or "").lower().strip()
            if anchor_resource and anchor_resource in known_resources:
                resources.append(anchor_resource)
            for resource in topic_state.get("last_explicit_resources", []) or []:
                normalized_resource = str(resource).lower().strip()
                if normalized_resource in known_resources and normalized_resource not in resources:
                    resources.append(normalized_resource)
        return resources

    def _should_inherit_resources(
        self,
        normalized_message: str,
        normalized_keywords: list[str],
        topic_state: dict,
    ) -> bool:
        if not topic_state.get("last_explicit_resources"):
            return False
        if any(marker in normalized_message for marker in self.REFERENTIAL_MARKERS):
            return True
        if any(marker in normalized_message for marker in self.CODE_MARKERS):
            return True
        if any(marker in normalized_message for marker in self.PROCEDURE_MARKERS):
            return True
        if len(normalized_message) <= 32 and topic_state.get("active_topic"):
            return True
        return bool(
            {"yaml", "yml", "manifest", "code", "example", "sample", "demo", "예시", "코드"} & set(normalized_keywords)
        )

    def _extract_actions(self, normalized_message: str, normalized_keywords: list[str]) -> list[str]:
        actions: list[str] = []
        for name, markers in self.ACTION_MARKERS.items():
            if any(marker in normalized_message for marker in markers) or any(marker in normalized_keywords for marker in markers):
                actions.append(name)
        return actions

    def _extract_formats(self, normalized_message: str, normalized_keywords: list[str]) -> list[str]:
        formats: list[str] = []
        for name, markers in self.FORMAT_MARKERS.items():
            if any(marker in normalized_message for marker in markers) or any(marker.strip() in normalized_keywords for marker in markers if marker.strip()):
                formats.append(name)
        return formats

    def _determine_response_shape(
        self,
        normalized_message: str,
        format_constraints: list[str],
        actions: list[str],
    ) -> str:
        if "table" in format_constraints or any(marker in normalized_message for marker in self.TABLE_MARKERS):
            return "table"
        if "yaml" in format_constraints or "cli" in format_constraints or any(marker in normalized_message for marker in self.CODE_MARKERS):
            return "code"
        if "compare" in actions:
            return "comparison"
        if any(marker in normalized_message for marker in self.PROCEDURE_MARKERS):
            return "procedure"
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

    def _is_document_query(
        self,
        normalized_message: str,
        resources: list[str],
        actions: list[str],
        format_constraints: list[str],
        topic_state: dict,
    ) -> bool:
        if resources or actions or format_constraints:
            return True
        if any(marker in normalized_message for marker in self.DOCUMENT_QUERY_HINTS):
            return True
        if topic_state.get("active_topic") or topic_state.get("selected_sources"):
            if len(normalized_message) <= 40:
                return True
        return False

from __future__ import annotations

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
        "왜",
        "어떻게",
        "what",
        "how",
        "why",
        "compare",
        "difference",
        "explain",
    )

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
            if any(marker in normalized_message for marker in markers) or any(marker in normalized_keywords for marker in markers):
                resources.append(name)

        active_entities = [str(value).lower() for value in topic_state.get("active_entities", []) if value]
        for entity in active_entities:
            if entity in normalized_keywords and entity not in resources:
                resources.append(entity)
        return resources

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

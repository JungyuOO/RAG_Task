from __future__ import annotations

import json
import re

from app.llm.base_agent import BaseAgent
from app.rag.utils import normalize_query_keywords, normalize_text


RETRIEVAL_SYSTEM_PROMPT = """당신은 RAG 시스템의 검색 최적화 에이전트입니다.
사용자의 질문을 분석하여 벡터 검색에 최적화된 쿼리를 생성합니다.

역할:
1. 사용자 질문에 포함된 핵심 키워드만 영어로 번역하여 짧고 간결한 쿼리 생성
2. 대안 쿼리 2-3개 생성 (이것들도 주어진 키워드 내에서만 구성)
3. 특징적인 버전이 언급되면 target_versions에 포함
4. 여러 주제가 혼합된 질문이면 multi_source: true
5. 질문에 명시적으로 언급된 Kubernetes/OpenShift 리소스를 resources에 포함. 질문에 없는 리소스를 임의로 유추해서 넣지 마세요.
6. 질문에서 요청하는 동작을 actions에 포함
7. 질문에서 요구하는 출력 형식을 format_constraints에 포함
8. 질문에 가장 적절한 응답 형태를 response_shape으로 지정

중요: "expanded_query"와 "alternatives" 작성 시, 질문에 존재하지 않는 연관 개념(예: 스케줄링, CPU, 멀티테넌트 환경, 리소스 할당, 설명 등)을 **절대로** 알아서 추가하지 마세요. 오직 사용자가 입력한 단어를 영어로 단순 번역한 수준(예: "time slicing")으로만 작성해야 문서의 원본 텍스트와 정확히 매칭됩니다. 길이도 가능한 짧게 유지하세요.

사용 가능한 문서 목록:
{available_sources}

의도 분석 결과:
{intent_result}

응답은 항상 JSON 객체로 반환하세요. 반드시 다음 필드를 포함:
{{
  "expanded_query": "확장된 검색 쿼리",
  "alternatives": ["대안 쿼리 1", "대안 쿼리 2"],
  "translated_keywords": ["사용자가 묻는 중심 용어의 단순 영어 번역 (예: time slicing, gpu)"],
  "target_versions": [""],
  "multi_source": false,
  "resources": [],
  "actions": [],
  "format_constraints": [],
  "response_shape": "text"
}}

중요: "translated_keywords"에는 질문에 명시되지 않은 부가 설명이나 관련 도메인 개념을 임의로 추가하지 마세요. 오직 질문에 등장한 핵심 명사만 단순히 영어로 번역한 단어들을 배열 형태로 반환하세요.
"""


class RetrievalAgent(BaseAgent):
    DOCUMENT_QUERY_HINTS = ("설명", "정리", "비교", "차이", "예시", "코드", "yaml", "문서", "페이지", "출처", "무엇", "뭐", "왜", "어떻게", "보여줘", "what", "how", "why", "compare", "difference", "explain")

    def __init__(self, llm_client) -> None:
        super().__init__(llm_client, system_prompt=RETRIEVAL_SYSTEM_PROMPT)

    async def expand(self, user_message: str, intent_result: dict, available_sources: list) -> dict:
        result = await self.call(
            user_message,
            intent_result=json.dumps(intent_result, ensure_ascii=False),
            available_sources=json.dumps(available_sources, ensure_ascii=False),
        )

        if "expanded_query" not in result:
            result["expanded_query"] = user_message
        if "alternatives" not in result or not isinstance(result["alternatives"], list):
            result["alternatives"] = []
        if "translated_keywords" not in result or not isinstance(result["translated_keywords"], list):
            result["translated_keywords"] = []
        if "target_versions" not in result or not isinstance(result["target_versions"], list):
            result["target_versions"] = self._detect_versions(user_message, available_sources)
        if "multi_source" not in result:
            result["multi_source"] = self._detect_multi_source(user_message)
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
        normalized_message = normalize_text(user_message).lower()
        query_result = query_result or {}
        topic_state = topic_state or {}
        normalized_keywords = normalize_query_keywords(user_message, query_result.get("search_keywords", []))

        resources = [str(r).lower() for r in query_result.get("resources", []) if r]
        actions = [str(a).lower() for a in query_result.get("actions", []) if a]
        format_constraints = [str(f).lower() for f in query_result.get("format_constraints", []) if f]
        response_shape = str(query_result.get("response_shape", "") or "").lower().strip()

        if not resources:
            resources = self._inherit_resources_from_topic(normalized_message, normalized_keywords, topic_state)

        needs_multiturn_state = any(marker in normalized_message for marker in ("다음", "계속", "step", "단계", "1단계", "2단계", "3단계"))

        if not response_shape:
            response_shape = self._fallback_response_shape(format_constraints, actions)
        intent = self._determine_intent(response_shape, format_constraints, actions)
        is_document_query = self._is_document_query(normalized_message, resources, actions, format_constraints, topic_state)

        return {
            "intent": intent,
            "is_document_query": is_document_query,
            "resources": resources,
            "actions": actions,
            "format_constraints": format_constraints,
            "response_shape": response_shape,
            "normalized_keywords": normalized_keywords,
            "needs_multiturn_state": needs_multiturn_state,
        }

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
        multi_markers = ("둘다", "둘 다", "그리고", "and", ",", "비교")
        return any(marker in normalized for marker in multi_markers)

    def _inherit_resources_from_topic(self, normalized_message: str, normalized_keywords: list[str], topic_state: dict) -> list[str]:
        if not (topic_state.get("last_explicit_resources") or topic_state.get("last_code_resource_kind") or (topic_state.get("last_example_anchor") or {}).get("resource_kind")):
            return []
        referential_markers = ("그거", "그건", "그것", "그 yaml", "그 코드", "그 예시", "그럼", "다시", "바꿔", "that", "this", "those", "it", "again", "also")
        code_markers = ("yaml", "manifest", "code", "example", "sample", "demo", "코드", "예시", "샘플")
        procedure_markers = ("단계", "절차", "순서", "step")
        should_inherit = (
            any(marker in normalized_message for marker in referential_markers)
            or any(marker in normalized_message for marker in code_markers)
            or any(marker in normalized_message for marker in procedure_markers)
            or (len(normalized_message) <= 32 and topic_state.get("active_topic"))
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
        if topic_state.get("active_topic") or topic_state.get("selected_sources"):
            if len(normalized_message) <= 40:
                return True
        return False

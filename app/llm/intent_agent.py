from __future__ import annotations

import json
import re

from app.llm.base_agent import BaseAgent
from app.rag.utils import normalize_query_keywords


INTENT_SYSTEM_PROMPT = """
당신은 RAG 시스템의 의도 분류 에이전트입니다.
사용자의 메시지와 대화 컨텍스트를 분석하여 의도를 분류합니다.

분류 가능한 의도:
- greeting
- rag
- general
- clarification
- step_navigation
- unsupported_language

응답은 항상 JSON 객체로만 반환하세요.
"""


class IntentAgent(BaseAgent):
    """LLM 기반 사용자 의도 분류 에이전트."""

    PHONETIC_MAP = {
        "피브이시": "PVC",
        "피브이": "PV",
        "오씨피": "OCP",
        "쿠버네티스": "Kubernetes",
        "인그레스": "Ingress",
        "디플로이먼트": "Deployment",
        "서비스메시": "Service Mesh",
    }

    def __init__(self, llm_client) -> None:
        super().__init__(llm_client, system_prompt=INTENT_SYSTEM_PROMPT)

    async def classify(self, user_message: str, context: dict) -> dict:
        unsupported = self._detect_unsupported_language(user_message)
        if unsupported is not None:
            return unsupported

        normalized_message = self._normalize_phonetic_terms(user_message)
        context_str = json.dumps(context or {}, ensure_ascii=False, default=str)
        result = await self.call(normalized_message, context=context_str)

        if "intent" not in result:
            result["intent"] = "general"
            result["confidence"] = 0.3

        if result.get("intent") == "rag":
            search_query = str(result.get("search_query") or normalized_message).strip()
            keywords = result.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                keywords = normalize_query_keywords(search_query)
            result["search_query"] = search_query
            result["keywords"] = keywords[:8]
        return result

    def _normalize_phonetic_terms(self, text: str) -> str:
        normalized = text
        for spoken, canonical in self.PHONETIC_MAP.items():
            normalized = re.sub(spoken, canonical, normalized, flags=re.IGNORECASE)
        return normalized

    def _detect_unsupported_language(self, text: str) -> dict | None:
        has_korean = any("\uAC00" <= ch <= "\uD7A3" for ch in text)
        has_cjk = any("\u4E00" <= ch <= "\u9FFF" for ch in text)
        if has_korean and has_cjk:
            return {
                "intent": "unsupported_language",
                "message": "한국어로 질문해 주세요.",
                "confidence": 1.0,
            }
        return None

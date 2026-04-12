from __future__ import annotations

import json

from app.llm.base_agent import BaseAgent
from app.rag.utils import normalize_domain_terms, normalize_query_keywords


INTENT_SYSTEM_PROMPT = """
You classify the user's message intent for a document-grounded RAG assistant.

Valid intents:
- greeting
- rag
- general
- clarification
- step_navigation
- unsupported_language

Return only a JSON object.
"""


class IntentAgent(BaseAgent):
    """Intent classifier with a heuristic fast-path and LLM fallback."""

    GREETING_MARKERS = ("안녕", "hello", "hi", "반가워")
    STEP_MARKERS = ("다음 단계", "다음 step", "next step", "step ", "1단계", "2단계", "3단계")
    FOLLOWUP_RAG_MARKERS = (
        "그 ",
        "그때",
        "그 다음",
        "그다음",
        "다시",
        "이어서",
        "이번에는",
        "방금",
        "같은",
        "that",
        "again",
        "continue",
    )
    EXPLAIN_MARKERS = ("설명", "차이", "관계", "명령어", "커맨드", "방법", "어떻게", "생성할 때", "기본적으로", "보통")
    UNSUPPORTED_BEGINNER_CONCEPT_PATTERNS = (
        ("storageclass", "pv"),
        ("storageclass", "pvc"),
        ("pv", "pvc"),
    )

    def __init__(self, llm_client) -> None:
        super().__init__(llm_client, system_prompt=INTENT_SYSTEM_PROMPT)

    async def classify(self, user_message: str, context: dict) -> dict:
        unsupported = self._detect_unsupported_language(user_message)
        if unsupported is not None:
            return unsupported

        normalized_message = normalize_domain_terms(user_message)
        heuristic = self._heuristic_classify(normalized_message, context or {})
        if heuristic is not None:
            return heuristic

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

        raw_doc_type = result.get("doc_type")
        if raw_doc_type and str(raw_doc_type).strip().lower() in {"operation_manual", "official"}:
            result["doc_type"] = str(raw_doc_type).strip().lower()
        else:
            result["doc_type"] = None
        return result

    def _heuristic_classify(self, normalized_message: str, context: dict) -> dict | None:
        lowered = normalized_message.casefold().strip()
        if not lowered:
            return {"intent": "general", "confidence": 0.2, "doc_type": None}
        if any(marker in lowered for marker in self.GREETING_MARKERS):
            return {"intent": "greeting", "confidence": 0.95, "doc_type": None}
        if self._looks_like_unsupported_beginner_concept(lowered):
            return {"intent": "general", "confidence": 0.9, "doc_type": None}

        procedure_state = context.get("procedure_state") or {}
        if procedure_state and any(marker in lowered for marker in self.STEP_MARKERS):
            return {"intent": "step_navigation", "step_target": "next", "confidence": 0.9, "doc_type": None}
        if self._looks_like_rag_query(lowered, context or {}):
            return {
                "intent": "rag",
                "search_query": normalized_message,
                "keywords": normalize_query_keywords(normalized_message)[:8],
                "confidence": 0.85,
                "doc_type": None,
            }
        return None

    def _looks_like_rag_query(self, lowered: str, context: dict) -> bool:
        strong_rag_hints = (
            "공식",
            "고객사",
            "문서",
            "메뉴얼",
            "매뉴얼",
            "가이드",
            "yaml",
            "cli",
            "configmap",
            "deployment",
            "service",
            "route",
            "pod",
            "pvc",
            "pv",
            "storageclass",
            "ingress",
            "oauth",
            "token requests",
            "identity provider",
            "ldap",
            "htpasswd",
            "authorization",
            "rbac",
            "mtu",
            "kubectl",
            "oc ",
            "명령어",
            "커맨드",
            "install-config",
            "compare",
            "difference",
        )
        has_rag_hint = any(marker in lowered for marker in strong_rag_hints)
        if has_rag_hint:
            return True
        if any(marker in lowered for marker in self.EXPLAIN_MARKERS) and any(resource in lowered for resource in ("pod", "route", "service", "deployment", "storageclass", "pv", "pvc", "ingress")):
            return True
        if any(marker in lowered for marker in self.FOLLOWUP_RAG_MARKERS) and context.get("selected_sources"):
            return True
        return False

    def _looks_like_unsupported_beginner_concept(self, lowered: str) -> bool:
        if not any(marker in lowered for marker in ("관계", "차이", "비교")):
            return False
        for left, right in self.UNSUPPORTED_BEGINNER_CONCEPT_PATTERNS:
            if left in lowered and right in lowered:
                return True
        return False

    @staticmethod
    def _detect_unsupported_language(text: str) -> dict | None:
        if not text or not text.strip():
            return None
        has_korean = any("\uAC00" <= ch <= "\uD7A3" or "\u1100" <= ch <= "\u11FF" or "\u3130" <= ch <= "\u318F" for ch in text)
        if has_korean:
            has_cjk = any("\u4E00" <= ch <= "\u9FFF" for ch in text)
            if has_cjk:
                return {
                    "intent": "unsupported_language",
                    "message": "한국어로 질문해 주세요.",
                    "confidence": 1.0,
                }
            return None
        if any(("\u3040" <= ch <= "\u309F") or ("\u30A0" <= ch <= "\u30FF") for ch in text):
            return {
                "intent": "unsupported_language",
                "message": "한국어로 질문해 주세요. 기술 키워드는 그대로 영어로 입력해도 됩니다.",
                "confidence": 1.0,
            }
        cjk_chars = [ch for ch in text if ("\u4E00" <= ch <= "\u9FFF") or ("\uF900" <= ch <= "\uFAFF")]
        if not cjk_chars:
            return None
        alpha_chars = [ch for ch in text if ch.isalpha()]
        cjk_ratio = len(cjk_chars) / max(len(alpha_chars), 1)
        if len(cjk_chars) >= 2 and cjk_ratio >= 0.1:
            return {
                "intent": "unsupported_language",
                "message": "한국어로 질문해 주세요. 기술 키워드는 그대로 영어로 입력해도 됩니다.",
                "confidence": 1.0,
            }
        return None

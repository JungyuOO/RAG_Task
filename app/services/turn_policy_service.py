from __future__ import annotations

from dataclasses import asdict, dataclass

from app.rag.types import ChatTurn
from app.rag.utils import normalize_text
from app.services.turn_policy_helpers import (
    build_clarification_scope_hint,
    cited_pages,
    competing_topics,
    contains_any_marker,
    has_uppercase_document_token,
    mentions_selected_source,
)


@dataclass(slots=True)
class TurnPolicyDecision:
    turn_type: str
    response_mode: str
    use_retrieval: bool
    use_memory_rewrite: bool
    allow_preview: bool
    allow_citations: bool
    needs_clarification: bool = False
    clarification_reason: str = ""
    clarification_prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TurnPolicyInput:
    user_message: str
    recent_turns: list[ChatTurn]
    summary: dict
    topic_state: dict


@dataclass(slots=True)
class _TurnPolicyContext:
    summary_topic: str
    active_topic: str
    last_retrieval_mode: str
    has_document_context: bool
    has_prior_context: bool


class TurnPolicyService:
    """Classify the current user turn and decide retrieval/clarification policy."""

    ACK_MARKERS = (
        "고마워",
        "감사",
        "감사해",
        "감사합니다",
        "좋다",
        "좋네",
        "좋아요",
        "좋습니다",
        "오케이",
        "알겠어",
        "알겠습니다",
        "그렇구나",
        "이해했어",
        "이해했습니다",
        "ㅇㅋ",
        "ㅇㅇ",
        "ok",
        "okay",
        "nice",
        "great",
        "good",
        "thanks",
        "thank you",
        "got it",
    )
    GREETING_MARKERS = (
        "안녕",
        "안녕하세요",
        "hi",
        "hello",
        "hey",
    )
    DOCUMENT_INTENT_MARKERS = (
        "무엇",
        "뭐",
        "어디",
        "언제",
        "왜",
        "어떻게",
        "설명",
        "정리",
        "비교",
        "차이",
        "종류",
        "찾아",
        "알려",
        "보여",
        "페이지",
        "문서",
        "pdf",
        "출처",
        "?",
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "compare",
        "difference",
        "explain",
        "page",
        "source",
        "document",
    )
    FOLLOW_UP_MARKERS = (
        "그거",
        "그건",
        "그 문서",
        "그 페이지",
        "다시",
        "이전",
        "방금",
        "그럼",
        "그리고",
        "that",
        "this",
        "those",
        "again",
        "previous",
        "above",
        "what about",
        "how about",
        "then",
        "also",
    )
    GENERAL_CHAT_MARKERS = (
        "추천",
        "잡담",
        "기분",
        "오늘",
        "취미",
        "일정",
        "recommend",
    )
    CASUAL_CHAT_MARKERS = (
        "너는",
        "넌",
        "이름",
        "누구",
        "뭐해",
        "뭐야",
        "무슨 일 해",
        "정체",
        "자기소개",
        "심심해",
        "모르겠어",
        "됐어",
        "who are you",
        "what are you",
        "what do you do",
        "your name",
    )
    CLARIFICATION_REFERENT_MARKERS = (
        "그거",
        "그건",
        "그 내용",
        "그 예시",
        "그 코드",
        "그 yaml",
        "다시 설명",
        "다시 보여",
        "예시 보여줘",
        "코드 보여줘",
        "that",
        "this",
        "those",
        "that one",
        "show me the example",
        "show the code",
        "example please",
        "code please",
        "explain again",
    )
    CODE_REQUEST_MARKERS = (
        "yaml",
        "manifest",
        "code",
        "example",
        "sample",
        "demo",
        "cli",
        "코드",
        "예시",
        "샘플",
        "보여줘",
    )
    GENERIC_FOCUS_MARKERS = (
        "문서",
        "페이지",
        "예시",
        "코드",
        "설명",
        "example",
        "code",
        "document",
        "page",
    )

    def classify_turn(
        self,
        user_message: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
    ) -> TurnPolicyDecision:
        return self.classify(
            TurnPolicyInput(
                user_message=user_message,
                recent_turns=recent_turns,
                summary=summary,
                topic_state=topic_state,
            )
        )

    def classify(self, policy_input: TurnPolicyInput) -> TurnPolicyDecision:
        normalized = self._normalize(policy_input.user_message)
        if not normalized:
            return self._general_chat()

        context = self._build_context(
            policy_input.recent_turns,
            policy_input.summary,
            policy_input.topic_state,
        )

        if self._is_greeting(normalized):
            return self._decision("greeting", "general", False, False, False, False)

        if self._is_conversational_ack(normalized, context):
            return self._decision("conversational_ack", "conversational", False, False, False, False)

        clarification = self._build_clarification_decision(
            normalized,
            policy_input.recent_turns,
            policy_input.summary,
            policy_input.topic_state,
            context,
        )
        if clarification is not None:
            return clarification

        if self._is_document_follow_up(
            normalized,
            policy_input.recent_turns,
            policy_input.summary,
            policy_input.topic_state,
            context,
        ):
            return self._decision("document_followup", "rag", True, True, True, True)

        if self._is_document_query(normalized, policy_input.topic_state):
            return self._decision("document_query", "rag", True, context.has_prior_context, True, True)

        return self._general_chat()

    def _normalize(self, value: str) -> str:
        return " ".join(normalize_text(value).lower().split())

    def _decision(
        self,
        turn_type: str,
        response_mode: str,
        use_retrieval: bool,
        use_memory_rewrite: bool,
        allow_preview: bool,
        allow_citations: bool,
    ) -> TurnPolicyDecision:
        return TurnPolicyDecision(
            turn_type,
            response_mode,
            use_retrieval,
            use_memory_rewrite,
            allow_preview,
            allow_citations,
        )

    def _general_chat(self) -> TurnPolicyDecision:
        return self._decision("general_chat", "general", False, False, False, False)

    def _build_context(
        self,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
    ) -> _TurnPolicyContext:
        last_assistant_turn = next((turn for turn in reversed(recent_turns) if turn.role == "assistant"), None)
        last_assistant_metadata = (last_assistant_turn.metadata or {}) if last_assistant_turn else {}
        return _TurnPolicyContext(
            summary_topic=str(summary.get("topic") or "").strip(),
            active_topic=str(topic_state.get("active_topic") or "").strip(),
            last_retrieval_mode=str(
                topic_state.get("last_retrieval_mode") or last_assistant_metadata.get("mode") or ""
            ),
            has_document_context=self._has_document_context(topic_state, last_assistant_metadata),
            has_prior_context=self._has_prior_context(topic_state),
        )

    def _has_document_context(self, topic_state: dict, last_assistant_metadata: dict) -> bool:
        return bool(
            topic_state.get("active_topic")
            or topic_state.get("selected_sources")
            or topic_state.get("last_answer_citations")
            or last_assistant_metadata.get("answer_citations")
            or last_assistant_metadata.get("preview_pages")
        )

    def _has_prior_context(self, topic_state: dict) -> bool:
        return bool(
            topic_state.get("active_topic")
            or topic_state.get("selected_sources")
            or topic_state.get("last_user_focus")
        )

    def _has_resolved_focus(self, topic_state: dict) -> bool:
        return bool(topic_state.get("last_user_focus"))

    def _has_follow_up_marker(self, normalized: str) -> bool:
        return contains_any_marker(normalized, self.FOLLOW_UP_MARKERS)

    def _has_document_intent(self, normalized: str) -> bool:
        substantive_markers = tuple(marker for marker in self.DOCUMENT_INTENT_MARKERS if marker != "?")
        return contains_any_marker(normalized, substantive_markers)

    def _mentions_selected_source(self, normalized: str, topic_state: dict) -> bool:
        return mentions_selected_source(normalized, topic_state)

    def _has_uppercase_document_token(self, normalized: str) -> bool:
        return has_uppercase_document_token(normalized)

    def _is_question_like_document_query(self, normalized: str) -> bool:
        return "?" in normalized and len(normalized) >= 12

    def _should_treat_as_short_contextual_follow_up(
        self,
        normalized: str,
        topic_state: dict,
        context: _TurnPolicyContext,
    ) -> bool:
        if self._has_resolved_focus(topic_state) and len(normalized) <= 32 and not self._looks_like_general_chat(normalized):
            return True
        if len(normalized) <= 24 and (context.summary_topic or context.active_topic) and not self._looks_like_general_chat(normalized):
            return True
        if len(normalized) <= 40 and topic_state.get("selected_sources") and not self._looks_like_general_chat(normalized):
            return True
        return False

    def _follow_up_has_enough_detail(self, normalized: str) -> bool:
        return self._has_document_intent(normalized) or len(normalized) >= 15

    def _has_multiple_scope_candidates(
        self,
        recent_turns: list[ChatTurn],
        topic_state: dict,
    ) -> bool:
        return len(self._extract_scope_candidates(recent_turns, topic_state)) >= 2

    def _should_clarify_from_topic_competition(
        self,
        normalized: str,
        topic_state: dict,
        context: _TurnPolicyContext,
    ) -> bool:
        last_user_focus = self._normalize(str(topic_state.get("last_user_focus") or ""))
        seen_topics = self._competing_topics(topic_state)
        pages = self._cited_pages(topic_state)
        selected_pages = [int(page) for page in topic_state.get("selected_pages", []) if str(page).isdigit()]
        focus_is_specific = self._focus_is_specific(last_user_focus)

        if focus_is_specific and len(seen_topics) <= 1:
            return False
        if len(seen_topics) >= 2:
            return True
        if len(set(pages)) >= 2 and len(set(selected_pages)) >= 2:
            return True
        if len(set(pages)) >= 2 and not focus_is_specific:
            return True
        if not focus_is_specific and len(normalized) <= 18 and (context.summary_topic or context.active_topic):
            return True
        return False

    def _build_clarification_scope_hint(self, topic_state: dict, context: _TurnPolicyContext) -> str:
        return build_clarification_scope_hint(
            topic_state,
            context.summary_topic,
            context.active_topic,
            self._normalize,
        )

    def _is_greeting(self, normalized: str) -> bool:
        return normalized in self.GREETING_MARKERS

    def _is_conversational_ack(self, normalized: str, context: _TurnPolicyContext) -> bool:
        if not contains_any_marker(normalized, self.ACK_MARKERS):
            return False
        if context.last_retrieval_mode == "rag" and context.has_document_context:
            return len(normalized) <= 40
        return normalized in self.ACK_MARKERS

    def _is_document_follow_up(
        self,
        normalized: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
        context: _TurnPolicyContext,
    ) -> bool:
        if not context.has_document_context or context.last_retrieval_mode != "rag":
            return False
        if self._should_clarify_referent(normalized, recent_turns, summary, topic_state, context):
            return False
        if self._has_follow_up_marker(normalized) and self._follow_up_has_enough_detail(normalized):
            return True
        return self._should_treat_as_short_contextual_follow_up(normalized, topic_state, context)

    def _is_document_query(self, normalized: str, topic_state: dict) -> bool:
        if self._looks_like_casual_chat(normalized):
            return False
        if self._has_document_intent(normalized) and len(normalized) >= 4:
            return True
        if self._is_question_like_document_query(normalized):
            return True
        if self._mentions_selected_source(normalized, topic_state):
            return True
        return self._has_uppercase_document_token(normalized)

    def _looks_like_general_chat(self, normalized: str) -> bool:
        return contains_any_marker(normalized, self.GENERAL_CHAT_MARKERS)

    def _looks_like_casual_chat(self, normalized: str) -> bool:
        return contains_any_marker(normalized, self.CASUAL_CHAT_MARKERS)

    def _build_clarification_decision(
        self,
        normalized: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
        context: _TurnPolicyContext,
    ) -> TurnPolicyDecision | None:
        if not context.has_document_context or context.last_retrieval_mode != "rag":
            return None
        if not self._should_clarify_referent(normalized, recent_turns, summary, topic_state, context):
            return None

        scope_hint = self._build_clarification_scope_hint(topic_state, context)
        prompt = (
            "어떤 부분을 말씀하시는지 조금만 더 구체적으로 알려주세요. "
            f"지금은 {scope_hint}처럼 후보가 여러 개라서 바로 하나로 특정하기 어렵습니다. "
            "예를 들어 `PV/PVC 예시`, `StorageClass 예시`, `정적 프로비저닝 설명`처럼 말씀해주시면 바로 이어서 답하겠습니다."
        )
        return TurnPolicyDecision(
            "clarification",
            "clarification",
            False,
            False,
            False,
            False,
            needs_clarification=True,
            clarification_reason="ambiguous_referent",
            clarification_prompt=prompt,
        )

    def _should_clarify_referent(
        self,
        normalized: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
        context: _TurnPolicyContext,
    ) -> bool:
        del summary
        looks_like_candidate = self._looks_like_clarification_candidate(normalized)
        is_code_request = self._is_code_request(normalized)

        if not (looks_like_candidate and is_code_request):
            return False
        if self._has_multiple_scope_candidates(recent_turns, topic_state):
            return True
        return self._should_clarify_from_topic_competition(normalized, topic_state, context)

    def _competing_topics(self, topic_state: dict) -> list[str]:
        return competing_topics(topic_state, self._normalize)

    def _cited_pages(self, topic_state: dict) -> list[int]:
        return cited_pages(topic_state)

    def _looks_like_clarification_candidate(self, normalized: str) -> bool:
        return contains_any_marker(normalized, self.CLARIFICATION_REFERENT_MARKERS)

    def _is_code_request(self, normalized: str) -> bool:
        return contains_any_marker(normalized, self.CODE_REQUEST_MARKERS)

    def _focus_is_specific(self, normalized_focus: str) -> bool:
        if not normalized_focus:
            return False
        informative_parts = [
            part
            for part in normalized_focus.split()
            if part and part not in self.GENERIC_FOCUS_MARKERS
        ]
        return bool(informative_parts)

    def _extract_scope_candidates(self, recent_turns: list[ChatTurn], topic_state: dict) -> list[str]:
        candidates: list[str] = []

        for item in topic_state.get("recent_user_topics", []):
            normalized = self._normalize(str(item))
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        recent_assistant = next((turn for turn in reversed(recent_turns) if turn.role == "assistant"), None)
        if recent_assistant:
            for token in recent_assistant.content.split():
                cleaned = token.strip(".,;:()[]{}!?\"'")
                if len(cleaned) >= 2 and cleaned.isupper() and cleaned.isalpha():
                    lowered = cleaned.lower()
                    if lowered not in candidates:
                        candidates.append(lowered)

        return candidates

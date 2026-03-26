from __future__ import annotations

from dataclasses import asdict, dataclass

from app.rag.types import ChatTurn
from app.rag.utils import normalize_text


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


class TurnPolicyService:
    """사용자 메시지의 턴 유형을 분류하고 검색/응답 전략을 결정하는 서비스.

    인사, 승인, 후속 질문, 문서 질의, 명확화 요청 등을 구분하여
    불필요한 검색을 차단하고 적절한 응답 모드를 선택한다.
    """

    ACK_MARKERS = (
        "고마워",
        "감사",
        "감사해",
        "감사합니다",
        "고맙",
        "좋다",
        "좋네",
        "좋아요",
        "좋습니다",
        "오 좋다",
        "오 좋네",
        "굿",
        "좋군요",
        "좋아",
        "오케이",
        "ok",
        "okay",
        "알겠어",
        "알겠습니다",
        "그렇구나",
        "이해했어",
        "이해했습니다",
        "ㅇㅇ",
        "ㅇㅋ",
        "ㅇㅋㅇㅋ",
        "ㄱㅅ",
        "ㄳ",
        "넵",
        "네",
        "응",
        "웅",
        "ㅎㅎ",
        "ㅋㅋ",
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
        # 의문사 (한국어)
        "무엇", "뭐", "어디", "언제", "왜", "어떻게",
        # 요청 동사 (한국어)
        "설명", "정리", "비교", "차이", "종류",
        "찾아", "알려", "보여",
        # 문서/출처 참조
        "페이지", "문서", "pdf", "출처",
        # 물음표
        "?",
        # 의문사 (영어)
        "what", "which", "where", "when", "why", "how",
        # 요청 동사 (영어)
        "compare", "difference", "explain",
        "page", "source", "document",
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
        "비교",
        "차이",
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
        "날씨",
        "일정",
        "recommend",
    )
    CASUAL_CHAT_MARKERS = (
        "너",
        "넌",
        "니",
        "애니",
        "이름",
        "누구",
        "뭐해",
        "뭐 하",
        "뭐하는",
        "무슨 역할",
        "정체",
        "자기소개",
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
        "데모",
        "매니페스트",
        "보여줘",
        "적어줘",
    )
    GENERIC_FOCUS_MARKERS = (
        "문서", "페이지", "예시", "코드", "설명",
        "example", "code", "document", "page",
    )
    # 도메인 키워드를 하드코딩하지 않음 — 인덱싱된 문서의 존재 여부와
    # 의문사/요청 동사 기반으로 문서 검색 필요성을 판단한다.

    def classify_turn(
        self,
        user_message: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
    ) -> TurnPolicyDecision:
        raw_lower = user_message.strip().lower()
        normalized = self._normalize(user_message)
        if not normalized:
            return TurnPolicyDecision("general_chat", "general", False, False, False, False)

        last_assistant_turn = next((turn for turn in reversed(recent_turns) if turn.role == "assistant"), None)
        last_assistant_metadata = (last_assistant_turn.metadata or {}) if last_assistant_turn else {}
        last_retrieval_mode = str(topic_state.get("last_retrieval_mode") or last_assistant_metadata.get("mode") or "")
        had_document_grounding = bool(
            topic_state.get("selected_sources")
            or topic_state.get("last_answer_citations")
            or last_assistant_metadata.get("answer_citations")
            or last_assistant_metadata.get("preview_pages")
        )

        if self._is_greeting(normalized):
            return TurnPolicyDecision("greeting", "general", False, False, False, False)

        if self._is_conversational_ack(normalized, last_retrieval_mode, had_document_grounding):
            return TurnPolicyDecision("conversational_ack", "conversational", False, False, False, False)

        clarification = self._build_clarification_decision(
            raw_lower,
            normalized,
            recent_turns,
            summary,
            topic_state,
            last_retrieval_mode,
            had_document_grounding,
        )
        if clarification is not None:
            return clarification

        if self._is_document_follow_up(raw_lower, normalized, recent_turns, summary, topic_state, last_retrieval_mode, had_document_grounding):
            return TurnPolicyDecision("document_followup", "rag", True, True, True, True)

        if self._is_document_query(normalized, topic_state):
            return TurnPolicyDecision("document_query", "rag", True, False, True, True)

        return TurnPolicyDecision("general_chat", "general", False, False, False, False)

    def _normalize(self, value: str) -> str:
        return " ".join(normalize_text(value).lower().split())

    def _is_greeting(self, normalized: str) -> bool:
        return normalized in self.GREETING_MARKERS

    def _is_conversational_ack(
        self,
        normalized: str,
        last_retrieval_mode: str,
        had_document_grounding: bool,
    ) -> bool:
        if not any(marker in normalized for marker in self.ACK_MARKERS):
            return False
        # 이전 RAG 컨텍스트가 있으면 40자 이하 허용,
        # 없어도 ACK 마커와 정확히 일치하면 의미 없는 메시지로 판단
        if last_retrieval_mode == "rag" and had_document_grounding:
            return len(normalized) <= 40
        return normalized in self.ACK_MARKERS

    def _is_document_follow_up(
        self,
        raw_lower: str,
        normalized: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
        last_retrieval_mode: str,
        had_document_grounding: bool,
    ) -> bool:
        if not had_document_grounding or last_retrieval_mode != "rag":
            return False
        if self._should_clarify_referent(raw_lower, normalized, recent_turns, summary, topic_state):
            return False
        if any(marker in normalized for marker in self.FOLLOW_UP_MARKERS):
            # 후속 마커만으로는 부족 — 문서 관련 의도(설명, 알려, 보여 등)가 함께 있거나
            # 충분히 구체적인 메시지(15자 이상)여야 문서 후속 질문으로 판단한다.
            # "그거 알아?", "그거 뭐야" 같은 짧은 일상 표현은 제외.
            doc_intent_markers = ("설명", "정리", "비교", "차이", "종류", "찾아", "알려", "보여", "어떻게")
            has_doc_intent = any(m in normalized for m in doc_intent_markers)
            if has_doc_intent or len(normalized) >= 15:
                return True
        if topic_state.get("last_user_focus") and len(normalized) <= 32 and not self._looks_like_general_chat(normalized):
            return True
        if len(normalized) <= 24 and summary.get("topic") and not self._looks_like_general_chat(normalized):
            return True
        if len(normalized) <= 40 and topic_state.get("selected_sources") and not self._looks_like_general_chat(normalized):
            return True
        return False

    def _is_document_query(self, normalized: str, topic_state: dict) -> bool:
        """문서 질의 여부를 판단한다.

        의문사/요청 동사가 포함되거나, 현재 선택된 출처 파일명이
        질문에 언급되거나, 대문자 약어/전문 용어가 포함되면 문서 검색을 수행한다.
        하드코딩된 도메인 키워드 목록 대신 형태적 특성으로 판별한다.
        """
        if self._looks_like_casual_chat(normalized):
            return False
        # "?"만으로는 문서 질의로 판단하지 않음 — 실질적인 의문사/요청 동사가 필요
        substantive_markers = [m for m in self.DOCUMENT_INTENT_MARKERS if m != "?"]
        has_substantive_intent = any(marker in normalized for marker in substantive_markers)
        has_question_mark = "?" in normalized
        # 의문사/요청 동사가 있어도 메시지가 너무 짧으면 (예: "뭐") 의미 없는 입력
        if has_substantive_intent and len(normalized) >= 4:
            return True
        # "?"만 있고 실질적 마커가 없으면 최소 길이 이상이어야 문서 질의로 판단
        if has_question_mark and len(normalized) >= 12:
            return True
        for source in topic_state.get("selected_sources", []):
            source_name = str(source).lower()
            if source_name.endswith(".pdf") and source_name in normalized:
                return True
        # 대문자 약어/전문 용어가 포함되면 문서 검색 수행 (PV, PVC, YAML 등)
        tokens = normalized.split()
        for token in tokens:
            if len(token) >= 2 and token.isupper() and token.isalpha():
                return True
        return False

    def _looks_like_general_chat(self, normalized: str) -> bool:
        return any(marker in normalized for marker in self.GENERAL_CHAT_MARKERS)

    def _looks_like_casual_chat(self, normalized: str) -> bool:
        return any(marker in normalized for marker in self.CASUAL_CHAT_MARKERS)

    def _build_clarification_decision(
        self,
        raw_lower: str,
        normalized: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
        last_retrieval_mode: str,
        had_document_grounding: bool,
    ) -> TurnPolicyDecision | None:
        if not had_document_grounding or last_retrieval_mode != "rag":
            return None
        if not self._should_clarify_referent(raw_lower, normalized, recent_turns, summary, topic_state):
            return None

        active_topic = str(topic_state.get("active_topic") or summary.get("topic") or "").strip()
        competing_topics = self._competing_topics(topic_state)
        cited_pages = self._cited_pages(topic_state)
        page_hint = ", ".join(f"p.{page}" for page in cited_pages[:3])
        topic_hint = ", ".join(competing_topics[:3])
        scope_hint = topic_hint or page_hint or (active_topic or "방금 설명한 내용")
        prompt = (
            "어느 부분을 말씀하시는지 조금만 더 구체적으로 알려주세요. "
            f"지금은 {scope_hint}처럼 후보가 여러 개라서 바로 하나로 정하기 어렵습니다. "
            "예를 들어 `PV/PVC 예시`, `StorageClass 예시`, `정적 프로비저닝 설명`처럼 말씀해 주시면 바로 이어서 답변드릴게요."
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
        raw_lower: str,
        normalized: str,
        recent_turns: list[ChatTurn],
        summary: dict,
        topic_state: dict,
    ) -> bool:
        last_user_focus = self._normalize(str(topic_state.get("last_user_focus") or ""))
        competing_topics = self._competing_topics(topic_state)
        cited_pages = self._cited_pages(topic_state)
        selected_pages = [int(page) for page in topic_state.get("selected_pages", []) if str(page).isdigit()]
        scope_candidates = self._extract_scope_candidates(recent_turns, topic_state)
        looks_like_candidate = self._looks_like_clarification_candidate(raw_lower, normalized)
        is_code_request = self._is_code_request(raw_lower, normalized)

        if not (looks_like_candidate and is_code_request):
            return False

        if len(scope_candidates) >= 2:
            return True
        focus_is_specific = self._focus_is_specific(last_user_focus)

        if focus_is_specific and len(competing_topics) <= 1:
            return False
        if len(competing_topics) >= 2:
            return True
        if len(set(cited_pages)) >= 2 and len(set(selected_pages)) >= 2:
            return True
        if len(set(cited_pages)) >= 2 and not focus_is_specific:
            return True
        if not focus_is_specific and len(normalized) <= 18 and (summary.get("topic") or topic_state.get("active_topic")):
            return True
        return False

    def _competing_topics(self, topic_state: dict) -> list[str]:
        topics: list[str] = []
        for item in topic_state.get("recent_user_topics", []):
            normalized = self._normalize(str(item))
            if normalized and normalized not in topics:
                topics.append(normalized)
        return topics

    def _cited_pages(self, topic_state: dict) -> list[int]:
        pages: list[int] = []
        for item in topic_state.get("last_answer_citations", []) or []:
            value = item.get("page_number")
            if value is None:
                continue
            if str(value).isdigit():
                page = int(value)
                if page not in pages:
                    pages.append(page)
        return pages

    def _looks_like_clarification_candidate(self, raw_lower: str, normalized: str) -> bool:
        return any(marker in raw_lower or marker in normalized for marker in self.CLARIFICATION_REFERENT_MARKERS)

    def _is_code_request(self, raw_lower: str, normalized: str) -> bool:
        return any(marker in raw_lower or marker in normalized for marker in self.CODE_REQUEST_MARKERS)

    def _focus_is_specific(self, normalized_focus: str) -> bool:
        if not normalized_focus:
            return False
        parts = [part for part in normalized_focus.split() if part]
        informative_parts = [
            part
            for part in parts
            if part not in self.GENERIC_FOCUS_MARKERS
        ]
        return bool(informative_parts)

    def _extract_scope_candidates(self, recent_turns: list[ChatTurn], topic_state: dict) -> list[str]:
        """최근 대화에서 경쟁하는 토픽 후보를 동적으로 추출한다.

        하드코딩된 도메인 키워드 대신, 세션 토픽 상태의 recent_user_topics와
        최근 어시스턴트 답변의 엔티티(대문자 약어, 한글 명사 등)를 후보로 사용한다.
        """
        candidates: list[str] = []

        # recent_user_topics에서 후보 추출
        for item in topic_state.get("recent_user_topics", []):
            normalized = self._normalize(str(item))
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        # 최근 어시스턴트 답변에서 대문자 약어 추출 (PV, PVC, RBAC 등)
        recent_assistant = next(
            (turn for turn in reversed(recent_turns) if turn.role == "assistant"), None,
        )
        if recent_assistant:
            for token in recent_assistant.content.split():
                cleaned = token.strip(".,;:()[]{}!?\"'")
                if len(cleaned) >= 2 and cleaned.isupper() and cleaned.isalpha():
                    lowered = cleaned.lower()
                    if lowered not in candidates:
                        candidates.append(lowered)

        return candidates

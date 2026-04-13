"""Chat-turn orchestration for the streaming RAG flow."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.bge_embeddings import EmbeddingModelUnavailableError
from app.rag.types import TurnPolicyDecision

logger = logging.getLogger("rag.pipeline")
_VERSION_PATTERN = re.compile(r"(?<!\d)(4\.(?:15|16|17|18|19|20|21))(?!\d)")

STAGE_MESSAGES = {
    "analyzing_intent": "질문 의도를 분석하는 중입니다.",
    "searching_documents": "자료실에서 관련 내용을 검색하는 중입니다.",
    "evaluating_relevance": "검색 결과의 관련도를 평가하는 중입니다.",
    "generating_answer": "답변을 생성하는 중입니다.",
}


@dataclass(slots=True)
class ChatTurnDeps:
    detect_non_korean_query: Any
    session_repository: Any
    should_skip_procedure_shortcut: Any
    detect_procedure_followup: Any
    build_procedure_followup_answer: Any
    looks_like_step_navigation_without_state: Any
    resolve_turn_context: Any
    domain_guard_state: Any
    prepare_retrieval_state: Any
    resolve_answer_route: Any
    interleave_context_items_by_source: Any
    build_context_blocks: Any
    ensure_topic_for_resolution: Any
    build_answer_cache_key: Any
    canonical_cache_query: Any
    build_policy_answer: Any
    build_missing_extractive_answer: Any
    select_code_example_context_items: Any
    resolve_requested_resource_kinds: Any
    prefer_block_type_items: Any
    finalize_answer: Any
    store_assistant_turn: Any
    build_llm_failure_fallback: Any
    get_prompt_composer: Any
    answer_service: Any
    answer_cache_repository: Any
    answer_rewrite_agent: Any
    llm: Any


class StreamingTurnSupport:
    async def _stream_version_clarification(self, session_id, user_message, deps, available_versions):
        versions_str = " / ".join(available_versions) if available_versions else "4.20"
        clarification_msg = (
            f"어떤 버전의 OpenShift Container Platform을 기준으로 답변할까요?\n\n"
            f"지원 버전: **{versions_str}**\n\n"
            f"버전을 명시해 주세요."
        )
        for char in clarification_msg:
            yield {"type": "token", "content": char, "cached": False}
            await asyncio.sleep(0.02)
        final_payload = deps.answer_service.build_context_payload(
            user_message.strip(), "clarification", 0.0, None, [], [], [], [],
            preview_finalized=True,
        )
        yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
        resolved_topic_id = None
        deps.store_assistant_turn(session_id, clarification_msg, final_payload, resolved_topic_id)
        yield {"type": "done", "cached": False}

    async def _handle_terminal_policy_answers(self, **kwargs):
        deps = self.deps
        session_id = kwargs["session_id"]
        rewritten_query = kwargs["rewritten_query"]
        top_score = kwargs["top_score"]
        policy_decision: TurnPolicyDecision = kwargs["policy_decision"]
        resolved_topic_id = kwargs["resolved_topic_id"]
        cache_key = kwargs["cache_key"]

        async def _emit(answer: str, mode: str, cache_answer: bool = False):
            for char in answer:
                yield {"type": "token", "content": char, "cached": False}
                await asyncio.sleep(0.03)
            final_payload = deps.answer_service.build_context_payload(
                rewritten_query, mode, top_score, None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
            deps.store_assistant_turn(session_id, answer, final_payload, resolved_topic_id)
            if cache_answer:
                deps.answer_cache_repository.set(cache_key, {"answer": answer})
            yield {"type": "done", "cached": False}

        if policy_decision.turn_type == "conversational_ack":
            return _emit(deps.build_policy_answer("conversational_ack", top_score), "conversational", False)
        if policy_decision.turn_type == "greeting":
            return _emit(deps.build_policy_answer("greeting", top_score), "greeting", False)
        if policy_decision.turn_type == "general_chat":
            return _emit(deps.build_policy_answer("general_chat", top_score), "general", False)
        if policy_decision.turn_type == "document_query" and not kwargs["use_retrieved_context"]:
            return _emit(deps.build_policy_answer("document_query", top_score), "general", False)
        return None

    async def _handle_extractive_routes(self, **kwargs):
        deps = self.deps
        session_id = kwargs["session_id"]
        user_message = kwargs["user_message"]
        rewritten_query = kwargs["rewritten_query"]
        top_score = kwargs["top_score"]
        use_retrieved_context = kwargs["use_retrieved_context"]
        grounded_pages = kwargs["grounded_pages"]
        ordered_context_items = kwargs["ordered_context_items"]
        selected_context_items = kwargs["selected_context_items"]
        preferred_preview_source = kwargs["preferred_preview_source"]
        response_mode = kwargs["response_mode"]
        policy_decision = kwargs["policy_decision"]
        query_interpretation = kwargs["query_interpretation"]
        answer_route = kwargs["answer_route"]
        doc_type = kwargs.get("doc_type", "")
        resolved_topic_id = kwargs["resolved_topic_id"]
        cache_key = kwargs["cache_key"]

        async def _emit(final_answer: str, final_payload: dict):
            yield {"type": "token", "content": final_answer, "cached": False}
            yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
            deps.store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
            deps.answer_cache_repository.set(cache_key, {"answer": final_answer})
            yield {"type": "done", "cached": False}

        async def _no_extractive_answer():
            no_answer = deps.build_missing_extractive_answer(answer_route)
            yield {"type": "token", "content": no_answer, "cached": False}
            final_payload = deps.answer_service.build_context_payload(
                rewritten_query, "clarification", top_score, None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
            deps.store_assistant_turn(session_id, no_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}

        if answer_route == "extractive_text" and use_retrieved_context:
            extractive_text_answer = deps.answer_service.build_extractive_text_answer(selected_context_items)
            if extractive_text_answer:
                expected_shape = self._expected_response_shape(query_interpretation, user_message)
                if expected_shape == "comparison":
                    compare_answer = deps.answer_service.build_extractive_compare_answer(selected_context_items)
                    if compare_answer:
                        extractive_text_answer = compare_answer
                try:
                    rewritten_extractive = await deps.answer_rewrite_agent.rewrite(
                        user_message,
                        extractive_text_answer,
                    )
                    if rewritten_extractive:
                        extractive_text_answer = rewritten_extractive
                except Exception:
                    pass
                if expected_shape and not self._answer_matches_shape(extractive_text_answer, expected_shape):
                    extractive_text_answer = self._coerce_answer_shape(extractive_text_answer, expected_shape)
                supporting_examples = self._build_supporting_examples(
                    answer=extractive_text_answer,
                    deps=deps,
                    user_message=user_message,
                    query_interpretation=query_interpretation,
                    ordered_context_items=ordered_context_items,
                    selected_context_items=selected_context_items,
                )
                final_answer, _answer_citations, final_payload = deps.finalize_answer(
                    answer=extractive_text_answer,
                    rewritten_query=rewritten_query,
                    use_retrieved_context=use_retrieved_context,
                    top_score=top_score,
                    selected_context_items=selected_context_items,
                    grounded_pages=grounded_pages,
                    preferred_preview_source=preferred_preview_source,
                    response_mode=response_mode,
                    policy_decision=policy_decision,
                    query_interpretation=query_interpretation,
                    answer_route=answer_route,
                    doc_type=doc_type,
                )
                if supporting_examples:
                    final_payload["supporting_examples"] = supporting_examples
                return _emit(final_answer, final_payload)
            return _no_extractive_answer()

        if answer_route == "extractive_compare" and use_retrieved_context:
            extractive_compare_answer = deps.answer_service.build_extractive_compare_answer(selected_context_items)
            if extractive_compare_answer:
                if not self._answer_matches_shape(extractive_compare_answer, "comparison"):
                    extractive_compare_answer = self._coerce_answer_shape(extractive_compare_answer, "comparison")
                final_answer, _answer_citations, final_payload = deps.finalize_answer(
                    answer=extractive_compare_answer,
                    rewritten_query=rewritten_query,
                    use_retrieved_context=use_retrieved_context,
                    top_score=top_score,
                    selected_context_items=selected_context_items,
                    grounded_pages=grounded_pages,
                    preferred_preview_source=preferred_preview_source,
                    response_mode=response_mode,
                    policy_decision=policy_decision,
                    query_interpretation=query_interpretation,
                    answer_route=answer_route,
                    doc_type=doc_type,
                )
                return _emit(final_answer, final_payload)
            return _no_extractive_answer()

        if answer_route == "extractive_code" and use_retrieved_context:
            code_context_items = deps.select_code_example_context_items(
                user_message,
                query_interpretation,
                ordered_context_items,
                selected_context_items,
            )
            extractive_code_answer = deps.answer_service.build_extractive_code_answer(
                code_context_items,
                requested_resource_kinds=deps.resolve_requested_resource_kinds(query_interpretation),
                user_message=user_message,
                query_interpretation=query_interpretation,
            )
            if extractive_code_answer:
                final_answer, _answer_citations, final_payload = deps.finalize_answer(
                    answer=extractive_code_answer,
                    rewritten_query=rewritten_query,
                    use_retrieved_context=use_retrieved_context,
                    top_score=top_score,
                    selected_context_items=code_context_items,
                    grounded_pages=grounded_pages,
                    preferred_preview_source=preferred_preview_source,
                    response_mode=response_mode,
                    policy_decision=policy_decision,
                    query_interpretation=query_interpretation,
                    answer_route=answer_route,
                    doc_type=doc_type,
                )
                final_payload["last_example_anchor"] = deps.answer_service.build_example_anchor(
                    code_context_items,
                    query_interpretation,
                )
                return _emit(final_answer, final_payload)
            return _no_extractive_answer()

        if answer_route == "extractive_table" and use_retrieved_context:
            table_context_items = deps.prefer_block_type_items(
                ordered_context_items or selected_context_items,
                block_type="table",
                limit=max(len(selected_context_items), 3),
            ) or selected_context_items
            extractive_table_answer = deps.answer_service.build_extractive_table_answer(table_context_items)
            if extractive_table_answer:
                final_answer, _answer_citations, final_payload = deps.finalize_answer(
                    answer=extractive_table_answer,
                    rewritten_query=rewritten_query,
                    use_retrieved_context=use_retrieved_context,
                    top_score=top_score,
                    selected_context_items=table_context_items,
                    grounded_pages=grounded_pages,
                    preferred_preview_source=preferred_preview_source,
                    response_mode=response_mode,
                    policy_decision=policy_decision,
                    query_interpretation=query_interpretation,
                    answer_route=answer_route,
                    doc_type=doc_type,
                )
                return _emit(final_answer, final_payload)
            return _no_extractive_answer()

        return None


class ChatTurnOrchestrator(StreamingTurnSupport):
    def __init__(self, deps: ChatTurnDeps) -> None:
        self.deps = deps

    @staticmethod
    def _resolve_effective_version_tag(
        version_tag: str | None,
        user_message: str | None = None,
        *topic_states: dict | None,
    ) -> str | None:
        if version_tag:
            return version_tag
        if user_message:
            match = _VERSION_PATTERN.search(user_message.strip())
            if match:
                return match.group(1)
        for topic_state in topic_states:
            if not isinstance(topic_state, dict):
                continue
            selected_versions = [str(value).strip() for value in topic_state.get("selected_versions", []) or [] if value]
            if selected_versions:
                return selected_versions[0]
        return None

    def _detect_version_selection(self, session_id: str, user_message: str, version_tag: str | None) -> tuple[str | None, str | None]:
        """이전 답변이 버전 확인 질문이고 현재 메시지가 버전 번호면 (원래 질문, 버전)을 반환한다."""
        if version_tag is not None:
            return None, None
        match = _VERSION_PATTERN.search(user_message.strip())
        if not match:
            return None, None
        selected_version = match.group(1)
        recent = self.deps.session_repository.recent_turns(session_id)
        if len(recent) < 2:
            return None, None
        last_assistant = None
        original_user_msg = None
        for i in range(len(recent) - 1, -1, -1):
            turn = recent[i]
            if turn.role == "assistant" and last_assistant is None:
                last_assistant = turn
            elif turn.role == "user" and last_assistant is not None:
                original_user_msg = turn.content
                break
        if last_assistant is None:
            return None, None
        meta = last_assistant.metadata if isinstance(last_assistant.metadata, dict) else {}
        is_clarification = (
            meta.get("mode") == "clarification"
            or meta.get("response_mode") == "clarification"
            or "어떤 버전의 OpenShift" in (last_assistant.content or "")
            or "어떤 버전의 openshift" in (last_assistant.content or "").lower()
        )
        if not is_clarification:
            return None, None
        if original_user_msg:
            return original_user_msg, selected_version
        return None, None


    async def _try_transform_previous_answer(
        self,
        session_id: str,
        user_message: str,
        append_user_turn: bool,
        topic_state_before: dict,
    ) -> AsyncIterator[dict] | None:
        deps = self.deps
        recent_turns = deps.session_repository.recent_turns(session_id)
        if not self._should_transform_previous_answer(user_message, recent_turns):
            return None

        last_assistant = next((turn for turn in reversed(recent_turns) if getattr(turn, "role", "") == "assistant"), None)
        if last_assistant is None or not str(getattr(last_assistant, "content", "") or "").strip():
            return None

        prior_answer = str(last_assistant.content or "").strip()
        prior_metadata = last_assistant.metadata if isinstance(last_assistant.metadata, dict) else {}
        prompt = (
            "다음은 직전 답변과 후속 사용자 요청입니다.\n\n"
            f"직전 답변:\n{prior_answer}\n\n"
            f"후속 요청:\n{user_message.strip()}\n\n"
            "후속 요청이 직전 답변을 요약하거나, 재정리하거나, 체크리스트화하거나, 비교 형태로 바꾸려는 의도라면 "
            "직전 답변 안의 정보만 사용해 요청 형식에 맞게 다시 작성해 주세요. "
            "새 사실 추가 금지, 메타 설명 금지, 최종 답변만 출력하세요."
        )
        try:
            transformed = (await deps.llm.generate([{"role": "user", "content": prompt}], max_tokens=420)).strip()
        except Exception:
            transformed = ""
        if not transformed:
            transformed = self._heuristic_transform_previous_answer(user_message, prior_answer) or ""
        if not transformed:
            return None

        expected_shape = self._expected_response_shape(prior_metadata.get("query_interpretation"), user_message)
        if expected_shape and not self._answer_matches_shape(transformed, expected_shape):
            transformed = self._coerce_answer_shape(transformed, expected_shape)

        use_retrieved_context = bool(prior_metadata.get("answer_citations") or prior_metadata.get("source_grounding"))
        transformed = deps.answer_service.sanitize_answer(transformed, use_retrieved_context)
        transformed = deps.answer_service.ensure_answer_source_line(
            transformed,
            prior_metadata.get("answer_citations", []),
            use_retrieved_context,
        )
        if not transformed:
            return None

        if append_user_turn:
            deps.session_repository.add_turn(session_id, "user", user_message)

        final_payload = dict(prior_metadata)
        final_payload["query"] = user_message.strip()
        final_payload["mode"] = final_payload.get("mode", "conversational")
        final_payload["preview_finalized"] = True
        final_payload["transformed_from_previous_answer"] = True

        async def _emit():
            yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
            yield {"type": "token", "content": transformed, "cached": False}
            resolved_topic_id = topic_state_before.get("last_active_topic_id") if isinstance(topic_state_before, dict) else None
            deps.store_assistant_turn(session_id, transformed, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}

        return _emit()

    @staticmethod
    def _split_answer_units(text: str) -> list[str]:
        normalized = re.sub(r"\[source:[^\]]+\]", "", text or "")
        normalized = normalized.replace("\r\n", "\n")
        line_units: list[str] = []
        for raw in normalized.split("\n"):
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^\d+\.\s+", "", line)
            line = re.sub(r"^[-*]\s+", "", line)
            if len(line) >= 6:
                line_units.append(line)
        if len(line_units) > 1:
            return line_units
        units: list[str] = []
        for piece in re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", normalized):
            line = piece.strip()
            if len(line) >= 6:
                units.append(line)
        return units or line_units

    def _heuristic_transform_previous_answer(self, user_message: str, prior_answer: str) -> str | None:
        lowered = (user_message or "").casefold()
        units = self._split_answer_units(prior_answer)
        if not units:
            return None
        step_match = re.search(r"(\d+)\s*단계", lowered)
        if step_match:
            limit = int(step_match.group(1))
            return "\n".join(f"{idx}. {line}" for idx, line in enumerate(units[:limit], start=1)).strip()
        if any(marker in lowered for marker in ("체크리스트", "체크 리스트", "checklist")):
            return "\n".join(f"- {line}" for line in units[:6]).strip()
        if any(marker in lowered for marker in ("요약", "정리", "summary", "목록", "리스트")):
            limit = int(step_match.group(1)) if step_match else 3
            return "\n".join(f"- {line}" for line in units[:limit]).strip()
        if any(marker in lowered for marker in ("단계", "절차", "순서", "step")):
            limit = int(step_match.group(1)) if step_match else min(6, len(units))
            return "\n".join(f"{idx}. {line}" for idx, line in enumerate(units[:limit], start=1)).strip()
        if any(marker in lowered for marker in ("비교", "차이", "compare", "difference")):
            return "\n".join(f"- {line}" for line in units[:6]).strip()
        return None

    @staticmethod
    def _followup_document_reference_markers() -> tuple[str, ...]:
        return (
            "이 문서",
            "그 문서",
            "이 메뉴얼",
            "그 메뉴얼",
            "이 매뉴얼",
            "그 매뉴얼",
            "that document",
            "this document",
            "that manual",
            "this manual",
        )

    def _derive_followup_allowed_sources(self, session_id: str, user_message: str) -> set[str] | None:
        lowered = (user_message or "").casefold()
        if not any(marker in lowered for marker in self._followup_document_reference_markers()):
            return None
        recent_turns = self.deps.session_repository.recent_turns(session_id)
        last_assistant = next((turn for turn in reversed(recent_turns) if getattr(turn, "role", "") == "assistant"), None)
        if last_assistant is None or not isinstance(last_assistant.metadata, dict):
            return None
        metadata = last_assistant.metadata
        sources: list[str] = []
        for citation in metadata.get("answer_citations", []) or []:
            source_path = str(citation.get("source_path") or "")
            if source_path and source_path not in sources:
                sources.append(source_path)
        if not sources:
            for source in metadata.get("source_grounding", []) or []:
                source_path = str(source.get("source_path") or "")
                if source_path and source_path not in sources:
                    sources.append(source_path)
        return set(sources[:2]) if sources else None

    @staticmethod
    def _transform_markers() -> tuple[str, ...]:
        return (
            "요약",
            "정리",
            "체크리스트",
            "체크 리스트",
            "단계",
            "절차",
            "순서",
            "목록",
            "리스트",
            "비교",
            "차이",
            "다시",
            "재작성",
            "step",
            "summary",
            "checklist",
            "compare",
            "difference",
        )

    @staticmethod
    def _answer_reference_markers() -> tuple[str, ...]:
        return (
            "이 답변",
            "그 답변",
            "이 내용",
            "그 내용",
            "이 응답",
            "그 응답",
            "앞 답변",
            "방금 답변",
            "this answer",
            "that answer",
            "this content",
            "that content",
        )

    @staticmethod
    def _document_reference_markers() -> tuple[str, ...]:
        return (
            "이 문서",
            "그 문서",
            "이 메뉴얼",
            "그 메뉴얼",
            "이 매뉴얼",
            "그 매뉴얼",
            "this document",
            "that document",
            "this manual",
            "that manual",
        )

    @staticmethod
    def _new_information_markers() -> tuple[str, ...]:
        return (
            "새로",
            "추가",
            "예시",
            "코드",
            "명령어",
            "방법",
            "설명",
            "이유",
            "관계",
            "yaml",
            "yml",
            "절차",
            "단계",
            "비교",
            "확인",
            "what",
            "why",
            "how",
            "which",
            "example",
            "examples",
            "show me",
        )

    @classmethod
    def _should_transform_previous_answer(cls, user_message: str, recent_turns: list) -> bool:
        if not recent_turns:
            return False
        last_assistant = next((turn for turn in reversed(recent_turns) if getattr(turn, "role", "") == "assistant"), None)
        if last_assistant is None:
            return False
        lowered = (user_message or "").casefold().strip()
        if not lowered:
            return False
        if not any(marker in lowered for marker in cls._transform_markers()):
            return False
        if any(marker in lowered for marker in cls._document_reference_markers()):
            return False
        if any(marker in lowered for marker in cls._answer_reference_markers()):
            return True
        if any(marker in lowered for marker in cls._new_information_markers()):
            return False
        return len(lowered) <= 64

    @staticmethod
    def _expected_response_shape(query_interpretation: dict | None, user_message: str) -> str:
        query_interpretation = query_interpretation or {}
        response_shape = str(query_interpretation.get("response_shape") or "").casefold().strip()
        if response_shape in {"comparison", "procedure", "code", "table"}:
            return response_shape
        lowered = (user_message or "").casefold()
        if "checklist" in lowered or "체크리스트" in lowered or "체크 리스트" in lowered:
            return "checklist"
        if "비교" in lowered or "차이" in lowered or "compare" in lowered or "difference" in lowered:
            return "comparison"
        if "단계" in lowered or "순서" in lowered or "절차" in lowered or "step" in lowered:
            return "procedure"
        return ""

    @staticmethod
    def _answer_matches_shape(answer: str, expected_shape: str) -> bool:
        if not answer or not expected_shape:
            return True
        stripped = answer.strip()
        if expected_shape == "checklist":
            bullets = re.findall(r"(?m)^\s*[-*]\s+\S+", stripped)
            return len(bullets) >= 2
        if expected_shape == "procedure":
            numbered = re.findall(r"(?m)^\s*(?:\d+\.\s+\S+|\d+\s*단계[:\s]+\S+)", stripped)
            return len(numbered) >= 2
        if expected_shape == "comparison":
            bullets = re.findall(r"(?m)^\s*[-*]\s+\S+", stripped)
            labeled_sections = re.findall(r"(?m)^\s*[^:\n]{2,40}:\s+\S+", stripped)
            return len(bullets) >= 2 or len(labeled_sections) >= 2
        return True

    def _coerce_answer_shape(self, answer: str, expected_shape: str) -> str:
        units = self._split_answer_units(answer)
        if not units or not expected_shape:
            return answer
        if expected_shape == "checklist":
            return "\n".join(f"- {line}" for line in units[:6]).strip()
        if expected_shape == "procedure":
            return "\n".join(f"{idx}. {line}" for idx, line in enumerate(units[:6], start=1)).strip()
        if expected_shape == "comparison":
            return "\n".join(f"- {line}" for line in units[:6]).strip()
        return answer

    def _fallback_low_signal_answer(
        self,
        answer: str,
        *,
        use_retrieved_context: bool,
        answer_route: str,
        selected_context_items: list[dict],
    ) -> str:
        if not use_retrieved_context or not answer:
            return answer
        if not self.deps.answer_service.looks_like_low_signal_retrieved_answer(answer):
            return answer
        if answer_route == "extractive_compare":
            fallback = self.deps.answer_service.build_extractive_compare_answer(selected_context_items)
            return fallback or answer
        if answer_route in {"extractive_text", "grounded_generation"}:
            fallback = self.deps.answer_service.build_extractive_text_answer(selected_context_items)
            return fallback or answer
        return answer

    def _postprocess_final_answer(
        self,
        answer: str,
        *,
        user_message: str,
        query_interpretation: dict | None,
        use_retrieved_context: bool,
        answer_route: str,
        selected_context_items: list[dict],
    ) -> str:
        processed = self._fallback_low_signal_answer(
            answer,
            use_retrieved_context=use_retrieved_context,
            answer_route=answer_route,
            selected_context_items=selected_context_items,
        )
        expected_shape = self._expected_response_shape(query_interpretation, user_message)
        if expected_shape and not self._answer_matches_shape(processed, expected_shape):
            processed = self._coerce_answer_shape(processed, expected_shape)
        return processed

    def _build_supporting_examples(
        self,
        *,
        answer: str,
        deps,
        user_message: str,
        query_interpretation: dict | None,
        ordered_context_items: list[dict],
        selected_context_items: list[dict],
    ) -> list[dict]:
        examples: list[dict] = []
        if not answer:
            return examples
        query_interpretation = query_interpretation or {}
        if bool(query_interpretation.get("generic_command_query")):
            return examples
        expected_shape = self._expected_response_shape(query_interpretation, user_message)
        if expected_shape in {"table", "code"}:
            return examples
        if "```" not in answer:
            code_context_items = deps.select_code_example_context_items(
                user_message,
                query_interpretation,
                ordered_context_items,
                selected_context_items,
            )
            if code_context_items:
                code_example = deps.answer_service.build_supporting_code_example(
                    code_context_items,
                    requested_resource_kinds=deps.resolve_requested_resource_kinds(query_interpretation),
                    user_message=user_message,
                    query_interpretation=query_interpretation,
                )
                if code_example:
                    examples.append(code_example)
        table_context_items = deps.prefer_block_type_items(
            ordered_context_items or selected_context_items,
            block_type="table",
            limit=2,
        ) or []
        if "|" not in answer and table_context_items:
            table_example = deps.answer_service.build_supporting_table_example(table_context_items)
            if table_example:
                examples.append(table_example)
        return examples

    async def run(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        uploaded_source_paths: set[str] | None = None,
        append_user_turn: bool = True,
        version_tag: str | None = None,
        available_versions: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        deps = self.deps

        # 버전 선택 응답 감지: 이전 답변이 버전 확인 질문이고 "4.15" 같은 버전 번호가 입력된 경우
        original_query, selected_version = self._detect_version_selection(session_id, user_message, version_tag)
        if original_query and selected_version:
            if append_user_turn:
                deps.session_repository.add_turn(session_id, "user", user_message)
            async for event in self.run(
                session_id=session_id,
                user_message=original_query,
                allowed_source_paths=allowed_source_paths,
                append_user_turn=False,
                version_tag=selected_version,
                available_versions=available_versions,
            ):
                yield event
            return

        lang_notice = deps.detect_non_korean_query(user_message)
        if lang_notice:
            yield {"type": "token", "content": lang_notice, "cached": False}
            yield {"type": "done", "cached": False}
            return

        followup_allowed_sources = self._derive_followup_allowed_sources(session_id, user_message)
        if followup_allowed_sources:
            if allowed_source_paths:
                allowed_source_paths = set(allowed_source_paths) | followup_allowed_sources
            else:
                allowed_source_paths = followup_allowed_sources

        topic_state_before = deps.session_repository.topic_state(session_id)
        current_topic_id_before = str(topic_state_before.get("last_active_topic_id") or "") if isinstance(topic_state_before, dict) else ""
        session_topics_before = deps.session_repository.list_topics(session_id)
        transformed_answer_events = await self._try_transform_previous_answer(
            session_id,
            user_message,
            append_user_turn,
            topic_state_before if isinstance(topic_state_before, dict) else {},
        )
        if transformed_answer_events is not None:
            async for event in transformed_answer_events:
                yield event
            return
        procedure_followup = None
        if not deps.should_skip_procedure_shortcut(user_message, session_topics_before, current_topic_id_before or None):
            procedure_followup = deps.detect_procedure_followup(
                user_message,
                topic_state_before.get("procedure_state", {}) if isinstance(topic_state_before, dict) else {},
            )
        if procedure_followup:
            built = deps.build_procedure_followup_answer(procedure_followup, topic_state_before.get("procedure_state", {}))
            if built is not None:
                procedure_answer, updated_procedure_state = built
                if append_user_turn:
                    deps.session_repository.add_turn(session_id, "user", user_message)
                final_payload = deps.answer_service.build_context_payload(
                    user_message.strip(), "conversational", 0.0, None, [], [], [], [],
                    preview_finalized=True,
                )
                final_payload["procedure_state"] = updated_procedure_state
                yield {"type": "context", **final_payload}
                yield {"type": "token", "content": procedure_answer, "cached": False}
                resolved_topic_id = topic_state_before.get("last_active_topic_id") if isinstance(topic_state_before, dict) else None
                deps.store_assistant_turn(session_id, procedure_answer, final_payload, resolved_topic_id)
                yield {"type": "done", "cached": False}
                return

        if deps.looks_like_step_navigation_without_state(
            user_message,
            topic_state_before.get("procedure_state", {}) if isinstance(topic_state_before, dict) else {},
        ):
            guidance_answer = (
                "어떤 작업의 몇 단계인지 조금 더 구체적으로 알려 주세요. "
                "예를 들어 `ConfigMap 생성 2단계`, `RBAC 설정 2단계`처럼 다시 적어 주시면 바로 이어서 설명하겠습니다."
            )
            if append_user_turn:
                deps.session_repository.add_turn(session_id, "user", user_message)
            final_payload = deps.answer_service.build_context_payload(
                user_message.strip(), "general", 0.0, None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **final_payload}
            yield {"type": "token", "content": guidance_answer, "cached": False}
            resolved_topic_id = topic_state_before.get("last_active_topic_id") if isinstance(topic_state_before, dict) else None
            deps.store_assistant_turn(session_id, guidance_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}
            return

        yield {"type": "status", "stage": "analyzing_intent", "message": STAGE_MESSAGES["analyzing_intent"]}
        turn_context = await deps.resolve_turn_context(session_id, user_message)
        state = deps.domain_guard_state(user_message, turn_context)
        effective_version_tag = self._resolve_effective_version_tag(
            version_tag,
            user_message,
            topic_state_before,
            turn_context.get("session_topic_state"),
            turn_context.get("topic_state"),
        )
        yield {"type": "status", "stage": "searching_documents", "message": STAGE_MESSAGES["searching_documents"]}
        try:
            if state is None:
                state = await deps.prepare_retrieval_state(
                    session_id,
                    user_message,
                    allowed_source_paths,
                    uploaded_source_paths=uploaded_source_paths,
                    version_tag=effective_version_tag,
                    turn_context=turn_context,
                )
        except EmbeddingModelUnavailableError:
            error_message = "임베딩 모델이 아직 준비되지 않았습니다. 잠시 후 다시 시도해 주세요."
            yield {"type": "token", "content": error_message, "cached": False, "error": "embedding_model_unavailable"}
            yield {"type": "done", "cached": False}
            return

        target_versions_from_agent = state.get("query_interpretation", {}).get("target_versions", [])
        use_retrieved_context_early = state.get("use_retrieved_context", False)
        rewritten_query = state["rewritten_query"]
        top_score = state["top_score"]
        use_retrieved_context = state["use_retrieved_context"]
        grounded_pages = state["grounded_pages"]
        ordered_context_items = state.get("ordered_context_items", [])
        selected_context_items = state["selected_context_items"]
        preferred_preview_source = state["preferred_preview_source"]
        preview_pages = state["preview_pages"]
        response_mode = state.get("response_mode", "rag" if use_retrieved_context else "general")
        turn_policy = state.get("turn_policy", {})
        query_interpretation = state.get("query_interpretation", {})
        resolved_topic_id = state.get("resolved_topic_id")
        doc_type = state.get("doc_type", "")
        no_doc_type_docs = state.get("no_doc_type_docs", False)

        # 요청한 doc_type에 해당하는 인덱싱 문서가 없으면 안내 메시지로 조기 종료
        if no_doc_type_docs:
            doc_label = "고객사 운영 메뉴얼" if doc_type == "operation_manual" else f"'{doc_type}' 문서"
            msg = (
                f"현재 {doc_label}가 자료실에 인덱싱되어 있지 않아 해당 문서 기반으로 답변드리기 어렵습니다. "
                f"자료실에 메뉴얼 문서를 업로드하고 인덱싱을 완료한 뒤 다시 질문해 주세요."
            )
            if append_user_turn:
                deps.session_repository.add_turn(session_id, "user", user_message)
            deps.session_repository.add_turn(session_id, "assistant", msg)
            context_payload = deps.answer_service.build_context_payload(
                user_message, "general", 0.0, None, [], [], [], [], preview_finalized=True,
            )
            yield {"type": "context", **deps.answer_service.public_context_payload(context_payload)}
            yield {"type": "token", "content": msg, "cached": False}
            yield {"type": "done"}
            return

        policy_decision = (
            TurnPolicyDecision(**turn_policy)
            if turn_policy
            else TurnPolicyDecision(
                turn_type=response_mode,
                response_mode=response_mode,
                use_retrieval=use_retrieved_context,
                use_memory_rewrite=False,
                allow_preview=use_retrieved_context,
                allow_citations=use_retrieved_context,
            )
        )
        answer_route = deps.resolve_answer_route(query_interpretation)
        code_example_request = answer_route == "extractive_code"
        interleaved_context_items = deps.interleave_context_items_by_source(selected_context_items)
        context_blocks, context_ids = deps.build_context_blocks(interleaved_context_items)

        user_turn_id: int | None = None
        if append_user_turn:
            user_turn_id = deps.session_repository.add_turn(session_id, "user", user_message)
            resolved_topic_id = deps.ensure_topic_for_resolution(session_id, user_message, state, user_turn_id)

        cache_key = deps.build_answer_cache_key(
            session_id=session_id,
            rewritten_query=deps.canonical_cache_query(user_message, rewritten_query, query_interpretation),
            context_ids=context_ids,
            answer_route=answer_route,
            query_interpretation=query_interpretation,
            topic_id=resolved_topic_id,
        )
        cached_answer = deps.answer_cache_repository.get(cache_key)
        if not policy_decision.allow_preview:
            preview_pages = []
            preferred_preview_source = None

        context_payload = deps.answer_service.build_context_payload(
            rewritten_query,
            response_mode,
            top_score,
            preferred_preview_source,
            preview_pages,
            selected_context_items,
            grounded_pages,
            [],
            preview_finalized=False,
        )
        yield {"type": "context", **deps.answer_service.public_context_payload(context_payload)}

        if cached_answer is not None:
            full_text = deps.answer_service.sanitize_answer(cached_answer["answer"], use_retrieved_context)
            streamed = ""
            for token in full_text.split(" "):
                chunk = token + " "
                streamed += chunk
                yield {"type": "token", "content": chunk, "cached": True}
            raw_cached = streamed.strip()
            logger.info(
                "[AnswerStream][cached] raw_answer_chars=%d preview=%r",
                len(raw_cached),
                raw_cached[:200],
            )
            raw_cached = self._postprocess_final_answer(
                raw_cached,
                user_message=user_message,
                query_interpretation=query_interpretation,
                use_retrieved_context=use_retrieved_context,
                answer_route=answer_route,
                selected_context_items=selected_context_items,
            )
            supporting_examples = self._build_supporting_examples(
                answer=raw_cached,
                deps=deps,
                user_message=user_message,
                query_interpretation=query_interpretation,
                ordered_context_items=ordered_context_items,
                selected_context_items=selected_context_items,
            )
            final_answer, _answer_citations, final_payload = deps.finalize_answer(
                answer=raw_cached,
                rewritten_query=rewritten_query,
                use_retrieved_context=use_retrieved_context,
                top_score=top_score,
                selected_context_items=selected_context_items,
                grounded_pages=grounded_pages,
                preferred_preview_source=preferred_preview_source,
                response_mode=response_mode,
                policy_decision=policy_decision,
                query_interpretation=query_interpretation,
                answer_route=answer_route,
                    doc_type=doc_type,
                )
            if supporting_examples:
                final_payload["supporting_examples"] = supporting_examples
            logger.info(
                "[AnswerStream][cached] final_answer_chars=%d preview=%r selected_context=%s",
                len(final_answer),
                final_answer[:200],
                [
                    {
                        "file": Path(str(item["chunk"].get("source_path") or "")).name,
                        "page": item["chunk"].get("page_number"),
                        "score": round(float(item.get("final_retrieval_score", item.get("rerank_score", 0.0))), 4),
                    }
                    for item in selected_context_items[:3]
                ],
            )
            if final_answer != raw_cached:
                logger.info(
                    "[AnswerStream][cached] replace_answer triggered raw_preview=%r final_preview=%r",
                    raw_cached[:200],
                    final_answer[:200],
                )
                yield {"type": "replace_answer", "content": final_answer}
            yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
            deps.store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": True}
            return

        terminal = await self._handle_terminal_policy_answers(
            session_id=session_id,
            rewritten_query=rewritten_query,
            top_score=top_score,
            policy_decision=policy_decision,
            resolved_topic_id=resolved_topic_id,
            cache_key=cache_key,
            use_retrieved_context=use_retrieved_context,
        )
        if terminal is not None:
            async for event in terminal:
                yield event
            return

        yield {"type": "status", "stage": "evaluating_relevance", "message": STAGE_MESSAGES["evaluating_relevance"]}

        if policy_decision.needs_clarification and policy_decision.clarification_prompt:
            clarification_answer = policy_decision.clarification_prompt.strip()
            for char in clarification_answer:
                yield {"type": "token", "content": char, "cached": False}
                await asyncio.sleep(0.03)
            final_payload = deps.answer_service.build_context_payload(
                rewritten_query, response_mode, top_score, None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
            deps.store_assistant_turn(session_id, clarification_answer, final_payload, resolved_topic_id)
            deps.answer_cache_repository.set(cache_key, {"answer": clarification_answer})
            yield {"type": "done", "cached": False}
            return

        extractive_events = await self._handle_extractive_routes(
            session_id=session_id,
            user_message=user_message,
            rewritten_query=rewritten_query,
            top_score=top_score,
            use_retrieved_context=use_retrieved_context,
            grounded_pages=grounded_pages,
            ordered_context_items=ordered_context_items,
            selected_context_items=selected_context_items,
            preferred_preview_source=preferred_preview_source,
            response_mode=response_mode,
            policy_decision=policy_decision,
            query_interpretation=query_interpretation,
            answer_route=answer_route,
            resolved_topic_id=resolved_topic_id,
            cache_key=cache_key,
            doc_type=doc_type,
        )
        if extractive_events is not None:
            async for event in extractive_events:
                yield event
            return

        is_new_topic = not use_retrieved_context and response_mode != "rag"
        prompt_composer = deps.get_prompt_composer()
        use_compact_primary = (
            use_retrieved_context
            and answer_route not in {"extractive_code", "extractive_table"}
            and str(query_interpretation.get("document_group_preference") or "") != "mixed"
            and str(query_interpretation.get("intent") or "") in {"explain", "procedure_followup", ""}
            and top_score >= 0.30
            and len(context_blocks) <= 3
        )
        if use_compact_primary:
            messages = prompt_composer.build_compact_llm_messages(
                user_message,
                code_example_request,
                context_blocks,
                query_interpretation=query_interpretation,
            )
        else:
            messages = prompt_composer.build_llm_messages(
                session_id,
                user_message,
                code_example_request,
                response_mode,
                turn_policy,
                top_score,
                context_blocks,
                is_new_topic=is_new_topic,
                topic_id=resolved_topic_id,
                query_interpretation=query_interpretation,
            )
        yield {"type": "status", "stage": "generating_answer", "message": STAGE_MESSAGES["generating_answer"]}
        parts: list[str] = []
        prompt_context_text = prompt_composer.build_prompt_context_text(context_blocks)
        stream_error: str | None = None
        try:
            async for token in deps.llm.stream_chat(messages):
                parts.append(token)
                yield {"type": "token", "content": token, "cached": False}
        except Exception as exc:
            stream_error = str(exc)

        raw_answer = "".join(parts).strip()
        if not raw_answer:
            try:
                generate_messages = messages
                if not use_compact_primary:
                    generate_messages = prompt_composer.build_compact_llm_messages(
                        user_message,
                        code_example_request,
                        context_blocks,
                        query_interpretation=query_interpretation,
                    )
                generated_answer = await deps.llm.generate(generate_messages)
                raw_answer = generated_answer.strip()
            except Exception as exc:
                stream_error = stream_error or str(exc)
        if not raw_answer:
            if not use_compact_primary:
                try:
                    fallback_messages = prompt_composer.build_llm_messages(
                        session_id,
                        user_message,
                        code_example_request,
                        response_mode,
                        turn_policy,
                        top_score,
                        context_blocks,
                        is_new_topic=is_new_topic,
                        topic_id=resolved_topic_id,
                        query_interpretation=query_interpretation,
                    )
                    generated_answer = await deps.llm.generate(fallback_messages)
                    raw_answer = generated_answer.strip()
                except Exception as exc:
                    stream_error = stream_error or str(exc)

        if not raw_answer:
            fallback = deps.build_llm_failure_fallback(
                user_message,
                use_retrieved_context,
                context_blocks,
                prompt_context_text,
                policy_decision,
            )
            raw_answer = fallback.strip()
            yield {"type": "token", "content": raw_answer, "cached": False, "error": stream_error or "empty_llm_stream"}
        logger.info(
            "[AnswerStream] raw_answer_chars=%d preview=%r",
            len(raw_answer),
            raw_answer[:200],
        )
        raw_answer = self._postprocess_final_answer(
            raw_answer,
            user_message=user_message,
            query_interpretation=query_interpretation,
            use_retrieved_context=use_retrieved_context,
            answer_route=answer_route,
            selected_context_items=selected_context_items,
        )
        supporting_examples = self._build_supporting_examples(
            answer=raw_answer,
            deps=deps,
            user_message=user_message,
            query_interpretation=query_interpretation,
            ordered_context_items=ordered_context_items,
            selected_context_items=selected_context_items,
        )
        final_answer, _answer_citations, final_payload = deps.finalize_answer(
            answer=raw_answer,
            rewritten_query=rewritten_query,
            use_retrieved_context=use_retrieved_context,
            top_score=top_score,
            selected_context_items=selected_context_items,
            grounded_pages=grounded_pages,
            preferred_preview_source=preferred_preview_source,
            response_mode=response_mode,
            policy_decision=policy_decision,
            query_interpretation=query_interpretation,
            answer_route=answer_route,
            doc_type=doc_type,
        )
        if supporting_examples:
            final_payload["supporting_examples"] = supporting_examples
        logger.info(
            "[AnswerStream] final_answer_chars=%d preview=%r selected_context=%s",
            len(final_answer),
            final_answer[:200],
            [
                {
                    "file": Path(str(item["chunk"].get("source_path") or "")).name,
                    "page": item["chunk"].get("page_number"),
                    "score": round(float(item.get("final_retrieval_score", item.get("rerank_score", 0.0))), 4),
                }
                for item in selected_context_items[:3]
            ],
        )
        final_answer = self._postprocess_final_answer(
            final_answer,
            user_message=user_message,
            query_interpretation=query_interpretation,
            use_retrieved_context=use_retrieved_context,
            answer_route=answer_route,
            selected_context_items=selected_context_items,
        )
        if final_answer != raw_answer:
            logger.info(
                "[AnswerStream] replace_answer triggered raw_preview=%r final_preview=%r",
                raw_answer[:200],
                final_answer[:200],
            )
            yield {"type": "replace_answer", "content": final_answer}
        yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
        deps.store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
        deps.answer_cache_repository.set(cache_key, {"answer": final_answer})
        yield {"type": "done", "cached": False}

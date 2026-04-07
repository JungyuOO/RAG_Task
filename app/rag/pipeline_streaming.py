"""Chat-turn orchestration for the streaming RAG flow."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.rag.bge_embeddings import EmbeddingModelUnavailableError
from app.rag.types import TurnPolicyDecision

_VERSION_PATTERN = re.compile(r"^\s*(\d+\.\d+)\s*$")

STAGE_MESSAGES = {
    "analyzing_intent": "질문 의도 분석중...",
    "searching_documents": "자료에서 관련 내용 검색중...",
    "evaluating_relevance": "검색 결과 적합성 평가중...",
    "generating_answer": "답변 생성중...",
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
    should_run_judge_agent: Any
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
    judge_agent: Any
    llm: Any


class StreamingTurnSupport:
    async def _stream_version_clarification(self, session_id, user_message, deps, available_versions):
        versions_str = " / ".join(available_versions) if available_versions else "4.15 / 4.16 / 4.17 / 4.18 / 4.19 / 4.20 / 4.21"
        clarification_msg = (
            f"어떤 버전의 OpenShift Container Platform을 기준으로 답변할까요?\n\n"
            f"지원 버전: **{versions_str}**\n\n"
            f"버전을 명시하거나 화면 상단의 버전 선택 버튼을 사용해 주세요."
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
            return _emit(deps.build_policy_answer("document_query", top_score), "general", True)
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

    def _detect_version_selection(self, session_id: str, user_message: str, version_tag: str | None) -> tuple[str | None, str | None]:
        """이전 턴이 버전 clarification이고 현재 메시지가 버전 번호이면 (원래 질문, 버전) 반환."""
        if version_tag is not None:
            return None, None
        match = _VERSION_PATTERN.match(user_message.strip())
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

    async def run(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        append_user_turn: bool = True,
        version_tag: str | None = None,
        available_versions: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        deps = self.deps

        # 버전 선택 응답 감지: 이전 턴이 버전 clarification이고 "4.15" 등의 버전 번호가 입력된 경우
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

        topic_state_before = deps.session_repository.topic_state(session_id)
        current_topic_id_before = str(topic_state_before.get("last_active_topic_id") or "") if isinstance(topic_state_before, dict) else ""
        session_topics_before = deps.session_repository.list_topics(session_id)
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
        yield {"type": "status", "stage": "searching_documents", "message": STAGE_MESSAGES["searching_documents"]}
        try:
            if state is None:
                state = await deps.prepare_retrieval_state(session_id, user_message, allowed_source_paths, version_tag=version_tag, turn_context=turn_context)
        except EmbeddingModelUnavailableError:
            error_message = "임베딩 모델이 아직 준비되지 않았습니다. 잠시 후 다시 시도해 주세요."
            yield {"type": "token", "content": error_message, "cached": False, "error": "embedding_model_unavailable"}
            yield {"type": "done", "cached": False}
            return

        target_versions_from_agent = state.get("query_interpretation", {}).get("target_versions", [])
        use_retrieved_context_early = state.get("use_retrieved_context", False)
        if (
            version_tag is None
            and not target_versions_from_agent
            and use_retrieved_context_early
        ):
            if append_user_turn:
                deps.session_repository.add_turn(session_id, "user", user_message)
            async for event in self._stream_version_clarification(
                session_id, user_message, deps, available_versions or []
            ):
                yield event
            return

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
            doc_label = "자사 운영 메뉴얼" if doc_type == "operation_manual" else f"'{doc_type}' 문서"
            msg = (
                f"현재 {doc_label}이 자료실에 인덱싱되어 있지 않아 해당 문서 기반으로 답변드리기 어렵습니다. "
                f"자료실에 메뉴얼 문서를 업로드하고 인덱싱을 완료한 후 다시 질문해 주세요."
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
            if final_answer != raw_cached:
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
        if use_retrieved_context and context_blocks and deps.should_run_judge_agent(policy_decision, query_interpretation, top_score):
            judge_result = await deps.judge_agent.evaluate(user_message, context_blocks, top_score)
            if not judge_result["relevant"]:
                clarification = judge_result["clarification_message"]
                for char in clarification:
                    yield {"type": "token", "content": char, "cached": False}
                    await asyncio.sleep(0.03)
                final_payload = deps.answer_service.build_context_payload(
                    rewritten_query, "clarification", top_score, None, [], [], [], [],
                    preview_finalized=True,
                )
                yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
                deps.store_assistant_turn(session_id, clarification, final_payload, resolved_topic_id)
                deps.answer_cache_repository.set(cache_key, {"answer": clarification})
                yield {"type": "done", "cached": False}
                return

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
        try:
            async for token in deps.llm.stream_chat(messages):
                parts.append(token)
                yield {"type": "token", "content": token, "cached": False}
        except Exception as exc:
            fallback = deps.build_llm_failure_fallback(
                user_message,
                use_retrieved_context,
                context_blocks,
                prompt_composer.build_prompt_context_text(context_blocks),
                policy_decision,
            )
            parts = [fallback]
            yield {"type": "token", "content": fallback, "cached": False, "error": str(exc)}

        raw_answer = "".join(parts).strip()
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
        if final_answer != raw_answer:
            yield {"type": "replace_answer", "content": final_answer}
        yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
        deps.store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
        deps.answer_cache_repository.set(cache_key, {"answer": final_answer})
        yield {"type": "done", "cached": False}

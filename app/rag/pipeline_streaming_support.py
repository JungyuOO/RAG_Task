from __future__ import annotations

import asyncio

from app.rag.types import TurnPolicyDecision


class StreamingTurnSupport:
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
                rewritten_query,
                mode,
                top_score,
                None,
                [],
                [],
                [],
                [],
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
        resolved_topic_id = kwargs["resolved_topic_id"]
        cache_key = kwargs["cache_key"]

        async def _emit(final_answer: str, final_payload: dict):
            yield {"type": "token", "content": final_answer, "cached": False}
            yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
            deps.store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
            deps.answer_cache_repository.set(cache_key, {"answer": final_answer})
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
                    extractive_code_answer,
                    rewritten_query,
                    use_retrieved_context,
                    top_score,
                    code_context_items,
                    grounded_pages,
                    preferred_preview_source,
                    response_mode,
                    policy_decision,
                    query_interpretation,
                    answer_route,
                )
                final_payload["last_example_anchor"] = deps.answer_service.build_example_anchor(
                    code_context_items,
                    query_interpretation,
                )
                return _emit(final_answer, final_payload)

            async def _no_code():
                no_code_answer = deps.build_missing_extractive_answer(answer_route)
                yield {"type": "token", "content": no_code_answer, "cached": False}
                final_payload = deps.answer_service.build_context_payload(
                    rewritten_query,
                    "clarification",
                    top_score,
                    None,
                    [],
                    [],
                    [],
                    [],
                    preview_finalized=True,
                )
                yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
                deps.store_assistant_turn(session_id, no_code_answer, final_payload, resolved_topic_id)
                yield {"type": "done", "cached": False}

            return _no_code()

        if answer_route == "extractive_table" and use_retrieved_context:
            table_context_items = deps.prefer_block_type_items(
                ordered_context_items or selected_context_items,
                block_type="table",
                limit=max(len(selected_context_items), 3),
            ) or selected_context_items
            extractive_table_answer = deps.answer_service.build_extractive_table_answer(table_context_items)
            if extractive_table_answer:
                final_answer, _answer_citations, final_payload = deps.finalize_answer(
                    extractive_table_answer,
                    rewritten_query,
                    use_retrieved_context,
                    top_score,
                    table_context_items,
                    grounded_pages,
                    preferred_preview_source,
                    response_mode,
                    policy_decision,
                    query_interpretation,
                    answer_route,
                )
                return _emit(final_answer, final_payload)

            async def _no_table():
                no_table_answer = deps.build_missing_extractive_answer(answer_route)
                yield {"type": "token", "content": no_table_answer, "cached": False}
                final_payload = deps.answer_service.build_context_payload(
                    rewritten_query,
                    "clarification",
                    top_score,
                    None,
                    [],
                    [],
                    [],
                    [],
                    preview_finalized=True,
                )
                yield {"type": "context", **deps.answer_service.public_context_payload(final_payload)}
                deps.store_assistant_turn(session_id, no_table_answer, final_payload, resolved_topic_id)
                yield {"type": "done", "cached": False}

            return _no_table()

        return None

"""Build retrieval state for a chat turn before answer generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.policy import TurnPolicyDecision

logger = logging.getLogger("rag.pipeline")


@dataclass(slots=True)
class RetrievalStateDeps:
    resolve_turn_context: Any
    build_non_retrieval_state: Any
    build_rewrite_context_from_topic: Any
    rewrite_query_with_llm: Any
    index_repository: Any
    query_agent: Any
    query_interpreter: Any
    expand_query_with_resource_aliases: Any
    expand_query_with_context: Any
    embedder: Any
    retrieval_service: Any
    retriever: Any
    reranker: Any
    metadata_aware_rerank: Any
    expand_local_context_items: Any
    expand_topic_anchor_context_items: Any
    should_use_retrieved_context: Any
    apply_precision_filter: Any
    apply_focus_filter: Any
    find_fallback_code_context_items: Any
    settings: Any


class RetrievalStateBuilder:
    """Build retrieval state for a chat turn using explicit collaborators."""

    def __init__(self, deps: RetrievalStateDeps) -> None:
        self.deps = deps

    async def run(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
    ) -> dict:
        deps = self.deps
        turn_context = deps.resolve_turn_context(session_id, user_message)
        resolution = turn_context["resolution"]
        resolved_topic = turn_context["resolved_topic"]
        topic_state = turn_context["topic_state"]
        scoped_recent_turns = turn_context["scoped_recent_turns"]
        policy: TurnPolicyDecision = turn_context["policy"]
        if not policy.use_retrieval:
            return deps.build_non_retrieval_state(user_message, turn_context)

        rewrite_context = (
            deps.build_rewrite_context_from_topic(resolved_topic, scoped_recent_turns)
            if resolved_topic is not None
            else None
        )
        rewritten_query = (
            await deps.rewrite_query_with_llm(session_id, user_message, rewrite_context=rewrite_context)
            if policy.use_memory_rewrite
            else user_message.strip()
        )

        index_items_all = deps.index_repository.load()
        all_sources: list[str] = []
        seen_sources: set[str] = set()
        for item in index_items_all:
            src = item["chunk"]["source_path"]
            if src not in seen_sources:
                seen_sources.add(src)
                all_sources.append(src)

        query_result = await deps.query_agent.refine_query(
            rewritten_query,
            context={
                "active_topic": topic_state.get("active_topic"),
                "selected_sources": topic_state.get("selected_sources", []),
            },
            available_sources=all_sources,
        )
        refined_query = query_result["refined_query"]
        alternative_queries = query_result.get("alternative_queries", [])
        logger.info(
            "[QueryAgent] ?먮낯=%r ??理쒖쟻??%r | ???%r | ?ㅼ썙??%r",
            rewritten_query, refined_query, alternative_queries, query_result.get("search_keywords", []),
        )

        query_interpretation = deps.query_interpreter.interpret(
            user_message,
            query_result=query_result,
            topic_state=topic_state,
        )
        logger.info(
            "[QueryInterpretation] intent=%s resources=%s actions=%s formats=%s shape=%s keywords=%s",
            query_interpretation.intent,
            query_interpretation.resources,
            query_interpretation.actions,
            query_interpretation.format_constraints,
            query_interpretation.response_shape,
            query_interpretation.normalized_keywords,
        )

        aliased_query = deps.expand_query_with_resource_aliases(refined_query, query_interpretation.to_dict())
        expanded_query = deps.expand_query_with_context(aliased_query, topic_state)
        expanded_query = self._expand_short_resource_query(
            expanded_query,
            user_message,
            query_interpretation,
        )
        query_vector = deps.embedder.encode(expanded_query)
        index_items = deps.retrieval_service.filter_index_items(index_items_all, allowed_source_paths)
        retrieved = deps.retriever.search_rrf(expanded_query, query_vector, index_items, rrf_k=60)

        query_interpretation_dict = query_interpretation.to_dict()

        if not query_interpretation.resources and topic_state.get("last_explicit_resources"):
            inherited = topic_state["last_explicit_resources"][:2]
            query_interpretation_dict["resources"] = inherited
            logger.info("[FollowupAnchor] inherited resources=%s", inherited)

        selected_source_names = {
            str(source).casefold().strip()
            for source in topic_state.get("selected_sources", []) or []
            if source
        }
        lowered_shape = str(query_interpretation.response_shape or "").casefold()
        lowered_intent = str(query_interpretation.intent or "").casefold()
        should_run_selected_source_pass = (
            bool(selected_source_names)
            and (
                lowered_shape == "code"
                or lowered_intent in {"explain", "yaml_example", "cli_example", "code_example", "procedure_followup"}
                or bool(topic_state.get("last_explicit_resources"))
            )
        )
        if should_run_selected_source_pass:
            selected_source_items = [
                item
                for item in index_items
                if Path(str(item["chunk"]["source_path"] or "")).name.casefold() in selected_source_names
            ]
            if selected_source_items:
                source_retrieved = deps.retriever.search_rrf(
                    expanded_query,
                    query_vector,
                    selected_source_items,
                    rrf_k=60,
                )
                for item in source_retrieved:
                    cid = item["chunk"]["chunk_id"]
                    if cid not in {existing["chunk"]["chunk_id"] for existing in retrieved}:
                        retrieved.append(item)
                if source_retrieved:
                    logger.info(
                        "[Retrieval] selected-source pass sources=%s added=%d",
                        sorted(selected_source_names),
                        len(
                            [
                                item
                                for item in source_retrieved
                                if item["chunk"]["chunk_id"] in {existing["chunk"]["chunk_id"] for existing in retrieved}
                            ]
                        ),
                    )

        seen_chunk_ids: set[str] = {item["chunk"]["chunk_id"] for item in retrieved}
        merged_extras: list[dict] = []
        for alt_query in alternative_queries[:2]:
            alt_aliased = deps.expand_query_with_resource_aliases(alt_query, query_interpretation_dict)
            alt_expanded = deps.expand_query_with_context(alt_aliased, topic_state)
            alt_vector = deps.embedder.encode(alt_expanded)
            alt_retrieved = deps.retriever.search_rrf(alt_expanded, alt_vector, index_items, rrf_k=60)
            for item in alt_retrieved:
                cid = item["chunk"]["chunk_id"]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    merged_extras.append(item)
        if merged_extras:
            combined = retrieved + merged_extras
            combined.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
            retrieved = combined[: deps.retriever.top_k]

        for index, item in enumerate(retrieved[:5]):
            chunk = item["chunk"]
            logger.info(
                "[Retrieval] #%d %s p.%s | rerank=%.4f dense=%.4f sparse=%.4f",
                index + 1,
                Path(chunk["source_path"]).name,
                chunk.get("page_number", "?"),
                item.get("rerank_score", 0),
                item.get("dense_score", 0),
                item.get("sparse_score", 0),
            )

        if retrieved:
            extended = deps.retriever.search_rrf(expanded_query, query_vector, index_items, rrf_k=60)
            extended = extended[:20]
            retrieved = deps.reranker.rerank(expanded_query, extended)
            retrieved = deps.metadata_aware_rerank(
                user_message,
                query_interpretation_dict,
                retrieved,
            )
            retrieved = deps.expand_local_context_items(
                user_message,
                query_interpretation_dict,
                index_items,
                retrieved,
            )
            retrieved = deps.expand_topic_anchor_context_items(
                user_message,
                query_interpretation_dict,
                index_items,
                retrieved,
                topic_state,
            )

        retrieval_metrics = deps.retriever.compute_retrieval_metrics(
            retrieved,
            min_score=deps.settings.retrieval_min_score,
        )
        top_score = retrieval_metrics["top_score"]
        use_retrieved_context = deps.should_use_retrieved_context(
            policy,
            retrieved,
            top_score,
            query_interpretation_dict,
        )
        logger.info(
            "[Retrieval] top_score=%.4f use_context=%s min_score=%.4f",
            top_score,
            use_retrieved_context,
            deps.settings.retrieval_min_score,
        )
        context_items = retrieved if use_retrieved_context else []
        grounded_pages = deps.retrieval_service.aggregate_page_grounding(context_items)
        ordered_context_items = deps.retrieval_service.order_context_items_by_grounded_pages(context_items, grounded_pages)
        selected_context_items = deps.retrieval_service.select_context_items_by_grounded_pages(
            ordered_context_items,
            grounded_pages,
        )
        selected_context_items = deps.apply_precision_filter(
            selected_context_items,
            query_interpretation_dict,
        )
        selected_context_items = deps.apply_focus_filter(
            selected_context_items,
            query_interpretation_dict,
        )
        if not selected_context_items and str(query_interpretation.response_shape or "").casefold() == "code":
            fallback_code_items = deps.find_fallback_code_context_items(
                user_message,
                query_interpretation_dict,
                index_items,
                topic_state,
            )
            if fallback_code_items:
                selected_context_items = fallback_code_items
                ordered_context_items = fallback_code_items
                grounded_pages = deps.retrieval_service.aggregate_page_grounding(fallback_code_items)
                top_score = max(
                    top_score,
                    max(float(item.get("rerank_score", 0.0)) for item in fallback_code_items),
                )
                use_retrieved_context = True
        preferred_preview_source = deps.retrieval_service.select_grounded_preview_source(grounded_pages)
        preview_pages = deps.retrieval_service.build_grounded_preview_pages(
            preferred_preview_source,
            grounded_pages,
        )
        return {
            "rewritten_query": rewritten_query,
            "top_score": top_score,
            "use_retrieved_context": use_retrieved_context,
            "grounded_pages": grounded_pages,
            "ordered_context_items": ordered_context_items,
            "selected_context_items": selected_context_items,
            "preferred_preview_source": preferred_preview_source,
            "preview_pages": preview_pages,
            "response_mode": "rag" if use_retrieved_context else "general",
            "turn_policy": policy.to_dict(),
            "retrieval_metrics": retrieval_metrics,
            "turn_resolution": resolution.to_dict(),
            "resolved_topic_id": resolution.topic_id,
            "query_interpretation": query_interpretation_dict,
        }

    @staticmethod
    def _expand_short_resource_query(
        expanded_query: str,
        user_message: str,
        query_interpretation: Any,
    ) -> str:
        """Expand very short resource queries to lift cross-encoder confidence."""
        resources = list(query_interpretation.resources or [])
        intent = str(query_interpretation.intent or "").casefold()
        response_shape = str(query_interpretation.response_shape or "").casefold()

        if not resources or len(user_message.strip()) > 30:
            return expanded_query

        if intent not in ("explain", "") and response_shape != "text":
            return expanded_query

        resource_str = " ".join(resources)
        suffix_parts = [f"{resource_str} 媛쒕뀗 ??븷 ?뱀쭠 ?ㅻ챸"]
        if intent == "explain" or response_shape == "text":
            suffix_parts.append("?숈옉 ?먮━ 援ъ꽦 ?붿냼")

        suffix = " ".join(suffix_parts)
        if suffix.strip() and suffix.strip() not in expanded_query:
            expanded_query = f"{expanded_query} {suffix}"
            logger.info("[QueryExpansion] short query expanded: %r", expanded_query)

        return expanded_query

"""Build retrieval state for a chat turn before answer generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.types import TurnPolicyDecision
from app.rag.utils import tokenize

logger = logging.getLogger("rag.pipeline")


@dataclass(slots=True)
class RetrievalStateDeps:
    resolve_turn_context: Any
    build_non_retrieval_state: Any
    build_rewrite_context_from_topic: Any
    rewrite_query_with_llm: Any
    index_repository: Any
    intent_agent: Any
    retrieval_agent: Any
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
        *,
        version_tag: str | None = None,
        turn_context: dict | None = None,
    ) -> dict:
        deps = self.deps
        target_versions = [version_tag] if version_tag else None
        # turn_context가 이미 계산된 경우 재사용 (중복 LLM 호출 방지)
        if turn_context is None:
            turn_context = await deps.resolve_turn_context(session_id, user_message)
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

        # turn_context에서 이미 계산된 intent_result 재사용 (중복 LLM 호출 제거)
        intent_result = turn_context.get("intent_result") or await deps.intent_agent.classify(
            user_message,
            context={
                "active_topic": topic_state.get("active_topic"),
                "selected_sources": topic_state.get("selected_sources", []),
                "procedure_state": topic_state.get("procedure_state", {}),
            },
        )

        # doc_type: intent에서 감지된 값 우선, 없으면 이전 턴에서 유지된 값
        doc_type = intent_result.get("doc_type") or topic_state.get("last_doc_type") or None
        if doc_type:
            logger.info("[DocType] doc_type=%s (from=%s)", doc_type,
                        "intent" if intent_result.get("doc_type") else "topic_state")

        query_result = await deps.retrieval_agent.expand(
            rewritten_query,
            intent_result=intent_result,
            available_sources=all_sources,
        )
        refined_query = query_result["refined_query"]
        alternative_queries = query_result.get("alternative_queries", [])
        logger.info(
            "[RetrievalAgent] 재작성=%r 검색질의=%r | 대안=%r | 번역키워드=%r",
            rewritten_query, refined_query, alternative_queries, query_result.get("translated_keywords", []),
        )

        query_interpretation = deps.retrieval_agent.interpret(
            user_message,
            query_result=query_result,
            topic_state=topic_state,
        )
        logger.info(
            "[QueryInterpretation] intent=%s resources=%s actions=%s formats=%s shape=%s keywords=%s",
            query_interpretation["intent"],
            query_interpretation["resources"],
            query_interpretation["actions"],
            query_interpretation["format_constraints"],
            query_interpretation["response_shape"],
            query_interpretation["normalized_keywords"],
        )

        aliased_query = deps.expand_query_with_resource_aliases(refined_query, query_interpretation)
        expanded_query = deps.expand_query_with_context(aliased_query, topic_state)
        expanded_query = self._expand_short_resource_query(
            expanded_query,
            user_message,
            query_interpretation,
        )
        query_vector = deps.embedder.encode(expanded_query)
        index_items = deps.retrieval_service.filter_index_items(index_items_all, allowed_source_paths, doc_type=doc_type)

        # BM25는 영어 문서에 대해 Lexical Exact Match를 수행하므로,
        # 한국어가 섞인 rewritten_query 대신 RetrievalAgent가 영어로 번역한 refined_query를 사용한다.
        # 이렇게 해야 한국어 토큰이 BM25에서 0점을 받는 문제를 방지할 수 있다.
        bm25_keyword_query = self._build_bm25_keyword_query(refined_query)

        logger.info(
            "[BM25] keyword_query=%r (tokens=%d vs expanded=%d)",
            bm25_keyword_query[:80],
            len(tokenize(bm25_keyword_query)),
            len(tokenize(expanded_query)),
        )

        # 첫 RRF 호출에서 candidate_pool_size개(넓은 풀)를 받아둔다.
        # top_k개는 selected_source_pass / alternative_queries 병합용,
        # 전체 풀은 cross-encoder 입력으로 재사용하여 중복 호출을 제거한다.
        rrf_k = getattr(deps.settings, "rrf_k", 30)
        base_rrf_pool = deps.retriever.search_rrf(
            expanded_query, query_vector, index_items, rrf_k=rrf_k,
            limit=deps.retriever.candidate_pool_size,
            target_versions=target_versions,
            keyword_query=bm25_keyword_query,
        )
        retrieved = base_rrf_pool[: deps.retriever.top_k]

        query_interpretation_dict = dict(query_interpretation)

        if not query_interpretation["resources"] and topic_state.get("last_explicit_resources"):
            inherited = topic_state["last_explicit_resources"][:2]
            query_interpretation_dict["resources"] = inherited
            logger.info("[FollowupAnchor] inherited resources=%s", inherited)

        selected_source_names = {
            str(source).casefold().strip()
            for source in topic_state.get("selected_sources", []) or []
            if source
        }
        lowered_shape = str(query_interpretation["response_shape"] or "").casefold()
        lowered_intent = str(query_interpretation["intent"] or "").casefold()
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
                    rrf_k=rrf_k,
                    target_versions=target_versions,
                    keyword_query=bm25_keyword_query,
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
            alt_bm25_kw = self._build_bm25_keyword_query(alt_query)
            alt_retrieved = deps.retriever.search_rrf(alt_expanded, alt_vector, index_items, rrf_k=rrf_k, target_versions=target_versions, keyword_query=alt_bm25_kw)
            for item in alt_retrieved:
                cid = item["chunk"]["chunk_id"]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    merged_extras.append(item)
        if merged_extras:
            retrieved = retrieved + merged_extras

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
            # 병합된 후보(selected_source_pass + alternative_queries)와
            # 첫 호출에서 캐싱해둔 base_rrf_pool을 합쳐 cross-encoder에 넘긴다.
            seen_ids: set[str] = set()
            extended: list[dict] = []
            for item in retrieved:
                cid = item["chunk"]["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    extended.append(item)
            for item in base_rrf_pool:
                cid = item["chunk"]["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    extended.append(item)
            extended.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
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
        if not selected_context_items and str(query_interpretation["response_shape"] or "").casefold() == "code":
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
            "doc_type": doc_type or "",
        }

    @staticmethod
    def _expand_short_resource_query(
        expanded_query: str,
        user_message: str,
        query_interpretation: Any,
    ) -> str:
        """Explain/what-is 인텐트 쿼리를 개념 검색에 적합하게 확장한다.

        resources가 없더라도 normalized_keywords를 활용하며, 쿼리 길이 제한을
        제거하여 한국어 긴 질문도 확장 대상에 포함한다.
        """
        resources = list(query_interpretation["resources"] or [])
        intent = str(query_interpretation["intent"] or "").casefold()
        response_shape = str(query_interpretation["response_shape"] or "").casefold()
        normalized_keywords = list(query_interpretation.get("normalized_keywords") or [])

        is_explain_intent = intent == "explain" or response_shape == "text"
        if not is_explain_intent:
            return expanded_query

        # 이미 충분히 긴 확장 쿼리는 더 늘리지 않는다
        if len(expanded_query.split()) > 25:
            return expanded_query

        # resources 우선, 없으면 normalized_keywords로 fallback
        user_lower = user_message.casefold()
        candidates = [r for r in resources if r.casefold() in user_lower]
        if not candidates:
            candidates = [kw for kw in normalized_keywords if len(kw) > 2 and kw.isascii()][:2]
        if not candidates:
            return expanded_query

        resource_str = " ".join(candidates[:2])
        suffix = f"{resource_str} definition concept what is overview explanation"
        if suffix.strip() not in expanded_query:
            expanded_query = f"{expanded_query} {suffix}"
            logger.info("[QueryExpansion] explain-intent expanded: %r", expanded_query[:120])
        return expanded_query

    # BM25 filler words to strip — generic verbs, question words, articles, etc.
    _BM25_STOPWORDS: set[str] = {
        # English question/filler
        "what", "is", "are", "how", "does", "do", "the", "a", "an", "in", "of",
        "and", "or", "to", "for", "with", "its", "it", "this", "that", "on",
        "by", "from", "about", "between", "can", "be", "has", "have",
        # Generic action/filler verbs
        "explain", "describe", "show", "tell", "discuss", "detail",
        "provide", "list", "give", "understand", "define",
        # Generic nouns that dilute specificity
        "concept", "concepts", "overview", "benefits", "benefit",
        "management", "resource", "resources", "works", "work",
        "features", "feature", "role", "roles",
        # Korean filler (after tokenize strips suffixes)
        "뭐야", "무엇", "어떻게", "왜", "설명", "개념", "역할", "특징",
        "동작", "원리", "구성", "요소", "방법", "차이", "비교",
    }

    @classmethod
    def _build_bm25_keyword_query(cls, query: str) -> str:
        """Strip filler/stopwords from a query to keep only core terms for BM25.

        Example:
          'What is time slicing in OpenShift Container Platform and Kubernetes?
           Explain the concept, how it works, and its benefits for resource management.'
        → 'time slicing openshift container platform kubernetes'
        """
        tokens = tokenize(query)
        kept = [t for t in tokens if t not in cls._BM25_STOPWORDS]
        if not kept:
            return query
        # De-duplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in kept:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return " ".join(unique)


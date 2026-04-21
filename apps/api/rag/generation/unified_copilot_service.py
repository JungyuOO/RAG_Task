from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from apps.api.api.schemas.chat import (
    CopilotChatHistoryTurn,
    CopilotChatResponse,
    CopilotChatSourceItem,
    CopilotChatStage,
)
from apps.api.api.schemas.copilot_chat import CopilotChatArtifact, CopilotCitationMapItem
from apps.api.api.schemas.ocp_live import OcpLiveResourceSummary
from apps.api.integrations.ocp.auth import OcpConnectionBroker
from apps.api.integrations.ocp.live_chat_service import LiveOcpChatService
from apps.api.rag.generation.answer_planner import AnswerPlanner
from apps.api.rag.generation.citation_grounding import CitationGroundingValidator
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient
from apps.api.rag.generation.response_cache import ChatResponseCache
from apps.api.rag.query.intent_agent import IntentAgent
from apps.api.rag.query.question_normalizer import QuestionNormalizer
from apps.api.rag.query.query_features import answer_source_budget, preferred_source_paths_for_query, tokenize_query
from apps.api.rag.query.query_router import QueryRouter
from apps.api.rag.query.query_rewrite_agent import QueryRewriteAgent
from apps.api.rag.query.synonym_expansion import expand_acronyms
from apps.api.rag.retrieval.document_retriever import DocumentRetriever
from apps.api.rag.retrieval.hybrid_fusion import merge_hybrid_sources
from apps.api.rag.retrieval.pgvector_bridge import PgvectorRetrievalBridge
from apps.api.rag.retrieval.rerank_decider import needs_rerank

logger = logging.getLogger("rag.chat")


class UnifiedCopilotService:
    """Routes chat requests between document retrieval and live OCP queries."""

    def __init__(
        self,
        *,
        live_chat_service: LiveOcpChatService,
        document_retriever: DocumentRetriever,
        pgvector_bridge: PgvectorRetrievalBridge,
        intent_agent: IntentAgent | None = None,
        question_normalizer: QuestionNormalizer | None = None,
        query_router: QueryRouter | None = None,
        query_rewrite_agent: QueryRewriteAgent | None = None,
        citation_validator: CitationGroundingValidator | None = None,
        response_cache: ChatResponseCache | None = None,
        llm_client: OpenAiCompatibleLlmClient | None = None,
        answer_planner: AnswerPlanner | None = None,
    ) -> None:
        self.live_chat_service = live_chat_service
        self.document_retriever = document_retriever
        self.pgvector_bridge = pgvector_bridge
        self.llm_client = llm_client
        self.intent_agent = intent_agent or IntentAgent(llm_client=llm_client)
        self.question_normalizer = question_normalizer
        self.query_router = query_router or QueryRouter(llm_client=llm_client)
        self.query_rewrite_agent = query_rewrite_agent or QueryRewriteAgent(llm_client=llm_client)
        self.citation_validator = citation_validator or CitationGroundingValidator()
        self.response_cache = response_cache or ChatResponseCache()
        self.answer_planner = answer_planner or AnswerPlanner()

    async def answer(
        self,
        *,
        message: str,
        connection_id: str,
        namespace: str,
        recent_turns: list[CopilotChatHistoryTurn] | None = None,
        broker: OcpConnectionBroker,
        progress: Callable[[CopilotChatStage], Awaitable[None] | None] | None = None,
        answer_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CopilotChatResponse:
        overall_started = time.perf_counter()
        recent_turns = recent_turns or []
        await self._emit_progress(
            progress,
            key="analyze_question",
            label="Analyze question",
            detail="Determine whether this request should use documentation retrieval, live cluster access, or both.",
        )

        retrieval_message = message
        decision = None
        if self._query_router_enabled():
            router_started = time.perf_counter()
            decision = await self.query_router.decide(
                user_message=message,
                recent_turns=recent_turns,
                has_connection=bool(connection_id),
                last_lane=self._last_assistant_lane(recent_turns),
                last_doc_sources=self._last_doc_source_paths(recent_turns),
            )
            retrieval_message = decision.search_query or message
            logger.info(
                "[router] lane=%s search_query=%r live_query=%r inherit=%s elapsed=%.2fs",
                decision.lane,
                decision.search_query,
                decision.live_query,
                decision.inherit_sources,
                time.perf_counter() - router_started,
            )
        else:
            if self.question_normalizer is not None:
                normalize_started = time.perf_counter()
                normalized = await self.question_normalizer.normalize(message=message, recent_turns=recent_turns)
                retrieval_message = normalized.text or message
                logger.info(
                    "[normalize] original=%r normalized=%r language=%s elapsed=%.2fs",
                    message,
                    retrieval_message,
                    normalized.language,
                    time.perf_counter() - normalize_started,
                )

            intent_started = time.perf_counter()
            decision = await self.intent_agent.classify(
                message=message,
                has_connection=bool(connection_id),
                recent_turns=recent_turns,
            )
            logger.info(
                "[intent] lane=%s doc_query=%r live_query=%r elapsed=%.2fs",
                decision.lane,
                decision.doc_query,
                decision.live_query,
                time.perf_counter() - intent_started,
            )

        if decision.lane == "needs_connection":
            response = CopilotChatResponse(
                lane="needs_connection",
                mode="guidance",
                fallback_used=False,
                preview_ready=False,
                answer="This request needs a live cluster connection. Configure Connection first, then ask again.",
                sources=[],
            )
            await self._emit_progress(
                progress,
                key="needs_connection",
                label="Connection required",
                detail="The request was classified as live cluster access, but no connection is available.",
            )
            await self._emit_answer_text(response.answer, answer_delta)
            return response

        live_message = (decision.live_query or message) if self._query_router_enabled() else message

        if decision.lane == "live":
            await self._emit_progress(
                progress,
                key="route_live",
                label="Route to live",
                detail="Use the live cluster route for current resource state.",
            )
            response = await self._answer_live_lane(
                message=live_message,
                connection_id=connection_id,
                namespace=namespace,
                broker=broker,
                progress=progress,
                recent_turns=recent_turns,
            )
            await self._emit_progress(
                progress,
                key="finalize_answer",
                label="Finalize answer",
                detail="Return the live cluster result.",
            )
            await self._emit_answer_text(response.answer, answer_delta)
            logger.info("[answer] lane=live total_elapsed=%.2fs", time.perf_counter() - overall_started)
            return response

        if decision.lane == "mixed":
            await self._emit_progress(
                progress,
                key="route_mixed",
                label="Route to mixed",
                detail="Combine document evidence with current cluster state.",
            )
            mixed_doc_query = getattr(decision, "search_query", "") or getattr(decision, "doc_query", "") or retrieval_message
            if mixed_doc_query == message and retrieval_message != message:
                mixed_doc_query = retrieval_message
            doc_response = await self._answer_doc_lane(
                message=mixed_doc_query,
                recent_turns=recent_turns,
                original_message=message,
                allow_llm_synthesis=False,
                allowed_source_paths=self._last_doc_source_paths(recent_turns) if getattr(decision, "inherit_sources", False) else None,
                skip_rewrite=self._query_router_enabled(),
                progress=progress,
            )
            live_response = None
            if connection_id:
                live_response = await self._answer_live_lane(
                    message=live_message,
                    connection_id=connection_id,
                    namespace=namespace,
                    broker=broker,
                    progress=progress,
                    recent_turns=recent_turns,
                )
            response = await self._build_mixed_response(
                message=message,
                doc_response=doc_response,
                live_response=live_response,
                live_available=bool(connection_id),
                answer_delta=answer_delta,
            )
            await self._emit_progress(
                progress,
                key="finalize_answer",
                label="Finalize answer",
                detail="Merge document and live answers into one response.",
            )
            logger.info("[answer] lane=mixed total_elapsed=%.2fs", time.perf_counter() - overall_started)
            return response

        doc_query = getattr(decision, "search_query", "") or getattr(decision, "doc_query", "") or retrieval_message
        if doc_query == message and retrieval_message != message:
            doc_query = retrieval_message
        response = await self._answer_doc_lane(
            message=doc_query,
            recent_turns=recent_turns,
            original_message=message,
            allowed_source_paths=self._last_doc_source_paths(recent_turns) if getattr(decision, "inherit_sources", False) else None,
            skip_rewrite=self._query_router_enabled(),
            progress=progress,
            answer_delta=answer_delta,
        )
        await self._emit_progress(
            progress,
            key="finalize_answer",
            label="Finalize answer",
            detail="Return the document-grounded answer.",
        )
        logger.info(
            "[answer] lane=%s mode=%s total_elapsed=%.2fs",
            response.lane,
            response.mode,
            time.perf_counter() - overall_started,
        )
        return response

    async def _answer_live_lane(
        self,
        *,
        message: str,
        connection_id: str,
        namespace: str,
        broker: OcpConnectionBroker,
        progress: Callable[[CopilotChatStage], Awaitable[None] | None] | None = None,
        recent_turns: list[CopilotChatHistoryTurn] | None = None,
    ) -> CopilotChatResponse:
        started = time.perf_counter()
        live_response = await self.live_chat_service.answer(
            connection_id=connection_id,
            message=message,
            namespace=namespace,
            broker=broker,
            progress=progress,
            recent_turns=recent_turns,
        )
        logger.info("[live] query=%r mode=%s elapsed=%.2fs", message, live_response.mode, time.perf_counter() - started)
        return CopilotChatResponse(
            lane="live",
            mode=live_response.mode,
            fallback_used=False,
            preview_ready=False,
            answer=live_response.answer,
            sources=live_response.sources or [self._map_live_item(item) for item in live_response.items[:8]],
            artifacts=live_response.artifacts,
            citation_map=self._build_citation_map(live_response.sources or [self._map_live_item(item) for item in live_response.items[:8]]),
        )

    async def _answer_doc_lane(
        self,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        original_message: str = "",
        allow_llm_synthesis: bool = True,
        allowed_source_paths: list[str] | None = None,
        skip_rewrite: bool = False,
        progress: Callable[[CopilotChatStage], Awaitable[None] | None] | None = None,
        answer_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CopilotChatResponse:
        if skip_rewrite:
            rewritten_query = message.strip()
            resolved_allowed_source_paths = list(allowed_source_paths or [])
        else:
            rewrite_started = time.perf_counter()
            rewrite = await self.query_rewrite_agent.rewrite(message=message, recent_turns=recent_turns)
            rewritten_query = rewrite.rewritten_query
            resolved_allowed_source_paths = list(allowed_source_paths or rewrite.allowed_source_paths)
            logger.info(
                "[rewrite] original=%r rewritten=%r allowed_sources=%d elapsed=%.2fs",
                message,
                rewritten_query,
                len(resolved_allowed_source_paths),
                time.perf_counter() - rewrite_started,
            )
        retrieval_query = expand_acronyms(rewritten_query) if self._synonym_expansion_enabled() else rewritten_query
        source_hints = preferred_source_paths_for_query(retrieval_query)
        sparse_allowed_source_paths = list(resolved_allowed_source_paths)
        dense_allowed_source_paths = list(resolved_allowed_source_paths)
        if source_hints:
            hinted = [*sparse_allowed_source_paths, *source_hints]
            deduped: list[str] = []
            seen_paths: set[str] = set()
            for path in hinted:
                normalized = str(path or "").strip()
                if not normalized or normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
                deduped.append(normalized)
            sparse_allowed_source_paths = deduped
            dense_allowed_source_paths = deduped

        cache_key = self.response_cache.build_key(
            {
                "lane": "doc",
                "query": retrieval_query,
                "allowed_sources": sorted(sparse_allowed_source_paths),
                "allow_llm_synthesis": allow_llm_synthesis,
            }
        )
        cached = None
        if self._response_cache_enabled():
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                logger.info("[cache] hit lane=doc key=%s", cache_key[:12])
                await self._emit_answer_text(cached.answer, answer_delta)
                return cached
            logger.info("[cache] miss lane=doc key=%s", cache_key[:12])
        else:
            logger.info("[cache] disabled lane=doc key=%s", cache_key[:12])

        await self._emit_progress(
            progress,
            key="retrieve_keyword_docs",
            label="Retrieve lexical docs",
            detail="Search the lexical document index for relevant evidence.",
        )
        lexical_started = time.perf_counter()
        dense_allowed = dense_allowed_source_paths or None
        dense_additional = original_message if original_message and original_message.strip() != retrieval_query.strip() else None
        new_doc_response, pgvector_response = await self._retrieve_doc_candidates(
            message=retrieval_query,
            sparse_allowed_source_paths=sparse_allowed_source_paths or None,
            dense_allowed_source_paths=dense_allowed,
            additional_query=dense_additional,
        )
        lexical_paths = [
            source.source_path
            for source in (new_doc_response.sources if new_doc_response else [])
            if source.source_path
        ][: answer_source_budget(retrieval_query) + 1]
        logger.info(
            "[retrieve:sparse] query=%r sources=%d candidate_paths=%d elapsed=%.2fs",
            retrieval_query,
            len(new_doc_response.sources) if new_doc_response else 0,
            len(lexical_paths),
            time.perf_counter() - lexical_started,
        )

        await self._emit_progress(
            progress,
            key="retrieve_vector_docs",
            label="Retrieve dense docs",
            detail="Search the dense vector index for semantically similar evidence.",
        )
        dense_started = time.perf_counter()
        logger.info(
            "[retrieve:dense] query=%r source_filter=%d sources=%d elapsed=%.2fs",
            retrieval_query,
            len(dense_allowed or []),
            len(pgvector_response.sources) if pgvector_response else 0,
            time.perf_counter() - dense_started,
        )

        await self._emit_progress(
            progress,
            key="synthesize_sources",
            label="Merge sources",
            detail="Merge lexical and dense retrieval results into the final evidence set.",
        )
        merge_started = time.perf_counter()
        if self._should_prefer_hinted_sparse(source_hints, new_doc_response, pgvector_response):
            hybrid_response = None
            response = new_doc_response  # type: ignore[assignment]
        elif self._rrf_fusion_enabled():
            hybrid_response = self._merge_doc_responses(retrieval_query, pgvector_response, new_doc_response)
            if hybrid_response is not None:
                response = hybrid_response
            elif self._should_prefer_pgvector(retrieval_query, pgvector_response, new_doc_response):
                response = pgvector_response  # type: ignore[assignment]
            elif new_doc_response is not None:
                response = new_doc_response
            elif pgvector_response is not None:
                response = pgvector_response
            else:
                response = self._build_doc_no_match_response(message=retrieval_query)
        else:
            hybrid_response = self._legacy_merge_doc_responses(retrieval_query, pgvector_response, new_doc_response)
            if hybrid_response is not None:
                response = hybrid_response
            elif self._should_prefer_pgvector(retrieval_query, pgvector_response, new_doc_response):
                response = pgvector_response  # type: ignore[assignment]
            elif new_doc_response is not None:
                response = new_doc_response
            elif pgvector_response is not None:
                response = pgvector_response
            else:
                response = self._build_doc_no_match_response(message=retrieval_query)
        logger.info(
            "[retrieve:merge] lane=%s mode=%s sources=%d elapsed=%.2fs",
            response.lane,
            response.mode,
            len(response.sources),
            time.perf_counter() - merge_started,
        )

        rerank_started = time.perf_counter()
        response = await self._rerank_doc_response(message=retrieval_query, response=response)
        logger.info(
            "[retrieve:rerank] mode=%s sources=%d elapsed=%.2fs",
            response.mode,
            len(response.sources),
            time.perf_counter() - rerank_started,
        )

        if not allow_llm_synthesis:
            enriched = self._enrich_doc_response(response, message=rewritten_query)
            if self._response_cache_enabled():
                self.response_cache.set(cache_key, enriched)
            return enriched
        response = await self._synthesize_doc_response(
            message=rewritten_query,
            planning_message=retrieval_query,
            response=response,
            answer_delta=answer_delta,
        )
        response = self._enrich_doc_response(response, message=rewritten_query)
        if self._response_cache_enabled():
            self.response_cache.set(cache_key, response)
        return response

    async def _retrieve_doc_candidates(
        self,
        *,
        message: str,
        sparse_allowed_source_paths: list[str] | None,
        dense_allowed_source_paths: list[str] | None,
        additional_query: str | None,
    ) -> tuple[CopilotChatResponse | None, CopilotChatResponse | None]:
        lexical_task = self.document_retriever.answer(
            message=message,
            allowed_source_paths=sparse_allowed_source_paths,
        )
        dense_task = self.pgvector_bridge.answer(
            message=message,
            allowed_source_paths=dense_allowed_source_paths,
            additional_query=additional_query,
        )
        lexical_result, dense_result = await self._gather_optional(lexical_task, dense_task)
        return lexical_result, dense_result

    @staticmethod
    async def _gather_optional(*tasks: Awaitable[CopilotChatResponse | None]) -> tuple[CopilotChatResponse | None, ...]:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        normalized: list[CopilotChatResponse | None] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("[retrieve] task failed: %s", result)
                normalized.append(None)
            else:
                normalized.append(result)
        return tuple(normalized)

    async def _rerank_doc_response(
        self,
        *,
        message: str,
        response: CopilotChatResponse,
    ) -> CopilotChatResponse:
        if not response.sources:
            return response
        deterministically_ranked = self._deterministic_rerank_sources(message, response.sources)
        if deterministically_ranked != response.sources:
            answer = (
                self._build_hybrid_doc_answer(message, deterministically_ranked)
                if response.lane == "doc_hybrid"
                else self._build_answer_from_sources(message, deterministically_ranked, response.mode)
            )
            response = response.model_copy(update={"sources": deterministically_ranked, "answer": answer})

        if not self._llm_enabled():
            return response
        scores = [float(source.score or 0.0) for source in response.sources[:6]]
        if self._gap_triggered_rerank_enabled():
            threshold = self._rerank_gap_threshold()
            if not needs_rerank(scores, threshold=threshold):
                logger.info("[rerank:decision] skip (gap above %.2f)", threshold)
                return response
        if self._can_skip_rerank(response.sources):
            return response

        ranked_candidates = response.sources[:6]
        source_lines: list[str] = []
        for index, source in enumerate(ranked_candidates, start=1):
            context = self._source_context_text(source, limit=360)
            if context:
                source_lines.append(f"[{index}] {context}")
        if len(source_lines) < 2:
            return response

        order_example = "[" + ",".join(str(i) for i in range(1, len(source_lines) + 1)) + "]"
        prompt = "\n".join(
            [
                "Reorder these RAG evidence candidates by relevance to the question.",
                "Return JSON only.",
                f'{{"order":{order_example}}}',
                f"Question: {message.strip()}",
                "Candidates:",
                *source_lines,
            ]
        )
        try:
            raw = await self.llm_client.generate(
                [
                    {"role": "system", "content": "You rerank RAG evidence by relevance."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=160,
                temperature=0.0,
                purpose="rerank",
            )
            parsed = IntentAgent._extract_json(raw) or {}
            order = parsed.get("order")
            if not isinstance(order, list):
                return response

            ordered_sources: list[CopilotChatSourceItem] = []
            used: set[int] = set()
            for item in order:
                try:
                    position = int(item) - 1
                except Exception:
                    continue
                if 0 <= position < len(ranked_candidates) and position not in used:
                    used.add(position)
                    ordered_sources.append(ranked_candidates[position])
            for index, source in enumerate(ranked_candidates):
                if index not in used:
                    ordered_sources.append(source)
            merged = [*ordered_sources, *response.sources[len(ranked_candidates):]]
            answer = (
                self._build_hybrid_doc_answer(message, merged)
                if response.lane == "doc_hybrid"
                else self._build_answer_from_sources(message, merged, response.mode)
            )
            return response.model_copy(update={"sources": merged, "answer": answer})
        except Exception:
            return response

    @staticmethod
    def _deterministic_rerank_sources(
        message: str,
        sources: list[CopilotChatSourceItem],
    ) -> list[CopilotChatSourceItem]:
        if len(sources) < 2:
            return sources

        subtopics = UnifiedCopilotService._query_subtopics(message)
        query_text = str(message or "").casefold()
        concept_like = any(marker in query_text for marker in ("뭐", "개념", "전반", "overview", "about", "purpose", "관계"))
        procedure_like = any(marker in query_text for marker in ("방법", "절차", "설정", "바꾸", "configuring", "listing", "using", "확인"))

        scored: list[tuple[float, int, CopilotChatSourceItem]] = []
        for index, source in enumerate(sources):
            title = str(source.metadata.get("section_title") or source.label or "")
            title_text = title.casefold()
            context = UnifiedCopilotService._source_context_text(source, limit=520)
            context_tokens = set(tokenize_query(context))
            title_tokens = set(tokenize_query(title))
            score = float(source.score or 0.0)

            clause_scores: list[float] = []
            for clause in subtopics:
                clause_tokens = set(tokenize_query(clause))
                if not clause_tokens:
                    continue
                overlap = len(clause_tokens.intersection(context_tokens))
                title_overlap = len(clause_tokens.intersection(title_tokens))
                coverage = overlap / max(len(clause_tokens), 1)
                exact_phrase = 1.0 if clause.casefold() in title_text else 0.0
                clause_scores.append((coverage * 2.0) + (title_overlap * 0.75) + exact_phrase)

            score += sum(sorted(clause_scores, reverse=True)[:2])

            if concept_like and any(marker in title_text for marker in ("overview", "about", "introduction", "purpose", "understanding", "chapter 1")):
                score += 0.35
            if procedure_like and any(marker in title_text for marker in ("configuring", "listing", "using", "viewing", "procedure", "steps", "switching", "changing")):
                score += 0.28
            if concept_like and "example" in title_text:
                score -= 0.22
            if procedure_like and "overview" in title_text and "listing" not in title_text and "viewing" not in title_text:
                score -= 0.12

            scored.append((score, -index, source))

        reranked = [source for _score, _neg_index, source in sorted(scored, reverse=True)]
        return reranked

    @staticmethod
    def _query_subtopics(message: str) -> list[str]:
        text = str(message or "").strip()
        if not text:
            return []
        normalized = text
        for marker in (" 그리고 ", "이랑 ", " 하고 ", " 및 ", " also ", " and ", "/", ","):
            normalized = normalized.replace(marker, "|")
        parts = [part.strip() for part in normalized.replace("?", "|").split("|") if part.strip()]
        return parts[:3] or [text]

    async def _synthesize_doc_response(
        self,
        *,
        message: str,
        response: CopilotChatResponse,
        planning_message: str | None = None,
        answer_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CopilotChatResponse:
        if not self._llm_enabled() or not response.sources:
            await self._emit_answer_text(response.answer, answer_delta)
            return response

        plan = self.answer_planner.plan(message=planning_message or message, sources=response.sources)
        context_lines: list[str] = []
        for index, evidence in enumerate(plan.ordered_evidences, start=1):
            context = evidence.context[:420 if plan.paragraph_count <= 2 else 520]
            if context:
                context_lines.append(f"[{index}] {context}")
        paragraph_plan_lines = [
            f"- Paragraph {index}: use sources " + "".join(f"[{source_index + 1}]" for source_index in group)
            for index, group in enumerate(plan.paragraph_source_indexes, start=1)
            if group
        ]
        if not context_lines:
            await self._emit_answer_text(response.answer, answer_delta)
            return response

        prompt_lines = [
            "Answer the user's question using only the document evidence below.",
            "Rules:",
            "- Do not invent facts beyond the evidence.",
            "- If the evidence is insufficient, say so plainly.",
            "- Write 2-3 short paragraphs.",
            "- Keep each factual claim close to the wording of the cited evidence.",
            "- Use the evidence lines that match each part of the question; if one paragraph combines multiple evidence lines, cite all of them.",
        ]
        if self._force_korean_answers():
            prompt_lines.extend(
                [
                    "- Respond in Korean.",
                    "- Translate English evidence into natural Korean.",
                    "- Keep commands, API names, resource names, YAML keys, field names, and literal identifiers unchanged.",
                ]
            )
        if self._native_citation_prompt_enabled():
            prompt_lines.extend(
                [
                    "- The first factual sentence must end with a citation marker.",
                    "- End each paragraph with citation markers like [1] or [1][2].",
                    "- Use only the source numbers provided below.",
                    "- Do not add a references section.",
                ]
            )
        else:
            prompt_lines.extend(
                [
                    "- Do not include inline citation markers.",
                    "- Do not add a references section.",
                ]
            )
        prompt_lines.extend(
            [
                f"Question: {message.strip()}",
                "Document evidence:",
                *context_lines,
                *(
                    ["Paragraph plan:", *paragraph_plan_lines]
                    if paragraph_plan_lines
                    else []
                ),
            ]
        )
        prompt = "\n".join(prompt_lines)
        try:
            if answer_delta is None:
                answer = await self.llm_client.generate(
                    [
                        {"role": "system", "content": "You are a grounded RAG answer synthesizer."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self._synthesis_max_tokens(),
                    temperature=0.1,
                    purpose="doc_synthesis",
                )
                cleaned = self._finalize_cited_answer(
                    answer.strip(),
                    plan.sources,
                    paragraph_source_indexes=plan.paragraph_source_indexes,
                )
                if not cleaned:
                    return response
                if self._should_fallback_to_extractive(
                    cleaned,
                    plan.sources,
                    response_mode=response.mode,
                    paragraph_source_indexes=plan.paragraph_source_indexes,
                ):
                    extractive = await self._localize_extractive_answer(
                        self._build_extractive_fallback_answer(plan.sources)
                    )
                    if extractive:
                        return response.model_copy(
                            update={
                                "answer": extractive,
                                "sources": plan.sources,
                            }
                        )
                cleaned, pruned_sources = self._prune_sources_to_citations(cleaned, plan.sources)
                return response.model_copy(
                    update={
                        "answer": cleaned,
                        "mode": f"{response.mode}_llm_grounded",
                        "sources": pruned_sources,
                    }
                )

            parts: list[str] = []
            async for token in self.llm_client.stream_chat(
                [
                    {"role": "system", "content": "You are a grounded RAG answer synthesizer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._synthesis_max_tokens(),
                temperature=0.1,
                purpose="doc_synthesis_stream",
            ):
                parts.append(token)
                await self._emit_answer_delta(answer_delta, token)
            cleaned = self._finalize_cited_answer(
                "".join(parts).strip(),
                plan.sources,
                paragraph_source_indexes=plan.paragraph_source_indexes,
            )
            if not cleaned:
                return response
            if self._should_fallback_to_extractive(
                cleaned,
                plan.sources,
                response_mode=response.mode,
                paragraph_source_indexes=plan.paragraph_source_indexes,
            ):
                extractive = await self._localize_extractive_answer(
                    self._build_extractive_fallback_answer(plan.sources)
                )
                if extractive:
                    return response.model_copy(
                        update={
                            "answer": extractive,
                            "sources": plan.sources,
                        }
                    )
            cleaned, pruned_sources = self._prune_sources_to_citations(cleaned, plan.sources)
            return response.model_copy(
                update={
                    "answer": cleaned,
                    "mode": f"{response.mode}_llm_grounded",
                    "sources": pruned_sources,
                }
            )
        except Exception:
            extractive = await self._localize_extractive_answer(
                self._build_extractive_fallback_answer(plan.sources or response.sources)
            )
            if extractive:
                extractive, pruned_sources = self._prune_sources_to_citations(extractive, plan.sources or response.sources)
                await self._emit_answer_text(extractive, answer_delta)
                return response.model_copy(
                    update={
                        "answer": extractive,
                        "sources": pruned_sources,
                    }
                )
            fallback = self._finalize_cited_answer(
                response.answer,
                plan.sources or response.sources,
                paragraph_source_indexes=plan.paragraph_source_indexes if plan.sources else None,
            )
            if fallback:
                await self._emit_answer_text(fallback, answer_delta)
                return response.model_copy(
                    update={
                        "answer": fallback,
                        "sources": plan.sources or response.sources,
                    }
                )
            attached = self._attach_paragraph_citations(
                response.answer,
                plan.sources or response.sources,
                paragraph_source_indexes=plan.paragraph_source_indexes if plan.sources else None,
            )
            if attached:
                validated = self.citation_validator.validate(
                    attached,
                    plan.sources or response.sources,
                    enforce_alignment=False,
                )
                if validated:
                    await self._emit_answer_text(validated, answer_delta)
                    return response.model_copy(
                        update={
                            "answer": validated,
                            "sources": plan.sources or response.sources,
                        }
                    )
            await self._emit_answer_text(response.answer, answer_delta)
            return response

    async def _build_mixed_response(
        self,
        *,
        message: str,
        doc_response: CopilotChatResponse,
        live_response: CopilotChatResponse | None,
        live_available: bool,
        answer_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CopilotChatResponse:
        focused_live_response = self._focus_live_response_for_message(message=message, live_response=live_response)
        combined_sources = [*doc_response.sources, *(focused_live_response.sources if focused_live_response else [])][:8]
        if self._llm_enabled():
            plan = self.answer_planner.plan(message=message, sources=doc_response.sources)
            context_lines: list[str] = []
            for index, evidence in enumerate(plan.ordered_evidences, start=1):
                context = evidence.context[:420 if plan.paragraph_count <= 2 else 520]
                if context:
                    context_lines.append(f"[{index}] {context}")
            paragraph_plan_lines = [
                f"- Paragraph {index}: use document sources " + "".join(f"[{source_index + 1}]" for source_index in group)
                for index, group in enumerate(plan.paragraph_source_indexes, start=1)
                if group
            ]
            live_summary = self._strip_intro(focused_live_response.answer) if focused_live_response is not None else ""
            if context_lines or live_summary or not live_available:
                prompt_parts = [
                    "Answer the user's question by combining the document evidence and the current cluster state.",
                    "Rules:",
                    "- Keep the answer grounded in the provided evidence and live status.",
                    "- Write 2-3 short paragraphs.",
                    "- Separate document-backed explanation from current cluster status when that makes the answer clearer.",
                    "- If the document guidance and current cluster status differ, explain the difference explicitly.",
                    "- Keep every document-backed claim close to the wording of the cited evidence.",
                    "- Use the evidence lines that match each part of the question; if one paragraph combines multiple document evidence lines, cite all of them.",
                ]
                if self._force_korean_answers():
                    prompt_parts.extend(
                        [
                            "- Respond in Korean.",
                            "- Translate English document evidence into natural Korean.",
                            "- Keep commands, API names, resource names, YAML keys, field names, and literal identifiers unchanged.",
                        ]
                    )
                if self._native_citation_prompt_enabled():
                    prompt_parts.extend(
                        [
                            "- The first document-backed sentence must end with a citation marker.",
                            "- Add document citation markers like [1] or [1][2] only where document evidence is used.",
                            "- Use only the document source numbers shown below.",
                            "- Do not add a references section.",
                        ]
                    )
                else:
                    prompt_parts.extend(
                        [
                            "- Do not include inline citation markers.",
                            "- Do not add a references section.",
                        ]
                    )
                prompt_parts.extend(
                    [
                        f"Question: {message.strip()}",
                        "Document evidence:",
                        *(context_lines or ["- none"]),
                        *(
                            ["Paragraph plan:", *paragraph_plan_lines]
                            if paragraph_plan_lines
                            else []
                        ),
                        "Current cluster state:",
                        live_summary or ("Connection unavailable; live status could not be checked." if not live_available else "- none"),
                    ]
                )
                try:
                    if answer_delta is None:
                        answer = await self.llm_client.generate(
                            [
                                {"role": "system", "content": "You synthesize mixed RAG + live cluster answers."},
                                {"role": "user", "content": "\n".join(prompt_parts)},
                            ],
                            max_tokens=self._synthesis_max_tokens(),
                            temperature=0.12,
                            purpose="mixed_synthesis",
                        )
                        doc_sources = plan.sources or doc_response.sources
                        combined_sources = [*doc_sources, *(focused_live_response.sources if focused_live_response else [])][:8]
                        cleaned = self._finalize_cited_answer(
                            answer.strip(),
                            combined_sources,
                            paragraph_source_indexes=plan.paragraph_source_indexes,
                        )
                    else:
                        parts: list[str] = []
                        async for token in self.llm_client.stream_chat(
                            [
                                {"role": "system", "content": "You synthesize mixed RAG + live cluster answers."},
                                {"role": "user", "content": "\n".join(prompt_parts)},
                            ],
                            max_tokens=self._synthesis_max_tokens(),
                            temperature=0.12,
                            purpose="mixed_synthesis_stream",
                        ):
                            parts.append(token)
                            await self._emit_answer_delta(answer_delta, token)
                        doc_sources = plan.sources or doc_response.sources
                        combined_sources = [*doc_sources, *(focused_live_response.sources if focused_live_response else [])][:8]
                        cleaned = self._finalize_cited_answer(
                            "".join(parts).strip(),
                            combined_sources,
                            paragraph_source_indexes=plan.paragraph_source_indexes,
                        )
                    if cleaned:
                        cleaned, pruned_sources = self._prune_sources_to_citations(cleaned, combined_sources)
                        return CopilotChatResponse(
                            lane="mixed",
                            mode="doc_plus_live_llm",
                            fallback_used=doc_response.fallback_used or bool(live_response and live_response.fallback_used),
                            preview_ready=doc_response.preview_ready,
                            answer=cleaned,
                            sources=pruned_sources,
                            artifacts=[*(doc_response.artifacts or []), *((focused_live_response.artifacts or []) if focused_live_response else [])][:6],
                            citation_map=self._build_citation_map(pruned_sources),
                        )
                except Exception:
                    pass

        answer = await self._build_deterministic_mixed_answer(
            message=message,
            doc_sources=doc_response.sources,
            live_response=focused_live_response,
            live_available=live_available,
        )
        await self._emit_answer_text(answer, answer_delta)
        return CopilotChatResponse(
            lane="mixed",
            mode="doc_plus_live",
            fallback_used=doc_response.fallback_used or bool(live_response and live_response.fallback_used),
            preview_ready=doc_response.preview_ready,
            answer=answer,
            sources=combined_sources,
            artifacts=[*(doc_response.artifacts or []), *((focused_live_response.artifacts or []) if focused_live_response else [])][:6],
            citation_map=self._build_citation_map(combined_sources),
        )

    async def _build_deterministic_mixed_answer(
        self,
        *,
        message: str,
        doc_sources: list[CopilotChatSourceItem],
        live_response: CopilotChatResponse | None,
        live_available: bool,
    ) -> str:
        targeted = await self._build_targeted_mixed_answer(
            message=message,
            doc_sources=doc_sources,
            live_response=live_response,
            live_available=live_available,
        )
        if targeted:
            return targeted

        parts: list[str] = []
        doc_section = await self._build_deterministic_doc_section(message=message, sources=doc_sources)
        if doc_section:
            parts.append(doc_section)

        if live_response is not None:
            live_text = self._strip_intro(live_response.answer)
            if live_text:
                parts.append("현재 OCP 상태\n" + live_text)
        elif not live_available:
            parts.append("현재 OCP 상태\n클러스터 연결이 없어 현재 상태를 확인하지 못했습니다.")

        if not parts:
            fallback = await self._localize_extractive_answer(self._strip_intro(message))
            return fallback or str(message or "").strip()

        return "\n\n".join(part for part in parts if part.strip()).strip()

    async def _build_targeted_mixed_answer(
        self,
        *,
        message: str,
        doc_sources: list[CopilotChatSourceItem],
        live_response: CopilotChatResponse | None,
        live_available: bool,
    ) -> str:
        lowered = str(message or "").casefold()
        if "영향" not in lowered and "impact" not in lowered:
            return ""

        live_source = None
        if live_response is not None and live_response.sources:
            for source in live_response.sources:
                if source.source_type == "live":
                    live_source = source
                    break
        if live_source is None:
            return ""

        metadata = dict(live_source.metadata or {})
        replicas = metadata.get("replicas")
        ready_replicas = metadata.get("ready_replicas")
        name = str(live_source.label or "")
        namespace = str(live_source.namespace or "")

        doc_section = await self._build_deterministic_doc_section(message=message, sources=doc_sources)
        parts: list[str] = []
        if doc_section:
            parts.append(doc_section)

        impact_lines = []
        if replicas not in {None, ""}:
            impact_lines.append(f"- 현재 desired replicas는 {replicas}입니다.")
        if ready_replicas not in {None, ""}:
            impact_lines.append(f"- 현재 ready replicas는 {ready_replicas}입니다.")
        if replicas not in {None, ""} and ready_replicas not in {None, ""}:
            if int(ready_replicas) == int(replicas):
                impact_lines.append("- 현재 값 기준으로는 요청된 replica 수만큼 인스턴스가 준비되어 있어 추가 용량이 실제로 반영된 상태로 볼 수 있습니다.")
            else:
                impact_lines.append("- 현재 값 기준으로는 desired replica 수와 ready replica 수가 아직 다르므로 확장 효과가 완전히 반영되었는지 추가 확인이 필요합니다.")
        impact_lines.extend(
            [
                "- replica 수가 늘어나면 동일 워크로드 인스턴스 수가 증가하므로 트래픽 분산과 처리 여유가 커질 수 있습니다.",
                "- 반대로 replica 수가 줄어들면 사용 가능한 인스턴스 수가 감소하므로 순간 부하나 장애 허용 범위를 다시 점검해야 합니다.",
            ]
        )
        parts.append(
            "\n".join(
                [
                    f"현재 OCP 상태",
                    f"{namespace} namespace의 Deployment {name} 변경 영향 요약입니다.",
                    *impact_lines,
                ]
            ).strip()
        )
        if not live_available:
            parts.append("참고: 현재 클러스터 연결이 불안정하면 실제 반영 상태를 다시 확인해야 합니다.")
        return "\n\n".join(part for part in parts if part.strip()).strip()

    async def _build_deterministic_doc_section(self, *, message: str, sources: list[CopilotChatSourceItem]) -> str:
        doc_sources = [source for source in sources if source.source_type == "doc"][:3]
        if not doc_sources:
            return ""

        targeted = await self._build_targeted_doc_section(message=message, sources=doc_sources)
        if targeted:
            return targeted

        rendered: list[str] = ["문서 기준"]
        for index, source in enumerate(doc_sources, start=1):
            section = str(source.metadata.get("section_title") or source.label or "document").strip()
            preview = str(source.metadata.get("preview_text") or source.metadata.get("synthesis_text") or "").strip()
            if not preview:
                continue
            cleaned_preview = self._clean_doc_preview_text(section=section, preview=preview[:320])
            localized = await self._localize_extractive_answer(cleaned_preview)
            localized = self._clean_doc_preview_text(section=section, preview=localized or cleaned_preview)
            rendered.append(f"- {section}: {localized}[{index}]")
        return "\n".join(rendered).strip()

    async def _build_targeted_doc_section(
        self,
        *,
        message: str,
        sources: list[CopilotChatSourceItem],
    ) -> str:
        lowered = str(message or "").casefold()
        if "rollout" in lowered and "history" in lowered:
            commands = self._extract_shell_commands_from_sources(sources)
            rollout_commands = [cmd for cmd in commands if "oc rollout history" in cmd.casefold()]
            describe_commands = [cmd for cmd in commands if "oc describe" in cmd.casefold()]
            lines = ["문서 기준"]
            resource_name = self._extract_named_resource_from_text(message)
            scope = f"{resource_name} 기준으로 " if resource_name else ""
            if rollout_commands:
                lines.append(f"- {scope}rollout history 확인은 `{rollout_commands[0]}` 명령으로 최근 revision 이력을 확인하는 방식입니다.[1]")
            if len(rollout_commands) > 1:
                lines.append(f"- 특정 revision 상세 확인은 `{rollout_commands[1]}` 처럼 `--revision` 옵션을 붙여 확인할 수 있습니다.[1]")
            if describe_commands:
                lines.append(f"- 더 자세한 상태 확인은 `{describe_commands[0]}` 명령으로 보강할 수 있습니다.[1]")
            if len(lines) > 1:
                return "\n".join(lines)
        return ""

    def _focus_live_response_for_message(
        self,
        *,
        message: str,
        live_response: CopilotChatResponse | None,
    ) -> CopilotChatResponse | None:
        if live_response is None:
            return None
        if live_response.mode != "tool:resource_list":
            return live_response

        lowered_message = str(message or "").casefold()
        matching_sources = [
            source
            for source in live_response.sources
            if source.source_type == "live"
            and str(source.label or "").strip()
            and str(source.label).casefold() in lowered_message
        ]
        selected = matching_sources[0] if len(matching_sources) == 1 else None
        if selected is None:
            selected = self._build_live_source_from_artifacts(
                lowered_message=lowered_message,
                artifacts=live_response.artifacts,
            )
        if selected is None:
            return live_response

        filtered_artifacts = self._filter_live_artifacts_to_label(
            artifacts=live_response.artifacts,
            label=str(selected.label or ""),
        )
        focused_answer = self._build_focused_live_summary(source=selected)
        return live_response.model_copy(
            update={
                "answer": focused_answer,
                "sources": [selected],
                "artifacts": filtered_artifacts,
            }
        )

    @staticmethod
    def _filter_live_artifacts_to_label(
        *,
        artifacts: list[CopilotChatArtifact],
        label: str,
    ) -> list[CopilotChatArtifact]:
        filtered: list[CopilotChatArtifact] = []
        for artifact in artifacts or []:
            if artifact.artifact_type != "resource_list":
                filtered.append(artifact)
                continue
            matched_items = [item for item in artifact.items if str(item.name or "") == label]
            if not matched_items:
                continue
            filtered.append(
                artifact.model_copy(
                    update={
                        "items": matched_items,
                        "payload": {**artifact.payload, "count": len(matched_items)},
                    }
                )
            )
        return filtered

    @staticmethod
    def _build_live_source_from_artifacts(
        *,
        lowered_message: str,
        artifacts: list[CopilotChatArtifact],
    ) -> CopilotChatSourceItem | None:
        for artifact in artifacts or []:
            if artifact.artifact_type != "resource_list":
                continue
            for item in artifact.items:
                label = str(item.name or "").strip()
                if not label or label.casefold() not in lowered_message:
                    continue
                metadata = dict(item.metadata or {})
                return CopilotChatSourceItem(
                    source_type="live",
                    label=label,
                    namespace=str(item.namespace or ""),
                    kind=str(item.kind or ""),
                    provenance=["live", "tool:list_resources"],
                    metadata=metadata,
                )
        return None

    @staticmethod
    def _build_focused_live_summary(*, source: CopilotChatSourceItem) -> str:
        namespace = str(source.namespace or "-")
        kind = str(source.kind or "Resource")
        name = str(source.label or "")
        metadata = dict(source.metadata or {})
        summary_lines = [f"{namespace} namespace의 {kind} {name} 현재 상태입니다."]
        replicas = metadata.get("replicas")
        ready_replicas = metadata.get("ready_replicas")
        if replicas not in {None, ""}:
            summary_lines.append(f"- replicas: {replicas}")
        if ready_replicas not in {None, ""}:
            summary_lines.append(f"- ready_replicas: {ready_replicas}")
        phase = str(metadata.get("phase") or "").strip()
        if phase:
            summary_lines.append(f"- phase: {phase}")
        host = str(metadata.get("host") or "").strip()
        if host:
            summary_lines.append(f"- host: {host}")
        return "\n".join(summary_lines)

    @staticmethod
    def _extract_shell_commands_from_sources(sources: list[CopilotChatSourceItem]) -> list[str]:
        commands: list[str] = []
        for source in sources:
            for candidate in (
                str(source.metadata.get("synthesis_text") or ""),
                str(source.metadata.get("preview_text") or ""),
            ):
                for match in re.findall(r"\$ ([^\n`]+)", candidate):
                    command = match.strip()
                    if command and command not in commands:
                        commands.append(command)
        return commands

    @staticmethod
    def _extract_named_resource_from_text(text: str) -> str:
        match = re.search(r"([a-z0-9][a-z0-9._-]+)\s+deployment", str(text or "").casefold())
        return match.group(1).strip() if match else ""

    @staticmethod
    def _clean_doc_preview_text(*, section: str, preview: str) -> str:
        text = " ".join(str(preview or "").split()).strip()
        title = " ".join(str(section or "").split()).strip()
        if not text:
            return ""
        if title and text.casefold().startswith(title.casefold()):
            text = text[len(title):].lstrip(" :.-")
        doubled_title = f"{title}: {title}" if title else ""
        if doubled_title and text.casefold().startswith(doubled_title.casefold()):
            text = text[len(doubled_title):].lstrip(" :.-")
        return text.strip()

    @staticmethod
    def _map_live_item(item: OcpLiveResourceSummary) -> CopilotChatSourceItem:
        return CopilotChatSourceItem(
            source_type="live",
            label=item.name,
            namespace=item.namespace,
            kind=item.kind,
            provenance=["live"],
            metadata=item.model_dump(),
        )

    @staticmethod
    def _strip_intro(answer: str) -> str:
        lines = [line.rstrip() for line in str(answer or "").splitlines()]
        if not lines:
            return ""
        if lines[0].startswith("Combined summary of the document evidence and current cluster state."):
            return "\n".join(lines[1:]).strip()
        return "\n".join(lines).strip()

    @staticmethod
    def _should_prefer_pgvector(
        message: str,
        pgvector_response: CopilotChatResponse | None,
        new_doc_response: CopilotChatResponse | None,
    ) -> bool:
        if pgvector_response is None:
            return False
        if new_doc_response is None:
            return True
        lowered = str(message or "").casefold()
        conceptual = any(marker in lowered for marker in ("what is", "difference", "explain", "configuration", "pattern", "concept"))
        pg_score = max((source.score for source in pgvector_response.sources), default=0.0)
        new_score = max((source.score for source in new_doc_response.sources), default=0.0)
        return conceptual and pg_score >= new_score

    @staticmethod
    def _should_prefer_hinted_sparse(
        source_hints: list[str],
        new_doc_response: CopilotChatResponse | None,
        pgvector_response: CopilotChatResponse | None,
    ) -> bool:
        if not source_hints or new_doc_response is None or not new_doc_response.sources:
            return False
        hinted = {str(path or "").replace("\\", "/") for path in source_hints if str(path or "").strip()}
        if not hinted:
            return False
        sparse_in_hint = [
            source
            for source in new_doc_response.sources
            if str(source.source_path or "").replace("\\", "/") in hinted
        ]
        if not sparse_in_hint:
            return False
        if pgvector_response is None or not pgvector_response.sources:
            return True
        dense_in_hint = any(
            str(source.source_path or "").replace("\\", "/") in hinted
            for source in pgvector_response.sources
        )
        return not dense_in_hint

    @staticmethod
    def _merge_doc_responses(
        message: str,
        pgvector_response: CopilotChatResponse | None,
        new_doc_response: CopilotChatResponse | None,
    ) -> CopilotChatResponse | None:
        top_sources = merge_hybrid_sources(
            message,
            pgvector_response,
            new_doc_response,
            source_key_builder=UnifiedCopilotService._source_key,
            source_merger=UnifiedCopilotService._merge_source_details,
        )
        if not top_sources:
            return None
        answer = UnifiedCopilotService._build_hybrid_doc_answer(message, top_sources)
        return CopilotChatResponse(
            lane="doc_hybrid",
            mode="hybrid_rrf_doc",
            fallback_used=False,
            preview_ready=any(bool(source.source_path) for source in top_sources),
            answer=answer,
            sources=top_sources,
            artifacts=[],
            citation_map=UnifiedCopilotService._build_citation_map(top_sources),
        )

    @staticmethod
    def _legacy_merge_doc_responses(
        message: str,
        pgvector_response: CopilotChatResponse | None,
        new_doc_response: CopilotChatResponse | None,
    ) -> CopilotChatResponse | None:
        if pgvector_response is None or new_doc_response is None:
            return None
        if not pgvector_response.sources or not new_doc_response.sources:
            return None

        merged_scores: dict[str, float] = {}
        merged_sources: dict[str, CopilotChatSourceItem] = {}
        merged_provenance: dict[str, set[str]] = {}
        for weight, response in ((1.8, pgvector_response), (0.7, new_doc_response)):
            for index, source in enumerate(response.sources):
                key = UnifiedCopilotService._source_key(source)
                merged_scores[key] = merged_scores.get(key, 0.0) + weight * (1.0 / (10 + index))
                merged_provenance.setdefault(key, set()).update(source.provenance or [response.lane])
                if key not in merged_sources:
                    merged_sources[key] = source
                else:
                    merged_sources[key] = UnifiedCopilotService._merge_source_details(merged_sources[key], source)

        ranked = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
        top_sources: list[CopilotChatSourceItem] = []
        answer_limit = answer_source_budget(message) + 1
        for key, score in ranked[:answer_limit]:
            source = merged_sources[key]
            top_sources.append(
                source.model_copy(
                    update={
                        "score": round(score, 3),
                        "provenance": sorted(merged_provenance.get(key) or []),
                        "metadata": {
                            **source.metadata,
                            "hybrid_origin_lanes": sorted(merged_provenance.get(key) or []),
                            "retrieval_backend": "doc_hybrid",
                        },
                    }
                )
            )
        answer = UnifiedCopilotService._build_hybrid_doc_answer(message, top_sources)
        return CopilotChatResponse(
            lane="doc_hybrid",
            mode="hybrid_rrf_doc",
            fallback_used=False,
            preview_ready=any(bool(source.source_path) for source in top_sources),
            answer=answer,
            sources=top_sources,
            artifacts=[],
            citation_map=UnifiedCopilotService._build_citation_map(top_sources),
        )

    @staticmethod
    def _build_hybrid_doc_answer(message: str, sources: Iterable[CopilotChatSourceItem]) -> str:
        parts: list[str] = []
        count = 0
        for source in sources:
            preview_text = str(source.metadata.get("preview_text") or "").strip()
            section_title = str(source.metadata.get("section_title") or source.label)
            if not preview_text:
                continue
            parts.append(f"{section_title}\n{preview_text[:260]}")
            count += 1
            if count >= 3:
                break
        if count == 0:
            return message.strip()
        return "\n\n".join(parts).strip()

    @staticmethod
    def _build_answer_from_sources(message: str, sources: Iterable[CopilotChatSourceItem], mode: str) -> str:
        del mode
        parts: list[str] = []
        count = 0
        for source in sources:
            preview = str(source.metadata.get("preview_text") or "").strip()
            section = str(source.metadata.get("section_title") or source.label or "document").strip()
            if not preview:
                continue
            parts.append(f"{section}\n{preview[:260]}")
            count += 1
            if count >= 3:
                break
        if count == 0:
            return message.strip()
        return "\n\n".join(parts).strip()

    @staticmethod
    def _build_citation_map(sources: Iterable[CopilotChatSourceItem]) -> list[CopilotCitationMapItem]:
        citation_map: list[CopilotCitationMapItem] = []
        for index, source in enumerate(sources, start=1):
            citation_map.append(
                CopilotCitationMapItem(
                    citation_number=index,
                    source_index=index - 1,
                    chunk_id=str(source.chunk_id or ""),
                    section_title=str(source.metadata.get("section_title") or source.label or "").strip(),
                    supporting_text=str(
                        source.metadata.get("preview_text")
                        or source.metadata.get("synthesis_text")
                        or source.metadata.get("manifest_yaml")
                        or ""
                    ).strip()[:420],
                    command_text=UnifiedCopilotService._extract_command_text(source),
                )
            )
        return citation_map

    @staticmethod
    def _extract_command_text(source: CopilotChatSourceItem) -> str:
        for candidate in (
            str(source.metadata.get("preview_text") or ""),
            str(source.metadata.get("synthesis_text") or ""),
        ):
            matched = re.search(r"`([^`]+)`", candidate)
            if matched:
                return matched.group(1).strip()
        return ""

    def _with_citation_map(self, response: CopilotChatResponse) -> CopilotChatResponse:
        return response.model_copy(update={"citation_map": self._build_citation_map(response.sources)})

    def _enrich_doc_response(self, response: CopilotChatResponse, *, message: str) -> CopilotChatResponse:
        artifacts = list(response.artifacts or [])
        command_artifact = self._build_command_template_artifact(message=message, sources=response.sources)
        if command_artifact is not None:
            artifacts.append(command_artifact)
        answer = self._anchor_answer_to_question(response.answer, message=message)
        return response.model_copy(
            update={
                "answer": answer,
                "artifacts": artifacts,
                "citation_map": self._build_citation_map(response.sources),
            }
        )

    @staticmethod
    def _anchor_answer_to_question(answer: str, *, message: str) -> str:
        text = str(answer or "").strip()
        question = str(message or "").strip()
        if not text or not question:
            return text

        lowered_answer = text.casefold()
        lowered_question = question.casefold()

        resource_match = re.search(
            r"([a-z0-9][a-z0-9._-]+)\s+(deployment|pod|service|route)",
            lowered_question,
        )
        focus_terms: list[str] = []
        for token in ("rollout", "history", "replica", "체크리스트", "운영", "문서"):
            if token in lowered_question and token not in lowered_answer:
                focus_terms.append(token)

        prefix_parts: list[str] = []
        if resource_match:
            resource_name = resource_match.group(1)
            resource_kind = resource_match.group(2)
            if resource_name not in lowered_answer:
                prefix_parts.append(f"{resource_name} {resource_kind} 기준으로 보면")
        if focus_terms:
            korean_terms = ", ".join(focus_terms)
            prefix_parts.append(f"질문의 초점은 {korean_terms}입니다")

        if not prefix_parts:
            return text

        prefix = ". ".join(prefix_parts).strip()
        if not prefix.endswith("."):
            prefix += "."
        return f"{prefix} {text}".strip()

    def _build_command_template_artifact(
        self,
        *,
        message: str,
        sources: list[CopilotChatSourceItem],
    ) -> CopilotChatArtifact | None:
        lowered = str(message or "").casefold()
        wants_command = any(marker in lowered for marker in ("명령어", "command", "cli", "oc ", "kubectl", "뭐 쳐"))
        if not wants_command:
            return None
        command = ""
        source_index = 0
        for index, source in enumerate(sources, start=1):
            command = self._extract_command_text(source)
            if command:
                source_index = index
                break
        if not command:
            return None
        slots = []
        if "{namespace}" in command:
            slots.append("namespace")
        return CopilotChatArtifact(
            artifact_type="command_template",
            title="Canonical command",
            description="Exact command text grounded in retrieved evidence.",
            payload={
                "template": command,
                "resolved_command": command,
                "slots": slots,
                "source_index": source_index,
            },
        )

    @staticmethod
    def _build_extractive_fallback_answer(sources: list[CopilotChatSourceItem]) -> str:
        paragraphs: list[str] = []
        doc_index = 0
        for source in sources:
            if source.source_type != "doc":
                continue
            preview = str(source.metadata.get("preview_text") or source.metadata.get("synthesis_text") or "").strip()
            section = str(source.metadata.get("section_title") or source.label or "document").strip()
            if not preview:
                continue
            doc_index += 1
            paragraphs.append(f"{section}: {preview[:320]}[{doc_index}]")
            if len(paragraphs) >= 2:
                break
        return "\n\n".join(paragraphs).strip()

    async def _localize_extractive_answer(self, answer: str) -> str:
        text = str(answer or "").strip()
        if not text or not self._force_korean_answers() or not self._llm_enabled():
            return text
        try:
            localized = await self.llm_client.generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "Translate grounded documentation answers into Korean. "
                            "Preserve citation markers like [1], commands in backticks, API/resource names, "
                            "YAML keys, field names, and literal English identifiers."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Translate this answer into Korean without adding facts:\n\n{text}",
                    },
                ],
                max_tokens=self._synthesis_max_tokens(),
                temperature=0.0,
                purpose="doc_synthesis",
            )
            return str(localized or "").strip() or text
        except Exception:
            return text

    @staticmethod
    def _prune_sources_to_citations(
        answer: str,
        sources: list[CopilotChatSourceItem],
    ) -> tuple[str, list[CopilotChatSourceItem]]:
        text = str(answer or "").strip()
        if not text or not sources:
            return text, sources

        cited_order: list[int] = []
        for match in re.findall(r"\[(\d+)\]", text):
            try:
                index = int(match) - 1
            except ValueError:
                continue
            if 0 <= index < len(sources) and index not in cited_order:
                cited_order.append(index)

        if not cited_order:
            return text, sources

        keep_indexes = list(cited_order)
        answer_body = re.sub(r"\[(\d+)\]", "", text).strip()
        cited_scores = [
            UnifiedCopilotService._source_support_score(answer_body, sources[index])
            for index in cited_order
            if 0 <= index < len(sources)
        ]
        best_cited_score = max(cited_scores, default=0.0)

        for index, source in enumerate(sources):
            if index in keep_indexes:
                continue
            if source.source_type != "doc":
                keep_indexes.append(index)
                continue

            support_score = UnifiedCopilotService._source_support_score(answer_body, source)
            same_path_as_cited = any(
                str(source.source_path or "") == str(sources[cited_index].source_path or "")
                for cited_index in cited_order
                if 0 <= cited_index < len(sources)
            )
            if same_path_as_cited and support_score >= max(best_cited_score * 0.45, 0.12):
                keep_indexes.append(index)
                continue
            if support_score >= max(best_cited_score * 0.72, 0.18):
                keep_indexes.append(index)

        remap = {old_index: new_index for new_index, old_index in enumerate(keep_indexes, start=1)}

        def replace(match: re.Match[str]) -> str:
            try:
                old_index = int(match.group(1)) - 1
            except ValueError:
                return ""
            new_index = remap.get(old_index)
            return f"[{new_index}]" if new_index is not None else ""

        remapped_answer = re.sub(r"\[(\d+)\]", replace, text)
        pruned_sources = [sources[index] for index in keep_indexes]
        return remapped_answer, pruned_sources

    @staticmethod
    def _source_support_score(answer_body: str, source: CopilotChatSourceItem) -> float:
        answer_tokens = UnifiedCopilotService._alignment_tokens(answer_body)
        if not answer_tokens:
            return 0.0
        context = " ".join(
            [
                str(source.metadata.get("section_title") or source.label or ""),
                str(source.metadata.get("preview_text") or ""),
                str(source.metadata.get("synthesis_text") or ""),
            ]
        )
        source_tokens = UnifiedCopilotService._alignment_tokens(context)
        if not source_tokens:
            return 0.0
        overlap = len(answer_tokens.intersection(source_tokens))
        if overlap <= 0:
            return 0.0
        title_tokens = UnifiedCopilotService._alignment_tokens(str(source.metadata.get("section_title") or source.label or ""))
        title_overlap = len(answer_tokens.intersection(title_tokens))
        coverage = overlap / max(len(answer_tokens), 1)
        return float(overlap) + coverage + (title_overlap * 1.5)

    @staticmethod
    def _source_context_text(source: CopilotChatSourceItem, *, limit: int) -> str:
        section = str(source.metadata.get("section_title") or source.label or "document").strip()
        synthesis_text = str(source.metadata.get("synthesis_text") or source.metadata.get("preview_text") or "").strip()
        if not synthesis_text:
            return ""
        normalized = " ".join(synthesis_text.split())
        return f"{section}: {normalized[:limit]}"

    @staticmethod
    def _source_key(source: CopilotChatSourceItem) -> str:
        return "|".join([source.source_type, source.source_path or "", source.chunk_id or "", source.label or ""])

    @staticmethod
    def _merge_source_details(current: CopilotChatSourceItem, incoming: CopilotChatSourceItem) -> CopilotChatSourceItem:
        merged_metadata = dict(current.metadata)
        for key, value in incoming.metadata.items():
            if key not in merged_metadata or not merged_metadata.get(key):
                merged_metadata[key] = value
        if "preview_text" in current.metadata and "preview_text" in incoming.metadata:
            merged_metadata["preview_text"] = current.metadata.get("preview_text") or incoming.metadata.get("preview_text")
        return current.model_copy(
            update={
                "relative_source_path": current.relative_source_path or incoming.relative_source_path,
                "page_number": current.page_number or incoming.page_number,
                "metadata": merged_metadata,
            }
        )

    @staticmethod
    def _build_doc_no_match_response(*, message: str) -> CopilotChatResponse:
        return CopilotChatResponse(
            lane="doc_new",
            mode="no_match",
            fallback_used=False,
            preview_ready=False,
            answer=(
                "새 문서 retrieval 경로에서 바로 사용할 근거를 찾지 못했습니다. "
                "질문을 조금 더 구체화하거나 source 범위를 좁혀 다시 시도해 주세요.\n"
                f"- query: {message}"
            ),
            sources=[],
        )

    def _finalize_cited_answer(
        self,
        answer: str,
        sources: list[CopilotChatSourceItem],
        *,
        paragraph_source_indexes: list[list[int]] | None = None,
    ) -> str:
        raw = str(answer or "").strip()
        stripped = re.sub(r"\[(\d+)\]", "", raw).strip()
        if paragraph_source_indexes:
            cited = self._attach_paragraph_citations(
                stripped,
                sources,
                paragraph_source_indexes=paragraph_source_indexes,
            )
        elif self._native_citation_prompt_enabled() and re.search(r"\[\d+\]", raw):
            cited = raw
        else:
            cited = self._attach_paragraph_citations(
                stripped,
                sources,
                paragraph_source_indexes=paragraph_source_indexes,
            )
        return self.citation_validator.validate(
            cited,
            sources,
            enforce_alignment=paragraph_source_indexes is None,
        )

    def _should_fallback_to_extractive(
        self,
        answer: str,
        sources: list[CopilotChatSourceItem],
        *,
        response_mode: str,
        paragraph_source_indexes: list[list[int]] | None = None,
    ) -> bool:
        if not str(response_mode).startswith(("hybrid_rrf_doc", "pgvector_dense")):
            return False
        if not answer.strip() or not sources:
            return False
        if self._answer_contains_unsupported_command(answer, sources):
            return True
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
        if not paragraphs:
            return False
        saw_citation = False
        planned_pointer = 0
        for paragraph in paragraphs:
            citations = [int(match) for match in re.findall(r"\[(\d+)\]", paragraph)]
            if not citations:
                continue
            saw_citation = True
            body = re.sub(r"\[(\d+)\]", "", paragraph).strip()
            if not body:
                return True
            expected_indexes: list[int] = []
            if paragraph_source_indexes and planned_pointer < len(paragraph_source_indexes):
                expected_indexes = paragraph_source_indexes[planned_pointer]
                planned_pointer += 1
            candidate_indexes = expected_indexes or [index - 1 for index in citations if 1 <= index <= len(sources)]
            if not candidate_indexes:
                return True
            if not any(self.citation_validator._supports_paragraph(body, sources[index]) for index in candidate_indexes):
                return True
        return not saw_citation

    @staticmethod
    def _answer_contains_unsupported_command(answer: str, sources: list[CopilotChatSourceItem]) -> bool:
        commands = re.findall(r"`([^`]+)`", str(answer or ""))
        if not commands:
            return False
        context_parts: list[str] = []
        for source in sources:
            context_parts.append(str(source.metadata.get("preview_text") or ""))
            context_parts.append(str(source.metadata.get("synthesis_text") or ""))
        source_context = " ".join(context_parts)
        normalized_context = source_context.casefold()
        for command in commands:
            if command.strip() and command.casefold() not in normalized_context:
                return True
        return False

    @staticmethod
    def _attach_paragraph_citations(
        answer: str,
        sources: list[CopilotChatSourceItem],
        *,
        paragraph_source_indexes: list[list[int]] | None = None,
    ) -> str:
        doc_sources = [(index, source) for index, source in enumerate(sources) if source.source_type == "doc"]
        if not answer.strip() or not doc_sources:
            return answer.strip()

        blocks = [block.strip() for block in re.split(r"\n\s*\n", answer.strip()) if block.strip()]
        rendered: list[str] = []
        citation_added = False
        planned_pointer = 0
        for block in blocks:
            if UnifiedCopilotService._block_is_non_citable(block):
                rendered.append(block)
                continue
            source_indexes: list[int] = []
            if paragraph_source_indexes and planned_pointer < len(paragraph_source_indexes):
                source_indexes = paragraph_source_indexes[planned_pointer]
                planned_pointer += 1
            if not source_indexes:
                source_indexes = UnifiedCopilotService._match_sources_to_block(block, doc_sources)
            if not source_indexes:
                rendered.append(block)
                continue
            marker = "".join(f"[{index + 1}]" for index in source_indexes)
            rendered.append(UnifiedCopilotService._append_citation_marker(block, marker))
            citation_added = True
        if not citation_added and len(doc_sources) == 1 and rendered:
            rendered[-1] = UnifiedCopilotService._append_citation_marker(rendered[-1], f"[{doc_sources[0][0] + 1}]")
        return "\n\n".join(rendered).strip()

    @staticmethod
    def _block_is_non_citable(block: str) -> bool:
        trimmed = block.strip()
        return trimmed.startswith("```") or trimmed.startswith("#") or len(trimmed) < 8

    @staticmethod
    def _append_citation_marker(block: str, marker: str) -> str:
        lines = block.rstrip().splitlines()
        if not lines:
            return block
        lines[-1] = f"{lines[-1].rstrip()}{marker}".rstrip()
        return "\n".join(lines)

    @staticmethod
    def _can_skip_rerank(sources: list[CopilotChatSourceItem]) -> bool:
        if len(sources) < 2:
            return True
        first = float(sources[0].score or 0.0)
        second = float(sources[1].score or 0.0)
        return first > 0 and (first - second) >= 0.25

    @staticmethod
    def _match_sources_to_block(block: str, doc_sources: list[tuple[int, CopilotChatSourceItem]]) -> list[int]:
        block_tokens = UnifiedCopilotService._alignment_tokens(block)
        if not block_tokens:
            return []

        scored: list[tuple[float, int]] = []
        for index, source in doc_sources:
            context = " ".join(
                [
                    str(source.metadata.get("section_title") or source.label or ""),
                    str(source.metadata.get("preview_text") or ""),
                    str(source.metadata.get("synthesis_text") or ""),
                ]
            )
            source_tokens = UnifiedCopilotService._alignment_tokens(context)
            if not source_tokens:
                continue
            overlap = len(block_tokens.intersection(source_tokens))
            if overlap == 0:
                continue
            title_tokens = UnifiedCopilotService._alignment_tokens(str(source.metadata.get("section_title") or source.label or ""))
            title_overlap = len(block_tokens.intersection(title_tokens))
            scored.append((float(overlap) + (title_overlap * 1.5), index))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return [doc_sources[0][0]] if len(doc_sources) == 1 else []

        selected = [scored[0][1]]
        if len(scored) > 1 and scored[1][0] >= scored[0][0] * 0.75:
            selected.append(scored[1][1])
        return selected

    @staticmethod
    def _alignment_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9가-힣_-]+", str(text or "").casefold())
            if len(token) >= 2
        }

    def _llm_enabled(self) -> bool:
        return self.llm_client is not None and self.llm_client.is_enabled

    def _response_cache_enabled(self) -> bool:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return True
        return bool(getattr(settings, "use_response_cache", True))

    def _force_korean_answers(self) -> bool:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return True
        return bool(getattr(settings, "force_korean_answers", True))

    def _synthesis_max_tokens(self) -> int:
        if self.llm_client is None:
            return 600
        settings = getattr(self.llm_client, "settings", None)
        return int(getattr(settings, "llm_synthesis_max_tokens", 600) or 600)

    def _native_citation_prompt_enabled(self) -> bool:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return True
        return bool(getattr(settings, "use_native_citation_prompt", True))

    def _query_router_enabled(self) -> bool:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return False
        return bool(getattr(settings, "use_query_router", False))

    def _rrf_fusion_enabled(self) -> bool:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return True
        return bool(getattr(settings, "use_rrf_fusion", True))

    def _synonym_expansion_enabled(self) -> bool:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return True
        return bool(getattr(settings, "use_synonym_expansion", True))

    def _gap_triggered_rerank_enabled(self) -> bool:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return True
        return bool(getattr(settings, "use_gap_triggered_rerank", True))

    def _rerank_gap_threshold(self) -> float:
        settings = getattr(self.llm_client, "settings", None) if self.llm_client is not None else None
        if settings is None:
            return 0.15
        return float(getattr(settings, "rerank_gap_threshold", 0.15) or 0.15)

    @staticmethod
    def _last_assistant_lane(recent_turns: list[CopilotChatHistoryTurn]) -> str:
        for turn in reversed(recent_turns):
            if str(turn.role or "") == "assistant" and str(turn.lane or "").strip():
                return str(turn.lane or "").strip()
        return ""

    @staticmethod
    def _last_doc_source_paths(recent_turns: list[CopilotChatHistoryTurn]) -> list[str]:
        for turn in reversed(recent_turns):
            if str(turn.role or "") != "assistant":
                continue
            source_paths = getattr(turn, "source_paths", None) or []
            if source_paths:
                return [path for path in source_paths if str(path or "").strip()]
        return []

    @staticmethod
    async def _emit_progress(
        callback: Callable[[CopilotChatStage], Awaitable[None] | None] | None,
        *,
        key: str,
        label: str,
        detail: str,
    ) -> None:
        if callback is None:
            return
        outcome = callback(CopilotChatStage(key=key, label=label, detail=detail, status="running"))
        if inspect.isawaitable(outcome):
            await outcome

    @staticmethod
    async def _emit_answer_delta(callback: Callable[[str], Awaitable[None] | None] | None, delta: str) -> None:
        if callback is None or not delta:
            return
        outcome = callback(delta)
        if inspect.isawaitable(outcome):
            await outcome

    @classmethod
    async def _emit_answer_text(cls, text: str, callback: Callable[[str], Awaitable[None] | None] | None) -> None:
        if callback is None or not text:
            return
        chunk_size = 3
        total = len(text)
        num_chunks = (total + chunk_size - 1) // chunk_size
        delay = min(0.015, 2.0 / max(num_chunks, 1))
        for index in range(0, total, chunk_size):
            chunk = text[index:index + chunk_size]
            if not chunk:
                continue
            await cls._emit_answer_delta(callback, chunk)
            if index + chunk_size < total:
                await asyncio.sleep(delay)

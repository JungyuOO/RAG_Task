from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

from app.rag.pipeline_streaming import ChatTurnDeps, ChatTurnOrchestrator
from app.rag.prompting import PromptComposer
from app.rag.retrieval_state_builder import RetrievalStateBuilder, RetrievalStateDeps
from app.rag.utils import normalize_text, stable_hash


class PipelineRuntimeMixin:
    def _build_answer_cache_key(
        self,
        session_id: str,
        rewritten_query: str,
        context_ids: list[str],
        answer_route: str,
        query_interpretation: dict | None,
        topic_id: str | None,
    ) -> str:
        del topic_id
        interpretation = query_interpretation or {}
        cache_scope = {
            "session_id": session_id,
            "rewritten_query": rewritten_query,
            "context_ids": sorted(context_ids),
            "answer_route": answer_route,
            "intent": str(interpretation.get("intent", "") or ""),
            "response_shape": str(interpretation.get("response_shape", "") or ""),
            "resources": sorted(str(value) for value in interpretation.get("resources", []) if value),
            "actions": sorted(str(value) for value in interpretation.get("actions", []) if value),
            "format_constraints": sorted(str(value) for value in interpretation.get("format_constraints", []) if value),
            "normalized_keywords": sorted(str(value) for value in interpretation.get("normalized_keywords", []) if value),
        }
        return stable_hash(json.dumps(cache_scope, ensure_ascii=False, sort_keys=True))

    def _should_run_judge_agent(self, policy_decision, query_interpretation: dict | None, top_score: float) -> bool:
        query_interpretation = query_interpretation or {}
        if policy_decision.turn_type == "document_followup":
            return False
        if query_interpretation.get("resources") and top_score >= 0.35:
            return False
        return True

    def _canonical_cache_query(self, user_message: str, rewritten_query: str, query_interpretation: dict | None) -> str:
        query_interpretation = query_interpretation or {}
        normalized = normalize_text(user_message).lower()
        normalized_keywords = [str(value).lower() for value in query_interpretation.get("normalized_keywords", []) if value]
        has_referential_marker = any(marker in normalized for marker in ("洹멸굅", "洹멸굔", "洹몄?", "洹?yaml", "洹?肄붾뱶", "?ㅼ떆", "洹몃읆", "that", "this", "again"))
        if query_interpretation.get("resources") and normalized_keywords and not has_referential_marker:
            return " ".join(sorted(dict.fromkeys(normalized_keywords)))
        return rewritten_query

    @staticmethod
    def _topic_to_topic_state(topic: dict | None) -> dict:
        return PromptComposer.topic_to_topic_state(topic)

    def _build_rewrite_context_from_topic(self, topic: dict | None, topic_turns: list) -> dict | None:
        return self._get_prompt_composer().build_rewrite_context_from_topic(topic, topic_turns)

    def _ensure_topic_for_resolution(self, session_id: str, user_message: str, state: dict, user_turn_id: int | None) -> str | None:
        resolution = state.get("turn_resolution") or {}
        resolution_type = resolution.get("resolution_type", "")
        if resolution_type == "ambiguous" or user_turn_id is None:
            return None
        topic_id = state.get("resolved_topic_id")
        if not topic_id and resolution_type == "new_topic":
            created_topic = self.session_repository.create_topic(session_id, seed_label=user_message, seed_turn_id=user_turn_id)
            topic_id = created_topic["topic_id"]
            state["resolved_topic_id"] = topic_id
        if topic_id:
            self.session_repository.link_turn_to_topic(
                user_turn_id,
                session_id,
                topic_id,
                "user",
                resolution_type or "continue",
                float(resolution.get("confidence", 0.0) or 0.0),
            )
        return topic_id

    def _store_assistant_turn(self, session_id: str, content: str, metadata: dict, topic_id: str | None) -> int:
        enriched_metadata = dict(metadata or {})
        stored_procedure_state = enriched_metadata.pop("_stored_procedure_state", {})
        if stored_procedure_state and not enriched_metadata.get("procedure_state"):
            enriched_metadata["procedure_state"] = stored_procedure_state
        enriched_metadata.setdefault("procedure_state", {})
        turn_id = self.session_repository.add_turn(session_id, "assistant", content, metadata=enriched_metadata)
        if topic_id:
            self.session_repository.link_turn_to_topic(turn_id, session_id, topic_id, "assistant", "continue", 1.0)
        return turn_id

    async def _rewrite_query_with_llm(self, session_id: str, user_message: str, rewrite_context: dict | None = None) -> str:
        rewrite_context = rewrite_context or self.session_repository.build_rewrite_context(session_id, user_message)
        if rewrite_context is None:
            return user_message.strip()

        history_lines: list[str] = []
        for turn in rewrite_context["conversation_history"]:
            role_label = "사용자" if turn["role"] == "user" else "어시스턴트"
            line = f"- {role_label}: {turn['content']}"
            if turn.get("sources"):
                line += f" (출처: {', '.join(turn['sources'])})"
            history_lines.append(line)

        context_parts: list[str] = []
        if rewrite_context["active_topic"]:
            context_parts.append(f"현재 토픽: {rewrite_context['active_topic']}")
        if rewrite_context["active_entities"]:
            context_parts.append(f"활성 엔티티: {', '.join(rewrite_context['active_entities'])}")
        if rewrite_context["selected_sources"]:
            context_parts.append(f"참조 문서: {', '.join(rewrite_context['selected_sources'])}")

        prompt = (
            "당신은 RAG 검색 시스템의 질의 재작성기입니다.\n"
            "아래 대화 이력과 현재 질문을 보고, 검색에 적합한 독립적인 질의로 재작성하세요.\n\n"
            "규칙:\n"
            "1. 대명사(그거, 이거, it, that 등)나 생략된 주어가 있으면 대화 맥락에서 구체적 용어로 대체하세요.\n"
            "2. 이미 독립적이고 구체적인 질문이면 그대로 반환하세요.\n"
            "3. 완전히 새로운 토픽의 질문이면 그대로 반환하세요.\n"
            "4. 재작성된 질의만 출력하세요. 설명이나 부연은 불필요합니다.\n\n"
            f"대화 맥락:\n" + "\n".join(context_parts) + "\n\n"
            f"최근 대화:\n" + "\n".join(history_lines) + "\n\n"
            f"현재 질문: {user_message.strip()}\n\n"
            "재작성된 검색 질의:"
        )
        messages = [{"role": "user", "content": prompt}]
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                rewritten = await self.llm.generate(messages, max_tokens=200)
                rewritten = rewritten.strip().strip('"').strip("'")
                if not self._is_valid_rewritten_query(user_message, rewritten):
                    if attempt < max_retries:
                        continue
                    return user_message.strip()
                return rewritten
            except Exception:
                if attempt < max_retries:
                    continue
                return user_message.strip()
        return user_message.strip()

    def _is_valid_rewritten_query(self, user_message: str, rewritten: str) -> bool:
        if not rewritten:
            return False
        if len(rewritten) > max(len(user_message) * 3, 120):
            return False
        lowered = rewritten.casefold()
        invalid_markers = (
            "analyze the input data",
            "current topic:",
            "active entities:",
            "rules:",
            "recent conversation",
            "현재 토픽:",
            "활성 엔티티:",
            "최근 대화",
        )
        if any(marker in lowered for marker in invalid_markers):
            return False
        if "\n" in rewritten and len(rewritten.splitlines()) > 2:
            return False
        return True

    async def _prepare_retrieval_state(self, session_id: str, user_message: str, allowed_source_paths: set[str] | None = None, *, version_tag: str | None = None) -> dict:
        builder = RetrievalStateBuilder(self._build_retrieval_state_deps())
        return await builder.run(session_id, user_message, allowed_source_paths, version_tag=version_tag)

    async def inspect_retrieval(self, session_id: str, user_message: str, allowed_source_paths: set[str] | None = None) -> dict:
        state = await self._prepare_retrieval_state(session_id, user_message, allowed_source_paths)
        return self.answer_service.build_context_payload(
            state["rewritten_query"],
            state["response_mode"],
            state["top_score"],
            state["preferred_preview_source"],
            state["preview_pages"],
            state["selected_context_items"],
            state["grounded_pages"],
            [],
            preview_finalized=False,
        )

    def _interleave_context_items_by_source(self, items: list[dict]) -> list[dict]:
        source_groups: dict[str, list[dict]] = {}
        for item in items:
            src = item["chunk"]["source_path"]
            source_groups.setdefault(src, []).append(item)
        interleaved: list[dict] = []
        max_len = max((len(group) for group in source_groups.values()), default=0)
        for index in range(max_len):
            for src in list(source_groups.keys()):
                group = source_groups[src]
                if index < len(group):
                    interleaved.append(group[index])
        return interleaved

    def _build_context_blocks(self, context_items: list[dict]) -> tuple[list[str], list[str]]:
        blocks: list[str] = []
        chunk_ids: list[str] = []
        for item in context_items:
            chunk = item["chunk"]
            chunk_ids.append(chunk["chunk_id"])
            citation = f"{Path(chunk['source_path']).name}"
            page_start = chunk["metadata"].get("page_start")
            page_end = chunk["metadata"].get("page_end")
            if page_start and page_end:
                citation += f" p.{page_start}" if page_start == page_end else f" p.{page_start}-{page_end}"
            elif chunk["page_number"]:
                citation += f" p.{chunk['page_number']}"
            blocks.append(f"[{citation}]\n{chunk['text']}")
        return blocks, chunk_ids

    def _finalize_answer(self, **kwargs) -> tuple[str, list[dict], dict]:
        return self.answer_service.finalize_answer(
            **kwargs,
            retrieval_min_score=self.settings.retrieval_min_score,
        )

    def _get_available_versions(self) -> list[str]:
        """data/corpus/pdfs/ 하위 ocp-X.Y 폴더에서 버전 목록 추출."""
        source_dir = self.settings.rag_source_dir
        versions = []
        if source_dir.exists():
            for folder in sorted(source_dir.iterdir()):
                if folder.is_dir() and folder.name.startswith("ocp-"):
                    version = folder.name.removeprefix("ocp-")
                    if version:
                        versions.append(version)
        return versions or ["4.15", "4.16", "4.17", "4.18", "4.19", "4.20", "4.21"]

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        append_user_turn: bool = True,
        version_tag: str | None = None,
    ) -> AsyncIterator[dict]:
        available_versions = self._get_available_versions()
        orchestrator = ChatTurnOrchestrator(self._build_chat_turn_deps())
        async for event in orchestrator.run(
            session_id=session_id,
            user_message=user_message,
            allowed_source_paths=allowed_source_paths,
            append_user_turn=append_user_turn,
            version_tag=version_tag,
            available_versions=available_versions,
        ):
            yield event

    def _build_retrieval_state_deps(self) -> RetrievalStateDeps:
        return RetrievalStateDeps(
            resolve_turn_context=self._resolve_turn_context,
            build_non_retrieval_state=self._build_non_retrieval_state,
            build_rewrite_context_from_topic=self._build_rewrite_context_from_topic,
            rewrite_query_with_llm=self._rewrite_query_with_llm,
            index_repository=self.index_repository,
            intent_agent=self.intent_agent,
            retrieval_agent=self.retrieval_agent,
            expand_query_with_resource_aliases=self._expand_query_with_resource_aliases,
            expand_query_with_context=self._expand_query_with_context,
            embedder=self.embedder,
            retrieval_service=self.retrieval_service,
            retriever=self.retriever,
            reranker=self.reranker,
            metadata_aware_rerank=self._metadata_aware_rerank,
            expand_local_context_items=self._expand_local_context_items,
            expand_topic_anchor_context_items=self._expand_topic_anchor_context_items,
            should_use_retrieved_context=self._should_use_retrieved_context,
            apply_precision_filter=self._apply_precision_filter,
            apply_focus_filter=self._apply_focus_filter,
            find_fallback_code_context_items=self._find_fallback_code_context_items,
            settings=self.settings,
        )

    def _build_chat_turn_deps(self) -> ChatTurnDeps:
        return ChatTurnDeps(
            detect_non_korean_query=self._detect_non_korean_query,
            session_repository=self.session_repository,
            should_skip_procedure_shortcut=self._should_skip_procedure_shortcut,
            detect_procedure_followup=self._detect_procedure_followup,
            build_procedure_followup_answer=self._build_procedure_followup_answer,
            looks_like_step_navigation_without_state=self._looks_like_step_navigation_without_state,
            resolve_turn_context=self._resolve_turn_context,
            domain_guard_state=self._domain_guard_state,
            prepare_retrieval_state=self._prepare_retrieval_state,
            resolve_answer_route=self._resolve_answer_route,
            interleave_context_items_by_source=self._interleave_context_items_by_source,
            build_context_blocks=self._build_context_blocks,
            ensure_topic_for_resolution=self._ensure_topic_for_resolution,
            build_answer_cache_key=self._build_answer_cache_key,
            canonical_cache_query=self._canonical_cache_query,
            should_run_judge_agent=self._should_run_judge_agent,
            build_policy_answer=self._build_policy_answer,
            build_missing_extractive_answer=self._build_missing_extractive_answer,
            select_code_example_context_items=self._select_code_example_context_items,
            resolve_requested_resource_kinds=self._resolve_requested_resource_kinds,
            prefer_block_type_items=self._prefer_block_type_items,
            finalize_answer=self._finalize_answer,
            store_assistant_turn=self._store_assistant_turn,
            build_llm_failure_fallback=self._build_llm_failure_fallback,
            get_prompt_composer=self._get_prompt_composer,
            answer_service=self.answer_service,
            answer_cache_repository=self.answer_cache_repository,
            judge_agent=self.judge_agent,
            llm=self.llm,
        )

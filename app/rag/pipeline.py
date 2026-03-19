from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import Settings
from app.rag.cache import JsonFileCache
from app.rag.chunking import StructuredMarkdownChunker, TextChunker
from app.rag.embeddings import HashingEmbedder
from app.rag.index import VectorIndex
from app.rag.ingestion import DocumentIngestor
from app.rag.llm import LlmClient
from app.rag.memory import SessionStore
from app.rag.retrieval import HybridRetriever
from app.rag.utils import stable_hash
from app.repositories.cache_repository import CacheRepository
from app.repositories.index_repository import IndexRepository
from app.repositories.session_repository import SessionRepository
from app.services.answer_service import AnswerService
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService
from app.services.turn_policy_service import TurnPolicyDecision, TurnPolicyService


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ingestor = DocumentIngestor(settings)
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
        self.structured_chunker = StructuredMarkdownChunker(
            settings.structured_chunk_size,
            settings.structured_chunk_overlap,
        )
        self.embedder = HashingEmbedder(settings.vector_dim)
        self.index = VectorIndex(settings.rag_index_dir)
        self.retriever = HybridRetriever(settings.retrieval_top_k, settings.candidate_pool_size)
        self.embedding_cache = JsonFileCache(settings.rag_cache_dir / "embeddings")
        self.answer_cache = JsonFileCache(settings.rag_cache_dir / "answers")
        self.session_store = SessionStore(
            db_path=settings.rag_index_dir / "session_store.sqlite3",
            memory_window_turns=settings.memory_window_turns,
        )
        self.llm = LlmClient(settings)
        self.index_repository = IndexRepository(self.index)
        self.embedding_cache_repository = CacheRepository(self.embedding_cache)
        self.answer_cache_repository = CacheRepository(self.answer_cache)
        self.session_repository = SessionRepository(self.session_store)

        self.indexing_service = IndexingService(
            settings=settings,
            ingestor=self.ingestor,
            chunker=self.chunker,
            structured_chunker=self.structured_chunker,
            embedder=self.embedder,
            index_repository=self.index_repository,
            embedding_cache_repository=self.embedding_cache_repository,
        )
        self.retrieval_service = RetrievalService(settings)
        self.answer_service = AnswerService(self.retrieval_service)
        self.turn_policy_service = TurnPolicyService()

    def _build_llm_failure_fallback(
        self,
        user_message: str,
        use_retrieved_context: bool,
        context_blocks: list[str],
        context_text: str,
        policy: TurnPolicyDecision,
    ) -> str:
        if use_retrieved_context and context_blocks:
            return "LLM 호출에 실패했습니다. 현재는 검색된 문맥만 보여드릴게요.\n\n" + context_text[:1200]
        if policy.turn_type == "greeting":
            return "안녕하세요! 무엇을 도와드릴까요?"
        if policy.needs_clarification and policy.clarification_prompt:
            return policy.clarification_prompt
        if policy.response_mode == "conversational":
            return "네, 필요하시면 이어서 관련 내용을 더 설명드릴게요."
        return "현재 LLM 연결이 불안정해 일반 답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."

    def _is_code_example_request(self, user_message: str) -> bool:
        normalized = (user_message or "").lower()
        markers = (
            "yaml",
            "manifest",
            "code",
            "example",
            "sample",
            "demo",
            "cli",
            "kubectl",
            "oc ",
            "oc\n",
            "pv",
            "pvc",
            "\ucf54\ub4dc",
            "\uc608\uc2dc",
            "\ub370\ubaa8",
            "\ub9e4\ub2c8\ud398\uc2a4\ud2b8",
            "\uc124\uc815",
            "\uc801\uc5b4\uc918",
            "\ubcf4\uc5ec\uc918",
        )
        return any(marker in normalized for marker in markers)

    def _build_prompt_memory_snapshot(self, session_id: str) -> dict:
        snapshot = self.session_repository.memory_snapshot(session_id)
        summary = snapshot.get("session_summary", {})
        topic_state = snapshot.get("topic_state", {})
        recent_turns = snapshot.get("recent_turns", [])
        prompt_recent_turns = max(int(self.settings.llm_prompt_recent_turns), 1)
        compact_recent_turns = [
            {
                "role": turn.get("role", ""),
                "content": str(turn.get("content", ""))[:180],
            }
            for turn in recent_turns[-prompt_recent_turns:]
        ]
        return {
            "topic": summary.get("topic", ""),
            "user_goal": str(summary.get("user_goal", ""))[:180],
            "recent_documents": summary.get("recent_documents", [])[:3],
            "recent_pages": summary.get("recent_pages", [])[:4],
            "active_topic": topic_state.get("active_topic", ""),
            "selected_sources": topic_state.get("selected_sources", [])[:3],
            "selected_pages": topic_state.get("selected_pages", [])[:4],
            "last_retrieval_mode": topic_state.get("last_retrieval_mode", ""),
            "recent_turns": compact_recent_turns,
        }

    def _build_prompt_recent_turns(self, session_id: str) -> list[dict]:
        prompt_recent_turns = max(int(self.settings.llm_prompt_recent_turns), 1)
        recent_turns = self.session_repository.recent_turns(session_id)[-prompt_recent_turns:]
        return [
            {
                "role": turn.role,
                "content": str(turn.content)[:500],
            }
            for turn in recent_turns
        ]

    def _build_prompt_context_text(self, context_blocks: list[str]) -> str:
        max_items = max(int(self.settings.llm_prompt_context_items), 1)
        char_limit = max(int(self.settings.llm_prompt_context_char_limit), 600)
        selected_blocks = context_blocks[:max_items]
        parts: list[str] = []
        used = 0
        for block in selected_blocks:
            remaining = char_limit - used
            if remaining <= 0:
                break
            trimmed = block[:remaining]
            parts.append(trimmed)
            used += len(trimmed)
        return "\n\n".join(parts) if parts else "No reliable retrieved context."

    def rebuild_index(self, source_paths: list[Path]) -> dict:
        return self.indexing_service.rebuild_index(source_paths)

    def _chunk_documents(self, documents: list) -> list:
        return self.indexing_service.chunk_documents(documents)

    def _load_extracted_markdown(self, source_path: Path) -> str | None:
        return self.indexing_service.load_extracted_markdown(source_path)

    def _select_chunking_strategy(self, markdown_text: str | None, documents: list) -> str:
        return self.indexing_service.select_chunking_strategy(markdown_text, documents)

    def _build_context_items_payload(self, context_items: list[dict]) -> list[dict]:
        return self.retrieval_service.build_context_items_payload(context_items)

    def _aggregate_page_grounding(self, context_items: list[dict]) -> list[dict]:
        return self.retrieval_service.aggregate_page_grounding(context_items)

    def _aggregate_source_grounding(self, grounded_pages: list[dict]) -> list[dict]:
        return self.retrieval_service.aggregate_source_grounding(grounded_pages)

    def _select_grounded_preview_source(self, grounded_pages: list[dict]) -> str | None:
        return self.retrieval_service.select_grounded_preview_source(grounded_pages)

    def _build_grounded_preview_pages(
        self,
        preferred_preview_source: str | None,
        grounded_pages: list[dict],
        limit: int = 3,
    ) -> list[dict]:
        return self.retrieval_service.build_grounded_preview_pages(
            preferred_preview_source,
            grounded_pages,
            limit=limit,
        )

    def _chunk_page_numbers(self, item: dict) -> list[int]:
        return self.retrieval_service.chunk_page_numbers(item)

    def _order_context_items_by_grounded_pages(
        self,
        context_items: list[dict],
        grounded_pages: list[dict],
    ) -> list[dict]:
        return self.retrieval_service.order_context_items_by_grounded_pages(context_items, grounded_pages)

    def _select_context_items_by_grounded_pages(
        self,
        ordered_context_items: list[dict],
        grounded_pages: list[dict],
    ) -> list[dict]:
        return self.retrieval_service.select_context_items_by_grounded_pages(
            ordered_context_items,
            grounded_pages,
        )

    def _build_answer_citation_payload(
        self,
        answer: str,
        context_items: list[dict],
        grounded_pages: list[dict],
        preferred_preview_source: str | None,
    ) -> list[dict]:
        return self.answer_service.build_answer_citation_payload(
            answer,
            context_items,
            grounded_pages,
            preferred_preview_source,
        )

    def _sanitize_answer(
        self,
        answer: str,
        use_retrieved_context: bool,
    ) -> str:
        return self.answer_service.sanitize_answer(answer, use_retrieved_context)

    def _extract_answer_citations(self, answer: str) -> list[tuple[str, int, int]]:
        return self.answer_service.extract_answer_citations(answer)

    def _build_answer_aligned_preview_pages(
        self,
        answer_citations: list[dict],
        context_items: list[dict],
        preferred_preview_source: str | None,
        grounded_pages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        return self.answer_service.build_answer_aligned_preview_pages(
            answer_citations,
            context_items,
            preferred_preview_source,
            grounded_pages,
        )

    def _build_source_line(self, answer_citations: list[dict]) -> str:
        return self.answer_service.build_source_line(answer_citations)

    def _ensure_answer_source_line(
        self,
        answer: str,
        answer_citations: list[dict],
        use_retrieved_context: bool,
    ) -> str:
        return self.answer_service.ensure_answer_source_line(
            answer,
            answer_citations,
            use_retrieved_context,
        )

    def _build_context_payload(
        self,
        rewritten_query: str,
        use_retrieved_context: bool,
        top_score: float,
        preferred_preview_source: str | None,
        preview_pages: list[dict],
        context_items: list[dict],
        grounded_pages: list[dict],
        answer_citations: list[dict],
        response_mode: str,
        preview_finalized: bool = False,
    ) -> dict:
        return self.answer_service.build_context_payload(
            rewritten_query,
            response_mode,
            top_score,
            preferred_preview_source,
            preview_pages,
            context_items,
            grounded_pages,
            answer_citations,
            preview_finalized=preview_finalized,
        )

    def _build_extractive_code_answer(self, context_items: list[dict]) -> str | None:
        return self.answer_service.build_extractive_code_answer(context_items)

    def _filter_index_items(
        self,
        index_items: list[dict],
        allowed_source_paths: set[str] | None,
    ) -> list[dict]:
        return self.retrieval_service.filter_index_items(index_items, allowed_source_paths)

    def _prepare_retrieval_state(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
    ) -> dict:
        recent_turns = self.session_repository.recent_turns(session_id)
        structured_summary = self.session_repository.structured_summary(session_id)
        topic_state = self.session_repository.topic_state(session_id)
        policy = self.turn_policy_service.classify_turn(
            user_message,
            recent_turns,
            structured_summary,
            topic_state,
        )
        if not policy.use_retrieval:
            return {
                "rewritten_query": user_message.strip(),
                "top_score": 0.0,
                "use_retrieved_context": False,
                "grounded_pages": [],
                "selected_context_items": [],
                "preferred_preview_source": None,
                "preview_pages": [],
                "response_mode": policy.response_mode,
                "turn_policy": policy.to_dict(),
            }
        rewritten_query = (
            self.session_repository.rewrite_query(session_id, user_message)
            if policy.use_memory_rewrite
            else user_message.strip()
        )
        query_vector = self.embedder.encode(rewritten_query)
        index_items = self._filter_index_items(self.index_repository.load(), allowed_source_paths)
        retrieved = self.retriever.search(rewritten_query, query_vector, index_items)
        top_score = retrieved[0]["rerank_score"] if retrieved else 0.0
        use_retrieved_context = top_score >= self.settings.retrieval_min_score
        context_items = retrieved if use_retrieved_context else []
        grounded_pages = self._aggregate_page_grounding(context_items)
        ordered_context_items = self._order_context_items_by_grounded_pages(context_items, grounded_pages)
        selected_context_items = self._select_context_items_by_grounded_pages(ordered_context_items, grounded_pages)
        preferred_preview_source = self._select_grounded_preview_source(grounded_pages)
        preview_pages = self._build_grounded_preview_pages(preferred_preview_source, grounded_pages)
        return {
            "rewritten_query": rewritten_query,
            "top_score": top_score,
            "use_retrieved_context": use_retrieved_context,
            "grounded_pages": grounded_pages,
            "selected_context_items": selected_context_items,
            "preferred_preview_source": preferred_preview_source,
            "preview_pages": preview_pages,
            "response_mode": "rag" if use_retrieved_context else "general",
            "turn_policy": policy.to_dict(),
        }

    def inspect_retrieval(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
    ) -> dict:
        state = self._prepare_retrieval_state(session_id, user_message, allowed_source_paths)
        return self._build_context_payload(
            state["rewritten_query"],
            state["use_retrieved_context"],
            state["top_score"],
            state["preferred_preview_source"],
            state["preview_pages"],
            state["selected_context_items"],
            state["grounded_pages"],
            [],
            state["response_mode"],
            preview_finalized=False,
        )

    def list_library_documents(self) -> dict:
        return self.indexing_service.list_library_documents()

    def delete_library_document(self, source_path: Path) -> dict:
        return self.indexing_service.delete_library_document(source_path)

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        append_user_turn: bool = True,
    ) -> AsyncIterator[dict]:
        state = self._prepare_retrieval_state(session_id, user_message, allowed_source_paths)
        rewritten_query = state["rewritten_query"]
        top_score = state["top_score"]
        use_retrieved_context = state["use_retrieved_context"]
        grounded_pages = state["grounded_pages"]
        selected_context_items = state["selected_context_items"]
        preferred_preview_source = state["preferred_preview_source"]
        preview_pages = state["preview_pages"]
        response_mode = state.get("response_mode", "rag" if use_retrieved_context else "general")
        turn_policy = state.get("turn_policy", {})
        policy_decision = TurnPolicyDecision(**turn_policy) if turn_policy else TurnPolicyDecision(
            turn_type=response_mode,
            response_mode=response_mode,
            use_retrieval=use_retrieved_context,
            use_memory_rewrite=False,
            allow_preview=use_retrieved_context,
            allow_citations=use_retrieved_context,
        )
        code_example_request = self._is_code_example_request(user_message)

        context_blocks = []
        context_ids = []
        for item in selected_context_items:
            chunk = item["chunk"]
            context_ids.append(chunk["chunk_id"])
            citation = f"{Path(chunk['source_path']).name}"
            page_start = chunk["metadata"].get("page_start")
            page_end = chunk["metadata"].get("page_end")
            if page_start and page_end:
                citation += f" p.{page_start}" if page_start == page_end else f" p.{page_start}-{page_end}"
            elif chunk["page_number"]:
                citation += f" p.{chunk['page_number']}"
            context_blocks.append(f"[{citation}]\n{chunk['text']}")

        memory_snapshot = self.session_repository.memory_snapshot(session_id)
        cache_key = stable_hash(
            f"{session_id}:{rewritten_query}:{'|'.join(context_ids)}:"
            f"{json.dumps(memory_snapshot, ensure_ascii=False, sort_keys=True)}"
        )
        cached_answer = self.answer_cache_repository.get(cache_key)

        if append_user_turn:
            self.session_repository.add_turn(session_id, "user", user_message)
        if not policy_decision.allow_preview:
            preview_pages = []
            preferred_preview_source = None
        context_payload = self._build_context_payload(
            rewritten_query,
            use_retrieved_context,
            top_score,
            preferred_preview_source,
            preview_pages,
            selected_context_items,
            grounded_pages,
            [],
            response_mode,
            preview_finalized=False,
        )
        yield {"type": "context", **context_payload}

        if policy_decision.needs_clarification and policy_decision.clarification_prompt:
            clarification_answer = policy_decision.clarification_prompt.strip()
            yield {"type": "token", "content": clarification_answer, "cached": False}
            final_payload = self._build_context_payload(
                rewritten_query,
                use_retrieved_context,
                top_score,
                None,
                [],
                [],
                [],
                [],
                response_mode,
                preview_finalized=True,
            )
            yield {"type": "context", **final_payload}
            self.session_repository.add_turn(session_id, "assistant", clarification_answer, metadata=final_payload)
            self.answer_cache_repository.set(cache_key, {"answer": clarification_answer})
            yield {"type": "done", "cached": False}
            return

        if code_example_request and use_retrieved_context:
            extractive_answer = self._build_extractive_code_answer(selected_context_items)
            if extractive_answer:
                streamed = ""
                for token in extractive_answer.split(" "):
                    chunk = token + " "
                    streamed += chunk
                    yield {"type": "token", "content": chunk, "cached": False}
                answer = self._sanitize_answer(streamed.strip(), use_retrieved_context)
                answer_citations = (
                    self._build_answer_citation_payload(
                        answer,
                        selected_context_items,
                        grounded_pages,
                        preferred_preview_source,
                    )
                    if policy_decision.allow_citations
                    else []
                )
                final_answer = self._ensure_answer_source_line(answer, answer_citations, use_retrieved_context)
                if final_answer != answer:
                    suffix = final_answer[len(answer):]
                    if suffix:
                        yield {"type": "token", "content": suffix, "cached": False}
                if policy_decision.allow_preview:
                    final_source, final_preview_pages = self._build_answer_aligned_preview_pages(
                        answer_citations,
                        selected_context_items,
                        preferred_preview_source,
                        grounded_pages,
                    )
                else:
                    final_source, final_preview_pages = None, []
                final_payload = self._build_context_payload(
                    rewritten_query,
                    use_retrieved_context,
                    top_score,
                    final_source,
                    final_preview_pages,
                    selected_context_items,
                    grounded_pages,
                    answer_citations,
                    response_mode,
                    preview_finalized=True,
                )
                yield {"type": "context", **final_payload}
                self.session_repository.add_turn(session_id, "assistant", final_answer, metadata=final_payload)
                self.answer_cache_repository.set(cache_key, {"answer": final_answer})
                yield {"type": "done", "cached": False}
                return

        if cached_answer is not None:
            full_text = self._sanitize_answer(cached_answer["answer"], use_retrieved_context)
            streamed = ""
            for token in full_text.split(" "):
                chunk = token + " "
                streamed += chunk
                yield {"type": "token", "content": chunk, "cached": True}
            answer_citations = (
                self._build_answer_citation_payload(
                    streamed.strip(),
                    selected_context_items,
                    grounded_pages,
                    preferred_preview_source,
                )
                if policy_decision.allow_citations
                else []
            )
            final_answer = self._ensure_answer_source_line(streamed.strip(), answer_citations, use_retrieved_context)
            if final_answer != streamed.strip():
                suffix = final_answer[len(streamed.strip()):]
                if suffix:
                    yield {"type": "token", "content": suffix, "cached": True}
            if policy_decision.allow_preview:
                final_source, final_preview_pages = self._build_answer_aligned_preview_pages(
                    answer_citations,
                    selected_context_items,
                    preferred_preview_source,
                    grounded_pages,
                )
            else:
                final_source, final_preview_pages = None, []
            final_payload = self._build_context_payload(
                rewritten_query,
                use_retrieved_context,
                top_score,
                final_source,
                final_preview_pages,
                selected_context_items,
                grounded_pages,
                answer_citations,
                response_mode,
                preview_finalized=True,
            )
            yield {"type": "context", **final_payload}
            self.session_repository.add_turn(session_id, "assistant", final_answer, metadata=final_payload)
            yield {"type": "done", "cached": True}
            return

        system_prompt = (
            "You are a RAG assistant. "
            "If retrieved context is provided, answer from that context first and mention source file names and page numbers when possible. "
            "If retrieved context is weak or missing, answer as a general assistant. "
            "If the user is simply reacting, acknowledging, or thanking you after a document-grounded answer, respond conversationally without reusing document citations. "
            "Do not mention unrelated prior questions or prior document topics unless the current user message explicitly asks for them. "
            "When retrieval mode is general, do not mention source files, page numbers, uploaded library notes, or citations. "
            "When retrieved context is used, end the answer with a short source line such as '[file.pdf] p.5' or '[file.pdf] p.5-6'. "
            "Keep answers concise but grounded."
        )
        if code_example_request:
            system_prompt += (
                " If the retrieved context contains YAML, commands, manifests, or configuration examples, "
                "preserve resource names, field names, values, and command syntax exactly as written in the source whenever possible. "
                "Do not invent alternate example names if a concrete example already exists in the retrieved context."
            )
        summary = self.session_repository.summary(session_id)
        prompt_memory = self._build_prompt_memory_snapshot(session_id)
        recent_turns = self._build_prompt_recent_turns(session_id)
        context_text = self._build_prompt_context_text(context_blocks)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    f"Conversation summary:\n{summary or 'No summary yet.'}\n\n"
                    f"Session memory:\n{json.dumps(prompt_memory, ensure_ascii=False)}\n\n"
                    f"Retrieval mode: {response_mode}\n"
                    f"Turn policy: {json.dumps(turn_policy, ensure_ascii=False)}\n"
                    f"Top retrieval score: {top_score:.4f}\n\n"
                    f"Retrieved context:\n{context_text}"
                ),
            },
            *recent_turns,
            {"role": "user", "content": user_message},
        ]

        parts: list[str] = []
        try:
            async for token in self.llm.stream_chat(messages):
                parts.append(token)
                yield {"type": "token", "content": token, "cached": False}
        except Exception as exc:  # pragma: no cover
            fallback = self._build_llm_failure_fallback(
                user_message,
                use_retrieved_context,
                context_blocks,
                context_text,
                policy_decision,
            )
            parts = [fallback]
            yield {"type": "token", "content": fallback, "cached": False, "error": str(exc)}

        answer = self._sanitize_answer("".join(parts).strip(), use_retrieved_context)
        answer_citations = (
            self._build_answer_citation_payload(
                answer,
                selected_context_items,
                grounded_pages,
                preferred_preview_source,
            )
            if policy_decision.allow_citations
            else []
        )
        final_answer = self._ensure_answer_source_line(answer, answer_citations, use_retrieved_context)
        if final_answer != answer:
            suffix = final_answer[len(answer):]
            if suffix:
                yield {"type": "token", "content": suffix, "cached": False}
        if policy_decision.allow_preview:
            final_source, final_preview_pages = self._build_answer_aligned_preview_pages(
                answer_citations,
                selected_context_items,
                preferred_preview_source,
                grounded_pages,
            )
        else:
            final_source, final_preview_pages = None, []
        final_payload = self._build_context_payload(
            rewritten_query,
            use_retrieved_context,
            top_score,
            final_source,
            final_preview_pages,
            selected_context_items,
            grounded_pages,
            answer_citations,
            response_mode,
            preview_finalized=True,
        )
        yield {"type": "context", **final_payload}
        self.session_repository.add_turn(session_id, "assistant", final_answer, metadata=final_payload)
        self.answer_cache_repository.set(cache_key, {"answer": final_answer})
        yield {"type": "done", "cached": False}


from __future__ import annotations

from collections.abc import AsyncIterator
from collections import defaultdict
from pathlib import Path
import re

from app.config import Settings
from app.rag.cache import JsonFileCache
from app.rag.artifacts import extracted_markdown_path
from app.rag.chunking import StructuredMarkdownChunker, TextChunker
from app.rag.embeddings import HashingEmbedder
from app.rag.index import VectorIndex
from app.rag.ingestion import DocumentIngestor
from app.rag.llm import LlmClient
from app.rag.memory import SessionStore
from app.rag.retrieval import HybridRetriever
from app.rag.utils import stable_hash


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

    def rebuild_index(self, source_paths: list[Path]) -> dict:
        documents, skipped = self.ingestor.ingest_paths(source_paths)
        chunks = self._chunk_documents(documents)

        vectors: list[list[float]] = []
        for chunk in chunks:
            cache_key = stable_hash(chunk.text)
            cached = self.embedding_cache.get(cache_key)
            if cached is None:
                vector = self.embedder.encode(chunk.text)
                self.embedding_cache.set(cache_key, {"vector": vector})
            else:
                vector = cached["vector"]
            vectors.append(vector)

        self.index.save(chunks, vectors)
        return {
            "indexed_files": len({chunk.source_path for chunk in chunks}),
            "indexed_chunks": len(chunks),
            "skipped_files": len(skipped),
        }

    def _chunk_documents(self, documents: list) -> list:
        documents_by_path: dict[str, list] = defaultdict(list)
        for document in documents:
            documents_by_path[document.source_path].append(document)

        chunks = []
        for source_path, source_documents in documents_by_path.items():
            markdown_text = self._load_extracted_markdown(Path(source_path))
            strategy = self._select_chunking_strategy(markdown_text, source_documents)
            chunker = self.structured_chunker if strategy == "structured_markdown" else self.chunker
            source_chunks = chunker.split(source_documents, markdown_text=markdown_text)
            for chunk in source_chunks:
                chunk.metadata["chunking_strategy"] = strategy
            chunks.extend(source_chunks)
        return chunks

    def _load_extracted_markdown(self, source_path: Path) -> str | None:
        markdown_path = extracted_markdown_path(self.settings.rag_extract_dir, source_path)
        if not markdown_path.exists():
            return None
        return markdown_path.read_text(encoding="utf-8")

    def _select_chunking_strategy(self, markdown_text: str | None, documents: list) -> str:
        strategy = (self.settings.chunking_strategy or "auto").strip().lower()
        if strategy in {"page_window", "structured_markdown"}:
            return strategy

        if not markdown_text:
            return "page_window"

        page_count = max(sum(1 for line in markdown_text.splitlines() if line.startswith("## Page ")), 1)
        raw_lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
        content_lines = [
            line
            for line in raw_lines
            if not line.startswith("# ") and not line.startswith("## Page ") and not line.startswith("- loader:")
        ]
        short_lines = [line for line in content_lines if len(line) <= 60]
        bullet_lines = [line for line in content_lines if re.match(r"^[-*]\s", line)]
        avg_chars_per_page = sum(len(document.text) for document in documents) / max(page_count, 1)
        short_line_ratio = len(short_lines) / max(len(content_lines), 1)
        bullet_ratio = len(bullet_lines) / max(len(content_lines), 1)

        if avg_chars_per_page <= 420 and (short_line_ratio >= 0.55 or bullet_ratio >= 0.2):
            return "page_window"
        return "structured_markdown"

    def _build_context_items_payload(self, context_items: list[dict]) -> list[dict]:
        return [
            {
                "source_path": item["chunk"]["source_path"],
                "page_number": item["chunk"]["page_number"] or item["chunk"]["metadata"].get("page_start"),
                "page_start": item["chunk"]["metadata"].get("page_start"),
                "page_end": item["chunk"]["metadata"].get("page_end"),
                "chunking_strategy": item["chunk"]["metadata"].get("chunking_strategy"),
                "score": round(item["rerank_score"], 4),
                "text_preview": item["chunk"]["text"][:240],
            }
            for item in context_items
        ]

    def _build_fallback_preview_pages(
        self,
        preferred_preview_source: str | None,
        context_items: list[dict],
    ) -> list[dict]:
        if not preferred_preview_source:
            return []

        preferred_items = [
            item for item in context_items if item["chunk"]["source_path"] == preferred_preview_source
        ]
        preferred_items.sort(key=lambda item: item["rerank_score"], reverse=True)
        if not preferred_items:
            return []

        top_item = preferred_items[0]
        top_chunk = top_item["chunk"]
        page_start = top_chunk["metadata"].get("page_start") or top_chunk["page_number"] or 1
        page_end = top_chunk["metadata"].get("page_end") or page_start
        if page_end < page_start:
            page_end = page_start

        preview_pages: list[dict] = []
        for page_number in range(int(page_start), int(page_end) + 1):
            preview_pages.append(
                {
                    "source_path": top_chunk["source_path"],
                    "page_number": page_number,
                    "score": round(float(top_item["rerank_score"]), 4),
                }
            )
            if len(preview_pages) >= 3:
                break
        return preview_pages

    def _extract_answer_citations(self, answer: str) -> list[tuple[str, int, int]]:
        if not answer:
            return []

        citations: list[tuple[str, int, int]] = []
        patterns = (
            r"\[([^\[\]\n]+?\.pdf)\s+p\.(\d+)(?:-(\d+))?\]",
            r"\[([^\[\]\n]+?\.pdf)\]\s*p\.(\d+)(?:-(\d+))?",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
                file_name = Path(match.group(1).strip()).name
                page_start = int(match.group(2))
                page_end = int(match.group(3) or page_start)
                if page_end < page_start:
                    page_end = page_start
                citations.append((file_name, page_start, page_end))
        return citations

    def _build_answer_aligned_preview_pages(
        self,
        answer: str,
        context_items: list[dict],
        preferred_preview_source: str | None,
    ) -> tuple[str | None, list[dict]]:
        fallback_pages = self._build_fallback_preview_pages(preferred_preview_source, context_items)
        if not context_items:
            return preferred_preview_source, fallback_pages

        items_by_source: dict[str, list[dict]] = defaultdict(list)
        items_by_name: dict[str, list[dict]] = defaultdict(list)
        for item in context_items:
            source_path = item["chunk"]["source_path"]
            source_name = Path(source_path).name
            items_by_source[source_path].append(item)
            items_by_name[source_name].append(item)

        preview_pages: list[dict] = []
        seen_pages: set[tuple[str, int]] = set()
        chosen_source: str | None = None

        for file_name, cited_start, cited_end in self._extract_answer_citations(answer):
            candidate_items = items_by_name.get(file_name, [])
            if not candidate_items:
                continue

            matching_items = []
            for item in candidate_items:
                chunk = item["chunk"]
                page_start = int(chunk["metadata"].get("page_start") or chunk["page_number"] or 1)
                page_end = int(chunk["metadata"].get("page_end") or page_start)
                if page_end < page_start:
                    page_end = page_start
                if page_start <= cited_end and page_end >= cited_start:
                    matching_items.append(item)

            if not matching_items:
                continue

            matching_items.sort(key=lambda item: item["rerank_score"], reverse=True)
            if chosen_source is None:
                chosen_source = matching_items[0]["chunk"]["source_path"]

            for item in matching_items:
                chunk = item["chunk"]
                source_path = chunk["source_path"]
                if source_path != chosen_source:
                    continue
                page_start = int(chunk["metadata"].get("page_start") or chunk["page_number"] or 1)
                page_end = int(chunk["metadata"].get("page_end") or page_start)
                if page_end < page_start:
                    page_end = page_start
                overlap_start = max(page_start, cited_start)
                overlap_end = min(page_end, cited_end)
                for page_number in range(overlap_start, overlap_end + 1):
                    page_key = (source_path, page_number)
                    if page_key in seen_pages:
                        continue
                    preview_pages.append(
                        {
                            "source_path": source_path,
                            "page_number": page_number,
                            "score": round(float(item["rerank_score"]), 4),
                        }
                    )
                    seen_pages.add(page_key)
                    if len(preview_pages) >= 3:
                        return chosen_source, preview_pages

        if preview_pages:
            return chosen_source, preview_pages
        return preferred_preview_source, fallback_pages

    def _build_context_payload(
        self,
        rewritten_query: str,
        use_retrieved_context: bool,
        top_score: float,
        preferred_preview_source: str | None,
        preview_pages: list[dict],
        context_items: list[dict],
        preview_finalized: bool = False,
    ) -> dict:
        return {
            "query": rewritten_query,
            "mode": "rag" if use_retrieved_context else "general",
            "top_score": round(top_score, 4),
            "preferred_preview_source": preferred_preview_source,
            "preview_pages": preview_pages,
            "preview_finalized": preview_finalized,
            "items": self._build_context_items_payload(context_items),
        }

    def list_library_documents(self) -> dict:
        source_files = [
            path
            for path in self.settings.rag_source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
        indexed_by_path = {document["source_path"]: document for document in self.index.list_documents()}

        indexed_documents = []
        for path in sorted(source_files):
            aggregated = indexed_by_path.get(
                str(path),
                {
                    "file_name": path.name,
                    "source_path": str(path),
                    "extension": path.suffix.lower(),
                    "indexed_pages": 0,
                    "indexed_chunks": 0,
                    "loaders": [],
                },
            )
            indexed_documents.append(
                {
                    "file_name": aggregated["file_name"],
                    "source_path": aggregated["source_path"],
                    "extension": aggregated["extension"],
                    "indexed_pages": aggregated["indexed_pages"],
                    "indexed_chunks": aggregated["indexed_chunks"],
                    "loaders": aggregated["loaders"],
                }
            )

        return {
            "source_dir": str(self.settings.rag_source_dir),
            "total_files": len(source_files),
            "indexed_documents": indexed_documents,
        }

    def delete_library_document(self, source_path: Path) -> dict:
        self.index.delete_document(str(source_path))
        library_state = self.list_library_documents()
        indexed_chunks = sum(document["indexed_chunks"] for document in library_state["indexed_documents"])
        return {
            "indexed_files": library_state["total_files"],
            "indexed_chunks": indexed_chunks,
            "skipped_files": 0,
        }

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        append_user_turn: bool = True,
    ) -> AsyncIterator[dict]:
        rewritten_query = self.session_store.rewrite_query(session_id, user_message)
        query_vector = self.embedder.encode(rewritten_query)
        index_items = self.index.load()
        if allowed_source_paths:
            normalized_allowed = set()
            for path in allowed_source_paths:
                normalized_allowed.add(str(Path(path)))
                normalized_allowed.add(str(Path(path).resolve()))
            index_items = [
                item
                for item in index_items
                if str(Path(item["chunk"]["source_path"])) in normalized_allowed
                or str(Path(item["chunk"]["source_path"]).resolve()) in normalized_allowed
            ]
        retrieved = self.retriever.search(rewritten_query, query_vector, index_items)
        top_score = retrieved[0]["rerank_score"] if retrieved else 0.0
        use_retrieved_context = top_score >= self.settings.retrieval_min_score
        context_items = retrieved if use_retrieved_context else []

        context_blocks = []
        context_ids = []
        source_scores: dict[str, float] = defaultdict(float)
        source_best_scores: dict[str, float] = defaultdict(float)
        source_match_counts: dict[str, int] = defaultdict(int)
        for item in context_items:
            chunk = item["chunk"]
            context_ids.append(chunk["chunk_id"])
            score = float(item["rerank_score"])
            source_scores[chunk["source_path"]] += score
            source_best_scores[chunk["source_path"]] = max(source_best_scores[chunk["source_path"]], score)
            source_match_counts[chunk["source_path"]] += 1
            citation = f"{Path(chunk['source_path']).name}"
            page_start = chunk["metadata"].get("page_start")
            page_end = chunk["metadata"].get("page_end")
            if page_start and page_end:
                citation += f" p.{page_start}" if page_start == page_end else f" p.{page_start}-{page_end}"
            elif chunk["page_number"]:
                citation += f" p.{chunk['page_number']}"
            context_blocks.append(f"[{citation}]\n{chunk['text']}")

        cache_key = stable_hash(
            f"{session_id}:{rewritten_query}:{'|'.join(context_ids)}:{self.session_store.summary(session_id)}"
        )
        cached_answer = self.answer_cache.get(cache_key)

        if append_user_turn:
            self.session_store.add_turn(session_id, "user", user_message)
        preferred_preview_source = (
            max(
                source_scores.keys(),
                key=lambda source: (
                    source_best_scores[source],
                    source_match_counts[source],
                    source_scores[source],
                ),
            )
            if source_scores
            else None
        )
        preview_pages = self._build_fallback_preview_pages(preferred_preview_source, context_items)
        context_payload = self._build_context_payload(
            rewritten_query,
            use_retrieved_context,
            top_score,
            preferred_preview_source,
            preview_pages,
            context_items,
            preview_finalized=False,
        )
        yield {"type": "context", **context_payload}

        if cached_answer is not None:
            full_text = cached_answer["answer"]
            streamed = ""
            for token in full_text.split(" "):
                chunk = token + " "
                streamed += chunk
                yield {"type": "token", "content": chunk, "cached": True}
            final_source, final_preview_pages = self._build_answer_aligned_preview_pages(
                streamed.strip(),
                context_items,
                preferred_preview_source,
            )
            final_payload = self._build_context_payload(
                rewritten_query,
                use_retrieved_context,
                top_score,
                final_source,
                final_preview_pages,
                context_items,
                preview_finalized=True,
            )
            yield {"type": "context", **final_payload}
            self.session_store.add_turn(session_id, "assistant", streamed.strip(), metadata=final_payload)
            yield {"type": "done", "cached": True}
            return

        system_prompt = (
            "You are a RAG assistant. "
            "If retrieved context is provided, answer from that context first and mention source file names and page numbers when possible. "
            "If retrieved context is weak or missing, answer as a general assistant but clearly state that the uploaded library may not contain supporting evidence. "
            "Do not mention unrelated prior questions or prior document topics unless the current user message explicitly asks for them. "
            "Keep answers concise but grounded."
        )
        summary = self.session_store.summary(session_id)
        recent_turns = [turn.to_dict() for turn in self.session_store.recent_turns(session_id)]
        context_text = (
            "\n\n".join(context_blocks)
            if context_blocks
            else "No reliable retrieved context. Treat the answer as general knowledge unless the user asks about the uploaded documents."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    f"Conversation summary:\n{summary or 'No summary yet.'}\n\n"
                    f"Retrieval mode: {'rag' if use_retrieved_context else 'general'}\n"
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
            fallback = "LLM 호출에 실패했습니다. 현재는 검색된 문맥만 보여드릴게요.\n\n" + context_text[:1200]
            parts = [fallback]
            yield {"type": "token", "content": fallback, "cached": False, "error": str(exc)}

        answer = "".join(parts).strip()
        final_source, final_preview_pages = self._build_answer_aligned_preview_pages(
            answer,
            context_items,
            preferred_preview_source,
        )
        final_payload = self._build_context_payload(
            rewritten_query,
            use_retrieved_context,
            top_score,
            final_source,
            final_preview_pages,
            context_items,
            preview_finalized=True,
        )
        yield {"type": "context", **final_payload}
        self.session_store.add_turn(session_id, "assistant", answer, metadata=final_payload)
        self.answer_cache.set(cache_key, {"answer": answer})
        yield {"type": "done", "cached": False}

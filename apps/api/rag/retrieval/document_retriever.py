from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from apps.api.schemas.chat import CopilotChatResponse, CopilotChatSourceItem
from apps.api.schemas.indexing import ChunkRecord, SourceDescriptor, SourceType
from apps.api.rag.query.query_features import answer_source_budget
from apps.api.rag.indexing.chunking import BlockPreservingChunker
from apps.api.rag.indexing.enrich import MetadataEnricher
from apps.api.rag.indexing.normalize import BlockNormalizer
from apps.api.rag.indexing.parsers import ParserSelector, build_default_parser_selector
from apps.api.rag.retrieval.korean_tokenizer import tokenize_with_char_ngrams


class DocumentRetriever:
    """Minimal new-path lexical retriever built on top of parser/normalize/chunk stages."""

    def __init__(
        self,
        *,
        parser_selector: ParserSelector | None = None,
        block_normalizer: BlockNormalizer | None = None,
        metadata_enricher: MetadataEnricher | None = None,
        block_chunker: BlockPreservingChunker | None = None,
        source_descriptors: list[SourceDescriptor] | None = None,
        use_char_ngram_sparse: bool = True,
    ) -> None:
        self.parser_selector = parser_selector or build_default_parser_selector()
        self.block_normalizer = block_normalizer or BlockNormalizer()
        self.metadata_enricher = metadata_enricher or MetadataEnricher()
        self.block_chunker = block_chunker or BlockPreservingChunker()
        self._source_descriptors = source_descriptors
        self._use_char_ngram_sparse = bool(use_char_ngram_sparse)
        self._cache: list[ChunkRecord] | None = None
        self._chunk_tokens: dict[str, list[str]] = {}
        self._doc_frequency: Counter[str] = Counter()
        self._avg_doc_length = 0.0

    async def answer(self, *, message: str, allowed_source_paths: list[str] | None = None) -> CopilotChatResponse | None:
        query_tokens = self._tokenize(message)
        if not query_tokens:
            return None

        chunks = self._load_chunks()
        query_profile = self._build_query_profile(message)
        normalized_allowed = self._normalize_allowed_source_paths(allowed_source_paths)

        scored: list[tuple[float, ChunkRecord]] = []
        for chunk in chunks:
            if normalized_allowed and not self._source_path_allowed(chunk.source_path, normalized_allowed):
                continue
            score = self._score_chunk(
                chunk,
                query_tokens,
                query_profile=query_profile,
                total_docs=max(len(chunks), 1),
            )
            if score <= 0:
                continue
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return None

        top = self._select_seed_chunks(scored, limit=self._seed_limit_for_query(message))
        expanded_chunks = self._expand_chunks([chunk for _score, chunk in top], chunks)
        answer = self._build_answer(message, expanded_chunks)
        return CopilotChatResponse(
            lane="doc_new",
            mode="small_to_big_keyword_retrieval",
            fallback_used=False,
            preview_ready=True,
            answer=answer,
            sources=[
                CopilotChatSourceItem(
                    source_type="doc",
                    label=self._format_source_label(chunk),
                    source_path=chunk.source_path,
                    relative_source_path=self._relative_source_path(chunk.source_path),
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_start,
                    score=round(score, 3),
                    provenance=["doc_new"],
                    metadata={
                        "section_title": chunk.section_title,
                        "section_path": chunk.section_path,
                        "html_anchor": chunk.html_anchor,
                        "source_block_type": chunk.metadata.get("source_block_type", ""),
                        "block_types": chunk.metadata.get("block_types", []),
                        "document_group": chunk.metadata.get("document_group", ""),
                        "retrieval_score": round(score, 3),
                        "retrieval_backend": "doc_new",
                        "relative_source_path": self._relative_source_path(chunk.source_path),
                        "file_name": Path(chunk.source_path).name,
                        "preview_text": self._clean_preview_text(chunk.section_title, chunk.display_text),
                        "synthesis_text": str(chunk.display_text or "")[:500],
                    },
                )
                for score, chunk in top
            ],
        )

    @staticmethod
    def _normalize_allowed_source_paths(allowed_source_paths: list[str] | None) -> set[str]:
        return {
            str(path or "").replace("\\", "/").casefold()
            for path in (allowed_source_paths or [])
            if str(path or "").strip()
        }

    @staticmethod
    def _source_path_allowed(source_path: str, normalized_allowed: set[str]) -> bool:
        normalized_source = str(source_path or "").replace("\\", "/").casefold()
        if normalized_source in normalized_allowed:
            return True
        file_name = Path(source_path).name.casefold()
        return any(Path(path).name.casefold() == file_name for path in normalized_allowed)

    def _load_chunks(self) -> list[ChunkRecord]:
        if self._cache is not None:
            return self._cache

        chunks: list[ChunkRecord] = []
        for source in self._discover_sources():
            parser = self.parser_selector.select(source)
            bundle = parser.parse(source)
            normalized = self.block_normalizer.normalize_bundle(bundle)
            enriched = self.metadata_enricher.enrich_blocks(bundle, normalized)
            chunks.extend(self.block_chunker.chunk_document(bundle, enriched))

        self._prime_stats(chunks)
        self._cache = chunks
        return self._cache

    def _discover_sources(self) -> list[SourceDescriptor]:
        if self._source_descriptors is not None:
            return list(self._source_descriptors)

        root = Path.cwd()
        descriptors: list[SourceDescriptor] = []

        for path in sorted((root / "data" / "extracted_markdown").glob("*.html")):
            descriptors.append(
                SourceDescriptor(
                    source_type=SourceType.HTML,
                    source_path=str(path),
                    file_name=path.name,
                    locale=self._extract_locale(path.name),
                )
            )

        for path in sorted((root / "data" / "corpus" / "pdfs").rglob("*.md")):
            normalized = str(path).replace("\\", "/").casefold()
            document_group = "customer_generated" if any(marker in normalized for marker in ("customer-guide", "/generated/", "/customer/")) else "official_ocp"
            descriptors.append(
                SourceDescriptor(
                    source_type=SourceType.GENERATED_MANUAL,
                    source_path=str(path),
                    file_name=path.name,
                    locale=self._extract_locale(path.name),
                    document_group=document_group,
                )
            )
        return descriptors

    @staticmethod
    def _extract_locale(name: str) -> str:
        lowered = name.casefold()
        if "-ko" in lowered or "_ko" in lowered:
            return "ko"
        if "-en" in lowered or "_en" in lowered:
            return "en"
        return ""

    @staticmethod
    def _seed_limit_for_query(message: str) -> int:
        return answer_source_budget(message)

    def _tokenize(self, message: str) -> list[str]:
        if not self._use_char_ngram_sparse:
            return [token for token in re.findall(r"[a-zA-Z0-9가-힣_-]+", str(message).casefold()) if len(token) >= 2]
        return tokenize_with_char_ngrams(message)

    def _score_chunk(
        self,
        chunk: ChunkRecord,
        query_tokens: Iterable[str],
        *,
        query_profile: dict,
        total_docs: int,
    ) -> float:
        chunk_tokens = self._chunk_tokens.get(chunk.chunk_id) or self._tokenize(chunk.retrieval_text)
        bm25_score = self._bm25(query_tokens, chunk_tokens, total_docs=total_docs)

        haystack = " ".join(
            [
                chunk.retrieval_text.casefold(),
                chunk.section_title.casefold(),
                " ".join(str(value).casefold() for value in chunk.section_path),
                " ".join(str(value).casefold() for value in chunk.resource_hints),
            ]
        )

        lexical_score = 0.0
        for token in query_tokens:
            if token in chunk.section_title.casefold():
                lexical_score += 2.5
            elif token in haystack:
                lexical_score += 1.0

        source_block_type = str(chunk.metadata.get("source_block_type") or "")
        block_boost = 0.0
        if query_profile["prefers_code_like"] and source_block_type in {"code", "list"}:
            block_boost += 1.2
        if query_profile["prefers_explanatory"] and source_block_type in {"paragraph", "heading"}:
            block_boost += 0.8

        group_boost = 0.0
        document_group = str(chunk.metadata.get("document_group") or "")
        if query_profile["prefers_customer_guides"] and document_group == "customer_generated":
            group_boost += 0.7
        if query_profile["prefers_official"] and document_group == "official_ocp":
            group_boost += 0.5

        heading_penalty = 0.85 if source_block_type == "heading" else 1.0

        return (bm25_score * 5.0 + lexical_score + block_boost + group_boost) * heading_penalty

    @staticmethod
    def _format_source_label(chunk: ChunkRecord) -> str:
        source_name = Path(chunk.source_path).name
        if chunk.section_title:
            return f"{source_name} · {chunk.section_title}"
        return source_name

    @staticmethod
    def _relative_source_path(source_path: str) -> str:
        path = Path(source_path)
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _select_seed_chunks(scored: list[tuple[float, ChunkRecord]], *, limit: int) -> list[tuple[float, ChunkRecord]]:
        selected: list[tuple[float, ChunkRecord]] = []
        seen_sections: set[str] = set()

        for item in scored:
            chunk = item[1]
            section_ref = str(chunk.metadata.get("section_ref") or chunk.chunk_id)
            if section_ref in seen_sections:
                continue
            seen_sections.add(section_ref)
            selected.append(item)
            if len(selected) >= limit:
                return selected

        for item in scored:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _build_answer(message: str, chunks: list[ChunkRecord]) -> str:
        if not chunks:
            return "새 retrieval 경로에서 관련 문서를 찾지 못했습니다."
        parts = [message.strip()]
        for chunk in chunks[:3]:
            section = chunk.section_title or Path(chunk.source_path).stem
            parts.append(f"{section}\n{DocumentRetriever._clean_preview_text(chunk.section_title, chunk.display_text)}")
        return "\n\n".join(part for part in parts if part.strip())

    def _prime_stats(self, chunks: list[ChunkRecord]) -> None:
        self._chunk_tokens = {}
        self._doc_frequency = Counter()
        lengths: list[int] = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.retrieval_text)
            self._chunk_tokens[chunk.chunk_id] = tokens
            self._doc_frequency.update(set(tokens))
            lengths.append(len(tokens))
        self._avg_doc_length = sum(lengths) / max(len(lengths), 1)

    def _bm25(self, query_tokens: Iterable[str], chunk_tokens: list[str], *, total_docs: int) -> float:
        if not chunk_tokens:
            return 0.0
        token_counter = Counter(chunk_tokens)
        score = 0.0
        avg_len = max(self._avg_doc_length, 1.0)
        doc_len = len(chunk_tokens)
        k1 = 1.2
        b = 0.75
        for token in query_tokens:
            if token not in token_counter:
                continue
            tf = token_counter[token]
            df = self._doc_frequency.get(token, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + k1 * (1.0 - b + b * doc_len / avg_len)
            score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-6))
        return score

    @staticmethod
    def _build_query_profile(message: str) -> dict:
        lowered = str(message or "").casefold()
        prefers_code_like = any(marker in lowered for marker in ("yaml", "manifest", "example", "예시", "명령", "command", "설정", "sample"))
        prefers_explanatory = any(marker in lowered for marker in ("뭐야", "무엇", "설명", "what is", "difference", "차이"))
        prefers_customer_guides = any(marker in lowered for marker in ("방법", "절차", "how to", "runbook", "guide"))
        prefers_official = prefers_explanatory or any(marker in lowered for marker in ("official", "개념", "아키텍처"))
        return {
            "prefers_code_like": prefers_code_like,
            "prefers_explanatory": prefers_explanatory,
            "prefers_customer_guides": prefers_customer_guides,
            "prefers_official": prefers_official,
        }

    @staticmethod
    def _expand_chunks(seed_chunks: list[ChunkRecord], all_chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        by_chunk_id = {chunk.chunk_id: chunk for chunk in all_chunks}
        by_section_ref: dict[str, list[ChunkRecord]] = {}
        for chunk in all_chunks:
            section_ref = str(chunk.metadata.get("section_ref") or "")
            if section_ref:
                by_section_ref.setdefault(section_ref, []).append(chunk)

        expanded: list[ChunkRecord] = []
        seen: set[str] = set()

        def add(chunk: ChunkRecord | None) -> None:
            if chunk is None or chunk.chunk_id in seen:
                return
            seen.add(chunk.chunk_id)
            expanded.append(chunk)

        for seed in seed_chunks:
            add(seed)
            section_ref = str(seed.metadata.get("section_ref") or "")
            for chunk in by_section_ref.get(section_ref, [])[:2]:
                add(chunk)
            for ref in seed.expansions:
                if ref.relation in {"previous_block", "next_block"}:
                    add(by_chunk_id.get(ref.target_id))
        return expanded

    @staticmethod
    def _clean_preview_text(section_title: str, display_text: str) -> str:
        preview = str(display_text or "").replace("\r\n", "\n").strip()
        title = str(section_title or "").strip()
        if title and preview.startswith(title):
            remainder = preview[len(title) :].lstrip(" \n:-")
            if remainder:
                preview = remainder
        preview = re.sub(r"\s+", " ", preview).strip()
        return preview[:220]



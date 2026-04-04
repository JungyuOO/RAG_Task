from __future__ import annotations

import logging
import re

from app.config import Settings
from app.llm import AnswerAgent, IntentAgent, JudgeAgent, RetrievalAgent
from app.rag.answer import AnswerGenerator
from app.rag.bge_embeddings import BGEOllamaEmbedder
from app.rag.cache import JsonFileCache
from app.rag.chunking import StructuredMarkdownChunker, TextChunker
from app.rag.context import TurnContextResolver
from app.rag.index import VectorIndex
from app.rag.indexing import IndexingService
from app.rag.ingestion_pdf import DocumentIngestor
from app.rag.llm import LlmClient
from app.rag.memory import SessionStore
from app.rag.pipeline_context_support import PipelineContextMixin
from app.rag.pipeline_retrieval_support import PipelineRetrievalMixin
from app.rag.pipeline_runtime_support import PipelineRuntimeMixin
from app.rag.prompting import PromptComposer
from app.rag.retrieval import BGEReranker, HybridRetriever
from app.rag.retrieval_service import RetrievalService
from app.rag.utils import normalize_text
from app.session.repository import SessionRepository
from app.storage import CacheRepository, IndexRepository

logger = logging.getLogger("rag.pipeline")


class RagPipeline(PipelineContextMixin, PipelineRetrievalMixin, PipelineRuntimeMixin):
    RESOURCE_QUERY_PHRASES = {
        "pv": ("persistent volume",),
        "pvc": ("persistent volume claim",),
    }
    RESOURCE_KIND_ALIASES = {
        "pv": {"pv", "persistentvolume"},
        "pvc": {"pvc", "persistentvolumeclaim"},
        "configmap": {"configmap"},
        "secret": {"secret"},
        "pod": {"pod"},
        "deployment": {"deployment"},
        "service": {"service"},
        "route": {"route"},
        "ingress": {"ingress"},
        "storageclass": {"storageclass"},
        "rolebinding": {"rolebinding"},
        "clusterrole": {"clusterrole"},
        "clusterrolebinding": {"clusterrolebinding"},
        "daemonset": {"daemonset"},
        "statefulset": {"statefulset"},
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ingestor = DocumentIngestor(settings)
        self.chunker = TextChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        self.structured_chunker = StructuredMarkdownChunker(
            chunk_size=settings.structured_chunk_size,
            overlap=settings.structured_chunk_overlap,
        )
        self.embedder = BGEOllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            timeout=settings.ollama_timeout,
        )
        settings.vector_dim = self.embedder.dim
        self.index = VectorIndex(settings.db_dsn)
        self.retriever = HybridRetriever(
            top_k=settings.retrieval_top_k,
            candidate_pool_size=settings.candidate_pool_size,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
            rerank_base_weight=settings.rerank_base_weight,
            rerank_overlap_weight=settings.rerank_overlap_weight,
        )
        self.embedding_cache = JsonFileCache(
            settings.rag_cache_dir / "embeddings",
            max_entries=settings.cache_max_entries,
            ttl_hours=settings.cache_ttl_hours,
        )
        self.answer_cache = JsonFileCache(
            settings.rag_cache_dir / "answers",
            max_entries=settings.cache_max_entries,
            ttl_hours=settings.cache_ttl_hours,
        )
        self.session_store = SessionStore(dsn=settings.db_dsn, memory_window_turns=settings.memory_window_turns)
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
        self.intent_agent = IntentAgent(self.llm)
        self.retrieval_agent = RetrievalAgent(self.llm)
        self.answer_agent = AnswerAgent(self.llm)
        self.judge_agent = JudgeAgent(self.llm)
        self.retrieval_service = RetrievalService(settings)
        self.answer_service = AnswerGenerator(self.retrieval_service)
        self.turn_context_resolver = TurnContextResolver()
        self.prompt_composer = PromptComposer(self.session_repository, self.settings)
        self.reranker = BGEReranker(top_k=5)

    def _get_prompt_composer(self) -> PromptComposer:
        composer = getattr(self, "prompt_composer", None)
        if composer is None:
            composer = PromptComposer(self.session_repository, self.settings)
            self.prompt_composer = composer
        return composer

    def _expand_query_with_context(self, query: str, topic_state: dict) -> str:
        if not topic_state:
            return query
        entities = topic_state.get("active_entities", [])
        sources = topic_state.get("selected_sources", [])
        if not entities and not sources:
            return query
        uppercase_re = re.compile(r"[A-Z]{2,}")
        has_explicit_keyword = bool(uppercase_re.search(query))
        query_lower = query.lower()
        expansion_tokens: list[str] = []
        if not has_explicit_keyword:
            for source in sources[:2]:
                stem = re.sub(r"\.[^.]+$", "", source)
                if stem.lower() not in query_lower:
                    expansion_tokens.append(stem)
            added = 0
            for entity in entities:
                if added >= 3:
                    break
                if entity.lower() not in query_lower and len(entity) >= 2:
                    expansion_tokens.append(entity)
                    added += 1
        if not expansion_tokens:
            return query
        return " ".join(expansion_tokens) + " " + query

    def _build_llm_failure_fallback(self, user_message: str, use_retrieved_context: bool, context_blocks: list[str], context_text: str, policy) -> str:  # noqa: ARG002
        if use_retrieved_context and context_blocks:
            fallback_excerpt = self._build_grounded_failure_excerpt(context_blocks)
            return "LLM 응답 생성에 실패했습니다. 검색된 문맥 기준으로 핵심만 정리해 드릴게요.\n\n" + fallback_excerpt
        if policy.turn_type == "greeting":
            return "안녕하세요! 무엇을 도와드릴까요?"
        if policy.needs_clarification and policy.clarification_prompt:
            return policy.clarification_prompt
        if policy.response_mode == "conversational":
            return "문서와 관련된 내용이 더 필요하시면 이어서 질문해 주세요."
        return "현재 LLM 연결이 불안정해 일반 답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."

    def _build_grounded_failure_excerpt(self, context_blocks: list[str]) -> str:
        cleaned_parts: list[str] = []
        for block in context_blocks[:3]:
            text = re.sub(r"```.*?```", " ", block, flags=re.DOTALL)
            filtered_lines: list[str] = []
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("[") and ".pdf" in line.lower():
                    continue
                if "|" in line and "---" in line:
                    continue
                if line.startswith("|"):
                    continue
                filtered_lines.append(line)
            compact = " ".join(filtered_lines)
            compact = re.sub(r"\s+", " ", compact).strip()
            if compact:
                cleaned_parts.append(compact[:220])
        if not cleaned_parts:
            return context_blocks[0][:400] if context_blocks else "관련 문맥을 요약하지 못했습니다."
        return "\n".join(f"- {part}" for part in cleaned_parts[:3])

    @staticmethod
    def _detect_non_korean_query(text: str) -> str | None:
        if not text or not text.strip():
            return None
        has_korean = any("\uAC00" <= ch <= "\uD7A3" or "\u1100" <= ch <= "\u11FF" or "\u3130" <= ch <= "\u318F" for ch in text)
        if has_korean:
            return None
        japanese_chars = [ch for ch in text if ("\u3040" <= ch <= "\u309F") or ("\u30A0" <= ch <= "\u30FF")]
        if japanese_chars:
            return "한국어로 질문해 주세요. 기술 키워드는 그대로 영어로 입력해도 됩니다."
        cjk_chars = [ch for ch in text if ("\u4E00" <= ch <= "\u9FFF") or ("\uF900" <= ch <= "\uFAFF")]
        if not cjk_chars:
            return None
        alpha_chars = [ch for ch in text if ch.isalpha()]
        latin_alpha_chars = [ch for ch in alpha_chars if "a" <= ch.lower() <= "z"]
        normalized = normalize_text(text).lower()
        tokens = re.findall(r"[a-z0-9][a-z0-9_./-]*", normalized)
        technical_token_markers = ("configmap", "secret", "pod", "deployment", "service", "daemonset", "statefulset", "namespace", "openshift", "kubernetes", "yaml", "kubectl", "role", "rolebinding", "clusterrole", "clusterrolebinding", "ingress", "route", "pvc", "storageclass")
        technical_tokens = [token for token in tokens if token.isupper() or any(char.isdigit() for char in token) or any(marker in token for marker in technical_token_markers)]
        total_alpha = len(alpha_chars)
        cjk_ratio = len(cjk_chars) / max(total_alpha, 1)
        latin_ratio = len(latin_alpha_chars) / max(total_alpha, 1)
        if len(cjk_chars) >= 2 and (cjk_ratio >= 0.1 or (latin_ratio < 0.7 and len(technical_tokens) <= 2)):
            return "한국어로 질문해 주세요. 기술 키워드는 그대로 영어로 입력해도 됩니다."
        return None

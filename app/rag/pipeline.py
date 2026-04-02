from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

logger = logging.getLogger("rag.pipeline")

from app.config import Settings
from app.llm import JudgeAgent, QueryAgent
from app.rag.answer import AnswerGenerator
from app.rag.turn_flow import ChatTurnDeps, ChatTurnOrchestrator
from app.rag.context import TurnContextResolver
from app.rag.indexing import IndexingService
from app.rag.policy import TurnPolicyDecision, TurnPolicyInput, TurnPolicyService
from app.rag.prompting import PromptComposer
from app.rag.query import QueryInterpreter
from app.rag.retrieval_state_builder import RetrievalStateBuilder, RetrievalStateDeps
from app.rag.retrieval_service import RetrievalService
from app.rag.cache import JsonFileCache
from app.rag.chunking import StructuredMarkdownChunker, TextChunker
from app.rag.bge_embeddings import BGEOllamaEmbedder
from app.rag.reranker import BGEReranker
from app.rag.index import VectorIndex
from app.rag.ingestion import DocumentIngestor
from app.rag.llm import LlmClient
from app.rag.memory import SessionStore
from app.rag.retrieval import HybridRetriever
from app.rag.utils import normalize_text, stable_hash, tokenize
from app.session.repository import SessionRepository
from app.storage import CacheRepository, IndexRepository


class RagPipeline:
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
        settings.vector_dim = self.embedder.dim  # 1024
        self.index = VectorIndex(settings.db_dsn)
        self.retriever = HybridRetriever(
            top_k=settings.retrieval_top_k,
            candidate_pool_size=settings.candidate_pool_size,
            dense_weight=settings.retrieval_dense_weight,
            sparse_weight=settings.retrieval_sparse_weight,
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
        self.session_store = SessionStore(
            dsn=settings.db_dsn,
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
        self.query_agent = QueryAgent(self.llm)
        self.judge_agent = JudgeAgent(self.llm)
        self.retrieval_service = RetrievalService(settings)
        self.answer_service = AnswerGenerator(self.retrieval_service)
        self.query_interpreter = QueryInterpreter()
        self.turn_policy_service = TurnPolicyService()
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
        """세션 토픽 상태의 엔티티와 출처를 쿼리에 주입하여 키워드 매칭을 보강한다.

        대명사("그거", "이거")나 생략된 주어가 있는 후속 질문에서,
        active_entities의 핵심 용어를 쿼리 앞에 추가하여
        BGE-M3 임베딩의 검색 정확도를 높인다.
        이미 쿼리에 포함된 토큰은 중복 추가하지 않는다.
        쿼리에 이미 명확한 기술 용어(대문자 약어 등)가 있으면 확장을 최소화한다.
        """
        if not topic_state:
            return query

        entities = topic_state.get("active_entities", [])
        sources = topic_state.get("selected_sources", [])
        if not entities and not sources:
            return query

        # 원본 쿼리에서 대문자 약어(2자 이상)를 직접 탐지 — tokenize()는 lower()를 적용하므로 사용 불가
        _uppercase_re = re.compile(r"[A-Z]{2,}")
        has_explicit_keyword = bool(_uppercase_re.search(query))

        query_lower = query.lower()
        expansion_tokens: list[str] = []

        if not has_explicit_keyword:
            # 출처 파일명에서 확장자 제거 후 핵심 키워드 추출
            for source in sources[:2]:
                stem = Path(source).stem
                if stem.lower() not in query_lower:
                    expansion_tokens.append(stem)

            # 엔티티 중 쿼리에 없는 것만 추가 (최대 3개)
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

    def _build_llm_failure_fallback(
        self,
        user_message: str,
        use_retrieved_context: bool,
        context_blocks: list[str],
        context_text: str,
        policy: TurnPolicyDecision,
    ) -> str:
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

        has_korean = any(
            '\uAC00' <= ch <= '\uD7A3' or '\u1100' <= ch <= '\u11FF' or '\u3130' <= ch <= '\u318F'
            for ch in text
        )
        if has_korean:
            return None

        japanese_chars = [
            ch for ch in text
            if ('\u3040' <= ch <= '\u309F') or ('\u30A0' <= ch <= '\u30FF')
        ]
        if japanese_chars:
            return "한국어로 질문해 주세요. 기술 키워드는 그대로 영어로 입력해도 됩니다."

        cjk_chars = [
            ch for ch in text
            if ('\u4E00' <= ch <= '\u9FFF') or ('\uF900' <= ch <= '\uFAFF')
        ]
        if not cjk_chars:
            return None

        alpha_chars = [ch for ch in text if ch.isalpha()]
        latin_alpha_chars = [ch for ch in alpha_chars if 'a' <= ch.lower() <= 'z']
        normalized = normalize_text(text).lower()
        tokens = re.findall(r"[a-z0-9][a-z0-9_./-]*", normalized)
        technical_token_markers = (
            "configmap",
            "secret",
            "pod",
            "deployment",
            "service",
            "daemonset",
            "statefulset",
            "namespace",
            "openshift",
            "kubernetes",
            "yaml",
            "kubectl",
            "role",
            "rolebinding",
            "clusterrole",
            "clusterrolebinding",
            "ingress",
            "route",
            "pvc",
            "storageclass",
        )
        technical_tokens = [
            token for token in tokens
            if token.isupper()
            or any(char.isdigit() for char in token)
            or any(marker in token for marker in technical_token_markers)
        ]
        total_alpha = len(alpha_chars)
        cjk_ratio = len(cjk_chars) / max(total_alpha, 1)
        latin_ratio = len(latin_alpha_chars) / max(total_alpha, 1)
        if len(cjk_chars) >= 2 and (cjk_ratio >= 0.1 or (latin_ratio < 0.7 and len(technical_tokens) <= 2)):
            return "한국어로 질문해 주세요. 기술 키워드는 그대로 영어로 입력해도 됩니다."
        return None

    def _resolve_turn_context(self, session_id: str, user_message: str) -> dict:
        recent_turns = self.session_repository.recent_turns(session_id)
        structured_summary = self.session_repository.structured_summary(session_id)
        session_topic_state = self.session_repository.topic_state(session_id)
        session_topics = self.session_repository.list_topics(session_id)
        current_topic_id = str(session_topic_state.get("last_active_topic_id") or "")
        resolver = getattr(self, "turn_context_resolver", TurnContextResolver())
        resolution = resolver.resolve(
            user_message=user_message,
            session_topics=session_topics,
            recent_turns=recent_turns,
            current_topic_id=current_topic_id or None,
        )
        resolved_topic = (
            self.session_repository.get_topic(resolution.topic_id)
            if resolution.topic_id
            else None
        )
        resolved_topic_state = self._topic_to_topic_state(resolved_topic)
        topic_state = resolved_topic_state or session_topic_state
        scoped_recent_turns = (
            self.session_repository.recent_topic_turns(session_id, resolution.topic_id)
            if resolution.topic_id
            else recent_turns
        )
        policy = self.turn_policy_service.classify(
            TurnPolicyInput(
                user_message=user_message,
                recent_turns=scoped_recent_turns,
                summary=structured_summary,
                topic_state=topic_state,
            )
        )
        if resolution.needs_clarification and resolution.clarification_prompt:
            policy = TurnPolicyDecision(
                turn_type="clarification",
                response_mode="clarification",
                use_retrieval=False,
                use_memory_rewrite=False,
                allow_preview=False,
                allow_citations=False,
                needs_clarification=True,
                clarification_reason="resolver_ambiguous_topic",
                clarification_prompt=resolution.clarification_prompt,
            )
        return {
            "recent_turns": recent_turns,
            "structured_summary": structured_summary,
            "session_topic_state": session_topic_state,
            "session_topics": session_topics,
            "resolution": resolution,
            "resolved_topic": resolved_topic,
            "topic_state": topic_state,
            "scoped_recent_turns": scoped_recent_turns,
            "policy": policy,
        }

    def _build_non_retrieval_state(self, user_message: str, turn_context: dict) -> dict:
        policy: TurnPolicyDecision = turn_context["policy"]
        resolution = turn_context["resolution"]
        return {
            "rewritten_query": user_message.strip(),
            "top_score": 0.0,
            "use_retrieved_context": False,
            "grounded_pages": [],
            "ordered_context_items": [],
            "selected_context_items": [],
            "preferred_preview_source": None,
            "preview_pages": [],
            "response_mode": policy.response_mode,
            "turn_policy": policy.to_dict(),
            "turn_resolution": resolution.to_dict(),
            "resolved_topic_id": resolution.topic_id,
        }

    def _domain_guard_state(self, user_message: str, turn_context: dict) -> dict | None:
        policy: TurnPolicyDecision = turn_context["policy"]
        if policy.use_retrieval:
            return None
        logger.info(
            "[InputGuard] type=%s mode=%s use_retrieval=%s",
            policy.turn_type,
            policy.response_mode,
            policy.use_retrieval,
        )
        return self._build_non_retrieval_state(user_message, turn_context)


    def _should_skip_procedure_shortcut(
        self,
        user_message: str,
        session_topics: list[dict],
        current_topic_id: str | None,
    ) -> bool:
        normalized = normalize_text(user_message).lower()
        if not normalized:
            return False

        explicit_switch_markers = ("back to", "switch to", "다시", "아까", "이전", "말고")
        if not any(marker in normalized for marker in explicit_switch_markers):
            return False

        for topic in session_topics:
            topic_id = str(topic.get("topic_id") or "")
            if current_topic_id and topic_id == current_topic_id:
                continue
            label = normalize_text(str(topic.get("topic_label") or "")).lower()
            sources = [normalize_text(str(value)).lower() for value in topic.get("sources", []) if value]
            entities = [normalize_text(str(value)).lower() for value in topic.get("entities", []) if value]
            if label and label in normalized:
                return True
            if any(source and source in normalized for source in sources[:3]):
                return True
            if any(entity and entity in normalized for entity in entities[:6]):
                return True
        return False

    def _detect_procedure_followup(self, user_message: str, procedure_state: dict) -> dict | None:
        steps = procedure_state.get("steps", [])
        if not steps:
            return None

        normalized = normalize_text(user_message).lower()
        if not normalized:
            return None

        step_match = re.search(r"(\d+)\s*단계", normalized)
        if step_match:
            return {"type": "jump", "step_number": int(step_match.group(1))}
        step_match = re.search(r"\bstep\s*(\d+)\b", normalized)
        if step_match:
            return {"type": "jump", "step_number": int(step_match.group(1))}
        if any(marker in normalized for marker in ("단계별", "step by step", "순서대로", "절차")):
            return {"type": "outline"}
        if any(marker in normalized for marker in ("다음", "계속", "next", "continue")):
            return {"type": "next"}
        if any(marker in normalized for marker in ("처음부터", "1단계부터", "first step")):
            return {"type": "jump", "step_number": 1}
        return None

    def _looks_like_step_navigation_without_state(self, user_message: str, procedure_state: dict) -> bool:
        if procedure_state.get("steps"):
            return False
        normalized = normalize_text(user_message).lower()
        if not normalized:
            return False
        if re.search(r"(\d+)\s*단계", normalized):
            return True
        if re.search(r"\bstep\s*(\d+)\b", normalized):
            return True
        return any(marker in normalized for marker in ("다음 단계", "next step"))

    def _resolve_requested_resource_kinds(self, query_interpretation: dict | None) -> set[str]:
        query_interpretation = query_interpretation or {}
        requested_kinds: set[str] = set()
        for resource in query_interpretation.get("resources", []) or []:
            normalized = str(resource).casefold().strip()
            requested_kinds.update(self.RESOURCE_KIND_ALIASES.get(normalized, {normalized}))
        return requested_kinds

    def _infer_item_resource_kinds(self, item: dict) -> set[str]:
        metadata = item["chunk"].get("metadata", {})
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        explicit_kind_match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$", lowered_text)
        inferred_kinds: set[str] = set()
        if explicit_kind_match:
            inferred_kinds.add(explicit_kind_match.group(1).casefold())
        for signal in metadata.get("code_signals", []) or []:
            normalized = str(signal).casefold().strip()
            for alias_set in self.RESOURCE_KIND_ALIASES.values():
                if normalized in alias_set:
                    inferred_kinds.update(alias_set)
        return inferred_kinds

    def _expand_query_with_resource_aliases(
        self,
        query: str,
        query_interpretation: dict | None,
    ) -> str:
        query_interpretation = query_interpretation or {}
        lowered_query = query.casefold()
        extra_tokens: list[str] = []
        for resource in query_interpretation.get("resources", []) or []:
            normalized = str(resource).casefold().strip()
            phrase_aliases = self.RESOURCE_QUERY_PHRASES.get(normalized, ())
            if any(phrase in lowered_query for phrase in phrase_aliases):
                continue
            for alias in sorted(self.RESOURCE_KIND_ALIASES.get(normalized, {normalized})):
                if alias and alias not in lowered_query:
                    extra_tokens.append(alias)
        if not extra_tokens:
            return query
        return f"{query} {' '.join(extra_tokens)}".strip()

    def _build_procedure_followup_answer(self, followup: dict, procedure_state: dict) -> tuple[str, dict] | None:
        steps = procedure_state.get("steps", [])
        if not steps:
            return None

        current_step = int(procedure_state.get("current_step") or 1)
        total_steps = int(procedure_state.get("total_steps") or len(steps))
        updated_state = {
            **procedure_state,
            "steps": steps,
            "total_steps": total_steps,
        }

        if followup["type"] == "outline":
            lines = ["이전 답변 기준 단계별 정리입니다."]
            for step in steps:
                lines.append(f"{step['step_number']}. {step['title']}")
            updated_state["current_step"] = current_step
            return "\n".join(lines).strip(), updated_state

        if followup["type"] == "next":
            requested_step = min(current_step + 1, total_steps)
        else:
            requested_step = int(followup.get("step_number") or 1)

        matched = next((step for step in steps if int(step["step_number"]) == requested_step), None)
        if matched is None:
            return f"이전 답변 기준으로는 {requested_step}단계가 없습니다. 현재 정리된 단계는 1단계부터 {total_steps}단계까지입니다.", updated_state

        updated_state["current_step"] = requested_step
        parts = [f"{requested_step}단계: {matched['title']}"]
        if matched.get("body"):
            parts.append(matched["body"])
        return "\n\n".join(parts).strip(), updated_state

    def _prefer_block_type_items(
        self,
        items: list[dict],
        *,
        block_type: str,
        limit: int | None = None,
    ) -> list[dict]:
        if not items:
            return []

        preferred = [
            item
            for item in items
            if block_type in str(item["chunk"].get("metadata", {}).get("block_types", "")).split(",")
        ]
        if not preferred:
            return []
        return preferred[:limit] if limit is not None else preferred

    def _heading_overlap_score(self, user_message: str, metadata: dict) -> float:
        query_tokens = {
            token
            for token in tokenize(user_message)
            if len(token) >= 2 and token not in {"yaml", "manifest", "code", "example", "sample", "demo"}
        }
        if not query_tokens:
            return 0.0

        section_title_tokens = set(tokenize(str(metadata.get("section_title", ""))))
        section_path_tokens = set(tokenize(str(metadata.get("section_path", ""))))
        parent_heading_tokens: set[str] = set()
        for heading in metadata.get("parent_headings", []) or []:
            parent_heading_tokens.update(tokenize(str(heading)))

        score = 0.0
        score += 0.2 * len(query_tokens & parent_heading_tokens)
        score += 0.5 * len(query_tokens & section_path_tokens)
        score += 0.8 * len(query_tokens & section_title_tokens)
        return score


    def _metadata_aware_score(
        self,
        user_message: str,
        query_interpretation: dict | None,
        item: dict,
    ) -> dict:
        query_interpretation = query_interpretation or {}
        metadata = item["chunk"].get("metadata", {})
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        block_types = {
            value.strip().casefold()
            for value in str(metadata.get("block_types", "")).split(",")
            if value.strip()
        }
        code_language = str(metadata.get("code_language", "")).casefold()
        code_subtype = str(metadata.get("code_subtype", "")).casefold()
        code_signals = {str(signal).casefold() for signal in metadata.get("code_signals", []) or []}
        explicit_kind_match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$", lowered_text)
        explicit_resource_kind = explicit_kind_match.group(1).casefold() if explicit_kind_match else ""
        query_tokens = {
            token
            for token in query_interpretation.get("normalized_keywords", tokenize(user_message))
            if len(token) >= 2 and token not in {"yaml", "manifest", "code", "example", "sample", "demo"}
        }
        resources = self._resolve_requested_resource_kinds(query_interpretation)
        actions = {str(value).casefold() for value in query_interpretation.get("actions", []) if value}
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
        intent = str(query_interpretation.get("intent", "") or "").casefold()

        heading_score = self._heading_overlap_score(user_message, metadata)
        resource_score = 0.0
        action_score = 0.0
        format_score = 0.0
        shape_score = 0.0
        lexical_score = 0.0
        completeness_score = 0.0

        # focus_multiplier: single-resource explain 시 sibling section을 multiplicative하게 감점
        focus_multiplier = 1.0
        is_single_resource_focus = (
            len(resources) == 1
            and (intent == "explain" or response_shape == "text")
        )

        if resources:
            matched_resources = 0
            for resource in resources:
                if resource == explicit_resource_kind:
                    matched_resources += 1
                    resource_score += 1.0
                elif resource in code_signals:
                    matched_resources += 1
                    resource_score += 0.9
                elif resource in lowered_text:
                    matched_resources += 1
                    resource_score += 0.55
            if matched_resources == 0 and explicit_resource_kind:
                resource_score -= 0.5
            if is_single_resource_focus:
                resource = next(iter(resources))
                section_title = str(metadata.get("section_title", "") or "").casefold()
                section_path = str(metadata.get("section_path", "") or "").casefold()
                parent_headings = " ".join(
                    str(value).casefold()
                    for value in metadata.get("parent_headings", []) or []
                )
                section_focus_text = " ".join([section_title, section_path, parent_headings]).strip()
                if resource and section_focus_text:
                    if resource in section_title:
                        resource_score += 0.9
                        focus_multiplier = 1.0
                    elif resource in section_path or resource in parent_headings:
                        resource_score += 0.55
                        focus_multiplier = 0.95
                    elif resource not in section_focus_text and matched_resources > 0:
                        resource_score -= 0.25

                # focus_ratio: chunk 내 target vs sibling resource 비율로 sibling 판정
                if resource:
                    focus_multiplier = self._compute_focus_multiplier(
                        resource, resources, lowered_text, section_focus_text,
                        focus_multiplier,
                    )

        if "create" in actions:
            if any(marker in lowered_text for marker in ("create", "생성", "만들", "작성")):
                action_score += 0.4
            if any(marker in str(metadata.get("section_title", "")).casefold() for marker in ("create", "생성")):
                action_score += 0.5
        if "compare" in actions and "table" in block_types:
            action_score += 0.5
        if "explain" in actions and "code" not in block_types:
            action_score += 0.25

        if "yaml" in format_constraints:
            if code_language in {"yaml", "yml"}:
                format_score += 1.2
            if code_subtype == "k8s_manifest":
                format_score += 0.9
        if "cli" in format_constraints:
            if code_subtype == "cli_command":
                format_score += 1.1
            if code_language in {"bash", "sh", "shell"}:
                format_score += 0.8
        if "table" in format_constraints and "table" in block_types:
            format_score += 1.0

        if response_shape == "code":
            if "code" in block_types:
                shape_score += 0.75
            elif "table" in block_types:
                shape_score -= 0.15
        elif response_shape == "table":
            if "table" in block_types:
                shape_score += 0.75
            elif "code" in block_types:
                shape_score -= 0.2
        elif response_shape in {"text", "comparison"} and "code" in block_types:
            shape_score -= 0.15

        for token in query_tokens:
            token_casefold = token.casefold()
            if token_casefold in code_signals:
                lexical_score += 0.35
            elif token_casefold in lowered_text:
                lexical_score += 0.12

        if intent in {"yaml_example", "cli_example", "code_example"} and "code" in block_types:
            shape_score += 0.25

        if "code" in block_types:
            completeness_score += self._code_completeness_score(lowered_text)

        metadata_score = heading_score + resource_score + action_score + format_score + shape_score + lexical_score + completeness_score
        rerank_score = float(item.get("rerank_score", 0.0))
        return {
            "heading_overlap_score": heading_score,
            "resource_match_score": resource_score,
            "action_match_score": action_score,
            "format_match_score": format_score,
            "shape_match_score": shape_score,
            "lexical_match_score": lexical_score,
            "completeness_score": completeness_score,
            "focus_multiplier": focus_multiplier,
            "metadata_score": metadata_score,
            "metadata_final_score": rerank_score * focus_multiplier + metadata_score,
        }

    def _compute_focus_multiplier(
        self,
        target_resource: str,
        all_requested_resources: set[str],
        lowered_text: str,
        section_focus_text: str,
        current_multiplier: float,
    ) -> float:
        """chunk 내 target resource 비율(focus_ratio)로 sibling 여부를 판정하여 multiplier를 결정한다."""
        from app.rag.query import QueryInterpreter

        # section 중심이 target resource이면 이미 좋은 multiplier
        if target_resource in section_focus_text.split():
            return max(current_multiplier, 0.95)

        # chunk 내 resource 언급 횟수 계산
        known_resources = set(QueryInterpreter.RESOURCE_MARKERS.keys())
        target_count = lowered_text.count(target_resource)
        sibling_count = 0
        for res in known_resources:
            if res != target_resource and res not in all_requested_resources:
                sibling_count += lowered_text.count(res)

        total_mentions = target_count + sibling_count
        if total_mentions == 0:
            return current_multiplier

        focus_ratio = target_count / total_mentions

        # focus_ratio 기반 multiplier
        if focus_ratio >= 0.6:
            return max(current_multiplier, 0.95)  # target 중심
        elif focus_ratio >= 0.3:
            return min(current_multiplier, 0.85)  # 혼합
        else:
            return min(current_multiplier, 0.65)  # sibling 중심

    def _code_completeness_score(self, lowered_text: str) -> float:
        field_lines = re.findall(r"(?im)^\s*([a-z][a-z0-9_-]*)\s*:", lowered_text)
        unique_fields = {field.casefold() for field in field_lines}
        if not unique_fields:
            return 0.0
        return min(len(unique_fields) * 0.05, 0.35)

    def _metadata_aware_rerank(
        self,
        user_message: str,
        query_interpretation: dict | None,
        items: list[dict],
    ) -> list[dict]:
        if not items:
            return []

        rescored: list[dict] = []
        for item in items:
            rescored.append({**item, **self._metadata_aware_score(user_message, query_interpretation, item)})

        rescored.sort(
            key=lambda item: (
                -float(item.get("metadata_final_score", 0.0)),
                -float(item.get("metadata_score", 0.0)),
                -float(item.get("rerank_score", 0.0)),
            )
        )
        return rescored

    def _has_code_content(self, item: dict) -> bool:
        """청크 텍스트에 코드 블록 패턴이 포함되어 있는지 확인한다."""
        text = str(item["chunk"].get("text", "") or "")
        if not text.strip():
            return False
        candidates = self.answer_service._extract_code_candidates(text)
        return len(candidates) > 0

    def _select_code_example_context_items(
        self,
        user_message: str,
        query_interpretation: dict | None,
        ordered_context_items: list[dict],
        selected_context_items: list[dict],
    ) -> list[dict]:
        candidates = ordered_context_items or selected_context_items
        code_candidates = self._prefer_block_type_items(
            candidates,
            block_type="code",
            limit=None,
        )
        if not code_candidates:
            # Fallback: scan all items for code content embedded in their text
            code_candidates = [item for item in candidates if self._has_code_content(item)]
        if not code_candidates:
            return selected_context_items

        requested_resource_kinds = self._resolve_requested_resource_kinds(query_interpretation)
        if requested_resource_kinds:
            explicit_kind_matches = [
                item for item in code_candidates
                if self._extract_explicit_resource_kind(item) in requested_resource_kinds
            ]
            if explicit_kind_matches:
                code_candidates = explicit_kind_matches
            exact_resource_matches = [
                item for item in code_candidates
                if self._infer_item_resource_kinds(item) & requested_resource_kinds
            ]
            if exact_resource_matches:
                code_candidates = exact_resource_matches

        rescored = self._metadata_aware_rerank(user_message, query_interpretation, code_candidates)

        # Precision filter: if any item has a positive resource match, remove items
        # that have a negative resource match so mismatched resources don't dominate.
        has_positive_match = any(item.get("resource_match_score", 0) > 0 for item in rescored)
        if has_positive_match:
            rescored = [
                item for item in rescored
                if item.get("resource_match_score", 0) >= 0
            ]

        for item in rescored:
            item["code_selection_score"] = float(item.get("metadata_final_score", 0.0))
            item["code_intent_score"] = (
                float(item.get("resource_match_score", 0.0))
                + float(item.get("action_match_score", 0.0))
                + float(item.get("format_match_score", 0.0))
                + float(item.get("shape_match_score", 0.0))
                + float(item.get("lexical_match_score", 0.0))
            )
        return rescored

    def _extract_explicit_resource_kind(self, item: dict) -> str:
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        explicit_kind_match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$", lowered_text)
        return explicit_kind_match.group(1).casefold() if explicit_kind_match else ""

    def _should_expand_local_context(self, query_interpretation: dict | None) -> bool:
        query_interpretation = query_interpretation or {}
        intent = str(query_interpretation.get("intent", "") or "").casefold()
        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
        return intent in {"yaml_example", "cli_example", "code_example", "table"} or response_shape in {"code", "table"}

    def _expand_local_context_items(
        self,
        user_message: str,
        query_interpretation: dict | None,
        index_items: list[dict],
        ranked_items: list[dict],
    ) -> list[dict]:
        if not ranked_items or not self._should_expand_local_context(query_interpretation):
            return ranked_items

        anchor_items = ranked_items[:2]
        anchors: list[dict] = []
        for item in anchor_items:
            chunk = item["chunk"]
            metadata = chunk.get("metadata", {})
            section_path = str(metadata.get("section_path", "") or "")
            section_prefix = section_path.split(">", 1)[0].strip().casefold() if section_path else ""
            anchors.append(
                {
                    "source_path": chunk["source_path"],
                    "page_number": int(chunk.get("page_number") or metadata.get("page_start") or 0),
                    "section_prefix": section_prefix,
                }
            )

        seen_chunk_ids = {item["chunk"]["chunk_id"] for item in ranked_items}
        expanded: list[dict] = list(ranked_items)
        for candidate in index_items:
            chunk = candidate["chunk"]
            chunk_id = chunk.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            metadata = chunk.get("metadata", {})
            candidate_page = int(chunk.get("page_number") or metadata.get("page_start") or 0)
            section_path = str(metadata.get("section_path", "") or "")
            candidate_prefix = section_path.split(">", 1)[0].strip().casefold() if section_path else ""

            matched_anchor = False
            for anchor in anchors:
                if chunk["source_path"] != anchor["source_path"]:
                    continue
                same_page = candidate_page and anchor["page_number"] and candidate_page == anchor["page_number"]
                adjacent_page = candidate_page and anchor["page_number"] and abs(candidate_page - anchor["page_number"]) == 1
                same_section_prefix = candidate_prefix and anchor["section_prefix"] and candidate_prefix == anchor["section_prefix"]
                if same_page or adjacent_page or same_section_prefix:
                    matched_anchor = True
                    break
            if not matched_anchor:
                continue

            expanded.append(
                {
                    "chunk": chunk,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "rerank_score": 0.0,
                }
            )
            seen_chunk_ids.add(chunk_id)

        return self._metadata_aware_rerank(user_message, query_interpretation, expanded)

    def _expand_topic_anchor_context_items(
        self,
        user_message: str,
        query_interpretation: dict | None,
        index_items: list[dict],
        ranked_items: list[dict],
        topic_state: dict,
    ) -> list[dict]:
        query_interpretation = query_interpretation or {}
        topic_state = topic_state or {}
        if not index_items:
            return ranked_items

        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
        intent = str(query_interpretation.get("intent", "") or "").casefold()
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        normalized_user = normalize_text(user_message).lower()
        followup_markers = (
            "그거", "그건", "그 문서", "그 페이지", "그 yaml", "그 코드",
            "그 타입", "타입", "종류", "특징", "자세히", "더 설명",
            "설치", "과정", "구성", "차이", "비교", "다음", "계속",
            "that", "this", "those", "again", "next", "continue",
        )
        text_followup_expansion = (
            intent in {"explain", "compare"}
            and bool(topic_state.get("selected_sources"))
            and (
                any(marker in normalized_user for marker in followup_markers)
                or len(normalized_user) <= 28
            )
        )
        if (
            response_shape not in {"code", "table"}
            and not format_constraints.intersection({"yaml", "cli", "table"})
            and not text_followup_expansion
        ):
            return ranked_items

        anchor_pages = {int(page) for page in topic_state.get("selected_pages", []) if str(page).isdigit()}
        anchor_pages.update(
            int(page) for page in topic_state.get("last_example_source_pages", []) if str(page).isdigit()
        )
        anchor_pages.update(
            int(item.get("page_number"))
            for item in topic_state.get("last_answer_citations", []) or []
            if str(item.get("page_number", "")).isdigit()
        )
        anchor_sections = {
            str(path).casefold().strip()
            for path in topic_state.get("last_grounded_section_paths", [])
            if path
        }
        anchor_sources = {
            str(source).casefold().strip()
            for source in topic_state.get("selected_sources", [])
            if source
        }
        if not anchor_pages and not anchor_sections and not anchor_sources:
            return ranked_items

        seen_chunk_ids = {item["chunk"]["chunk_id"] for item in ranked_items}
        source_anchor_scores: dict[str, float] = {}
        for item in ranked_items:
            source_path = str(item["chunk"].get("source_path") or "")
            source_anchor_scores[source_path] = max(
                source_anchor_scores.get(source_path, 0.0),
                float(item.get("rerank_score", 0.0)),
            )
        expanded: list[dict] = list(ranked_items)
        for candidate in index_items:
            chunk = candidate["chunk"]
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            source_name = Path(str(chunk.get("source_path") or "")).name.casefold()
            if anchor_sources and source_name not in anchor_sources:
                continue
            metadata = chunk.get("metadata", {})
            candidate_page = int(chunk.get("page_number") or metadata.get("page_start") or 0)
            section_path = str(metadata.get("section_path", "") or "").casefold().strip()
            same_page_band = any(abs(candidate_page - anchor_page) <= 1 for anchor_page in anchor_pages if candidate_page)
            same_section = bool(section_path and section_path in anchor_sections)
            if not same_page_band and not same_section and text_followup_expansion:
                same_section = bool(
                    section_path and any(
                        section_path.startswith(anchor_section) or anchor_section.startswith(section_path)
                        for anchor_section in anchor_sections
                    )
                )
            if not same_page_band and not same_section:
                continue
            continuity_score = 0.0
            if text_followup_expansion:
                continuity_score = source_anchor_scores.get(str(chunk.get("source_path") or ""), 0.0) * 0.35
            expanded.append(
                {
                    "chunk": chunk,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "rerank_score": continuity_score,
                }
            )
            seen_chunk_ids.add(chunk_id)

        return self._metadata_aware_rerank(user_message, query_interpretation, expanded)


    def _find_fallback_code_context_items(
        self,
        user_message: str,
        query_interpretation: dict | None,
        index_items: list[dict],
        topic_state: dict | None,
    ) -> list[dict]:
        query_interpretation = query_interpretation or {}
        topic_state = topic_state or {}
        if str(query_interpretation.get("response_shape", "") or "").casefold() != "code":
            return []

        selected_source_names = {
            str(source).casefold().strip()
            for source in topic_state.get("selected_sources", [])
            if source
        }
        candidate_pool: list[dict] = []
        for item in index_items:
            chunk = item["chunk"]
            source_name = Path(str(chunk.get("source_path") or "")).name.casefold()
            if selected_source_names and source_name not in selected_source_names:
                continue
            if self._has_code_content(item) or "code" in str(chunk.get("metadata", {}).get("block_types", "")).split(","):
                candidate_pool.append(
                    {
                        "chunk": chunk,
                        "rerank_score": float(item.get("rerank_score", 0.0)),
                        "dense_score": float(item.get("dense_score", 0.0)),
                        "sparse_score": float(item.get("sparse_score", 0.0)),
                        "score": float(item.get("score", 0.0)),
                    }
                )

        if not candidate_pool:
            return []

        selected = self._select_code_example_context_items(
            user_message,
            query_interpretation,
            candidate_pool,
            candidate_pool,
        )
        return selected[: max(int(self.settings.grounded_chunk_top_n), 1)]

    def _apply_precision_filter(
        self,
        items: list[dict],
        query_interpretation: dict | None,
    ) -> list[dict]:
        """Post-expansion precision pass: remove items too far below the top score or
        with a negative resource match when better-matched items are present.

        This keeps expansion's recall benefit while preventing low-relevance chunks
        from being sent to the LLM as context.
        """
        if not items:
            return items

        query_interpretation = query_interpretation or {}
        resources = {str(v).casefold() for v in query_interpretation.get("resources", []) if v}

        # Drop items with negative resource_match_score when positive matches exist
        if resources:
            has_positive = any(item.get("resource_match_score", 0) > 0 for item in items)
            if has_positive:
                items = [item for item in items if item.get("resource_match_score", 0) >= 0]

        # Drop items whose score is too far below the best
        if len(items) > 1:
            top_score = max(float(item.get("metadata_final_score", 0)) for item in items)
            if top_score > 0:
                threshold = top_score * 0.2
                items = [item for item in items if float(item.get("metadata_final_score", 0)) >= threshold]

        return items

    def _apply_focus_filter(
        self,
        items: list[dict],
        query_interpretation: dict | None,
    ) -> list[dict]:
        """single-resource explain일 때 sibling section chunk를 제거하고 context 수를 제한한다."""
        if not items:
            return items

        qi = query_interpretation or {}
        resources = [str(v).casefold() for v in qi.get("resources", []) if v]
        intent = str(qi.get("intent", "") or "").casefold()
        response_shape = str(qi.get("response_shape", "") or "").casefold()

        if len(resources) != 1 or (intent != "explain" and response_shape != "text"):
            return items

        target = resources[0]

        # focus_multiplier가 낮은(sibling) chunk를 분류
        focused = []
        sibling = []
        for item in items:
            fm = item.get("focus_multiplier", 1.0)
            if fm >= 0.85:
                focused.append(item)
            else:
                sibling.append(item)

        # sibling이 전체의 30% 미만이면 그대로, 이상이면 제거
        if focused and len(sibling) / max(len(items), 1) >= 0.3:
            items = focused

        # single-resource explain은 context 4개로 제한
        max_items = 4
        return items[:max_items]

    def _resolve_answer_route(self, query_interpretation: dict | None) -> str:
        query_interpretation = query_interpretation or {}
        intent = str(query_interpretation.get("intent", "") or "").casefold()
        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()

        if intent in {"yaml_example", "cli_example", "code_example"} or response_shape == "code":
            return "extractive_code"
        if intent == "table" or response_shape == "table":
            return "extractive_table"
        return "grounded_generation"


    def _build_missing_extractive_answer(self, answer_route: str) -> str:
        if answer_route == "extractive_code":
            return (
                "업로드한 문서에서 요청하신 YAML/코드 예시를 직접 찾지 못했습니다. "
                "문서에 실제 예시 블록이 있는지 다시 확인하시거나, 더 구체적인 범위나 페이지를 지정해 주세요."
            )
        if answer_route == "extractive_table":
            return (
                "업로드한 문서에서 요청하신 표나 비교 정보를 직접 찾지 못했습니다. "
                "키워드를 조금 더 구체적으로 적어 다시 질문해 주세요."
            )
        return "업로드한 문서에서 관련 내용을 찾을 수 없습니다."

    def _build_policy_answer(self, turn_type: str, top_score: float) -> str:
        if turn_type == "conversational_ack":
            return "네. 문서와 관련된 질문이 있으시면 이어서 질문해 주세요."
        if turn_type == "greeting":
            return "안녕하세요! 무엇을 도와드릴까요?"
        if turn_type == "general_chat":
            return (
                "죄송합니다. 업로드한 문서와 관련된 질문만 답변할 수 있습니다. "
                "문서 내용에 대해 질문해 주세요."
            )
        if turn_type == "document_query":
            if top_score >= self.settings.retrieval_retry_min_score:
                return (
                    "관련 내용을 찾기 어렵습니다. 질문을 조금 더 구체적으로 적어 주세요.\n"
                    "예: `스토리지 문서에서 PV 설명해줘`, `Service 종류를 서로 비교해줘`"
                )
            return (
                "업로드한 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 하시거나 "
                "관련 문서를 업로드해 주세요."
            )
        return "업로드한 문서에서 관련 내용을 찾을 수 없습니다."

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
            "format_constraints": sorted(
                str(value) for value in interpretation.get("format_constraints", []) if value
            ),
            "normalized_keywords": sorted(
                str(value) for value in interpretation.get("normalized_keywords", []) if value
            ),
        }
        return stable_hash(json.dumps(cache_scope, ensure_ascii=False, sort_keys=True))

    def _should_run_judge_agent(
        self,
        policy_decision: TurnPolicyDecision,
        query_interpretation: dict | None,
        top_score: float,
    ) -> bool:
        query_interpretation = query_interpretation or {}
        if policy_decision.turn_type == "document_followup":
            return False
        if query_interpretation.get("resources") and top_score >= 0.35:
            return False
        return True

    def _canonical_cache_query(
        self,
        user_message: str,
        rewritten_query: str,
        query_interpretation: dict | None,
    ) -> str:
        query_interpretation = query_interpretation or {}
        normalized = normalize_text(user_message).lower()
        normalized_keywords = [
            str(value).lower()
            for value in query_interpretation.get("normalized_keywords", [])
            if value
        ]
        has_referential_marker = any(
            marker in normalized
            for marker in ("洹멸굅", "洹멸굔", "洹몄?", "洹?yaml", "洹?肄붾뱶", "?ㅼ떆", "洹몃읆", "that", "this", "again")
        )
        if query_interpretation.get("resources") and normalized_keywords and not has_referential_marker:
            return " ".join(sorted(dict.fromkeys(normalized_keywords)))
        return rewritten_query

    @staticmethod
    def _topic_to_topic_state(topic: dict | None) -> dict:
        return PromptComposer.topic_to_topic_state(topic)

    def _build_rewrite_context_from_topic(self, topic: dict | None, topic_turns: list) -> dict | None:
        return self._get_prompt_composer().build_rewrite_context_from_topic(topic, topic_turns)

    def _ensure_topic_for_resolution(
        self,
        session_id: str,
        user_message: str,
        state: dict,
        user_turn_id: int | None,
    ) -> str | None:
        resolution = state.get("turn_resolution") or {}
        resolution_type = resolution.get("resolution_type", "")
        if resolution_type == "ambiguous" or user_turn_id is None:
            return None
        topic_id = state.get("resolved_topic_id")
        if not topic_id and resolution_type == "new_topic":
            created_topic = self.session_repository.create_topic(
                session_id,
                seed_label=user_message,
                seed_turn_id=user_turn_id,
            )
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

    def _store_assistant_turn(
        self,
        session_id: str,
        content: str,
        metadata: dict,
        topic_id: str | None,
    ) -> int:
        enriched_metadata = dict(metadata or {})
        stored_procedure_state = enriched_metadata.pop("_stored_procedure_state", {})
        if stored_procedure_state and not enriched_metadata.get("procedure_state"):
            enriched_metadata["procedure_state"] = stored_procedure_state
        enriched_metadata.setdefault("procedure_state", {})
        turn_id = self.session_repository.add_turn(session_id, "assistant", content, metadata=enriched_metadata)
        if topic_id:
            self.session_repository.link_turn_to_topic(
                turn_id,
                session_id,
                topic_id,
                "assistant",
                "continue",
                1.0,
            )
        return turn_id

    def _should_use_retrieved_context(
        self,
        policy: TurnPolicyDecision,
        retrieved: list[dict],
        top_score: float,
        query_interpretation: dict | None = None,
    ) -> bool:
        query_interpretation = query_interpretation or {}
        top_item = retrieved[0] if retrieved else {}
        lexical_signal = (
            float(top_item.get("sparse_score", 0.0))
            + float(top_item.get("title_score", 0.0))
            + float(top_item.get("title_match_bonus", 0.0))
            + float(top_item.get("compact_match_bonus", 0.0))
        )
        resource_match_score = float(top_item.get("resource_match_score", 0.0))
        lexical_match_score = float(top_item.get("lexical_match_score", 0.0))
        query_tokens = {
            token
            for token in query_interpretation.get("normalized_keywords", [])
            if len(token) >= 2 and token not in {"pdf", "설명", "explain"}
        }
        metadata = top_item.get("chunk", {}).get("metadata", {}) if top_item else {}
        structure_text = " ".join(
            [
                str(metadata.get("section_title", "") or ""),
                str(metadata.get("section_path", "") or ""),
                " ".join(str(value) for value in metadata.get("parent_headings", []) or []),
            ]
        )
        structure_tokens = set(tokenize(structure_text))
        has_structural_anchor = bool(query_tokens and query_tokens & structure_tokens)
        strong_resource_anchor = (
            resource_match_score >= 0.9
            or lexical_match_score >= 0.2
            or has_structural_anchor
        )

        if top_score < self.settings.retrieval_min_score or not retrieved:
            lowered_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
            lowered_intent = str(query_interpretation.get("intent", "") or "").casefold()
            has_explicit_resources = bool(query_interpretation.get("resources"))
            relaxed_threshold = self.settings.retrieval_min_score
            if has_explicit_resources:
                relaxed_threshold = min(
                    relaxed_threshold,
                    max(self.settings.retrieval_retry_min_score, 0.10),
                )
            if lowered_shape in {"code", "table", "procedure", "comparison"} or lowered_intent in {
                "yaml_example", "cli_example", "code_example", "table", "compare", "procedure_followup", "explain"
            }:
                relaxed_threshold = min(self.settings.retrieval_min_score, max(self.settings.retrieval_retry_min_score, 0.15))
            if (
                policy.turn_type in {"document_query", "document_followup"}
                and lowered_intent == "explain"
                and (lexical_signal >= 0.06 or has_structural_anchor)
            ):
                relaxed_threshold = min(
                    relaxed_threshold,
                    max(
                        self.settings.retrieval_retry_min_score,
                        0.08 if has_explicit_resources else 0.12,
                    ),
                )
            if (
                policy.turn_type in {"document_query", "document_followup"}
                and lowered_intent == "explain"
                and has_explicit_resources
                and strong_resource_anchor
            ):
                relaxed_threshold = min(
                    relaxed_threshold,
                    max(self.settings.retrieval_retry_min_score * 0.8, 0.04),
                )
            if top_score < relaxed_threshold or not retrieved:
                return False

        # Adaptive acceptance: top-1 vs top-2 score gap이 크면 top-1은 확실히 관련 있음
        if len(retrieved) >= 2:
            second_score = float(retrieved[1].get("rerank_score", 0.0))
            score_gap = top_score - second_score
            if score_gap > 0.15 and top_score >= 0.08:
                return True

        # Explicit resource가 chunk metadata에 존재하면 강하게 accept
        resources = {str(v).casefold() for v in query_interpretation.get("resources", []) if v}
        if resources and top_item:
            chunk_meta = top_item.get("chunk", {}).get("metadata", {})
            code_signals = {str(s).casefold() for s in chunk_meta.get("code_signals", []) or []}
            chunk_text = str(top_item.get("chunk", {}).get("text", "")).casefold()
            explicit_kind = ""
            kind_match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)", chunk_text)
            if kind_match:
                explicit_kind = kind_match.group(1).casefold()
            for res in resources:
                if res == explicit_kind or res in code_signals:
                    return True

        # Dense similarity alone can surface semantically adjacent but irrelevant chunks.
        # For first-turn document queries, require either stronger overall evidence or some lexical match.
        if (
            policy.turn_type == "document_query"
            and top_score < 0.2
            and lexical_signal <= 0.0
            and not strong_resource_anchor
        ):
            return False
        return True

    async def _rewrite_query_with_llm(
        self,
        session_id: str,
        user_message: str,
        rewrite_context: dict | None = None,
    ) -> str:
        """LLM을 사용하여 대화 맥락을 반영한 독립적 검색 질의로 재작성한다.

        하드코딩된 마커나 규칙 대신 LLM이 대화 흐름을 판단하여
        대명사 해소, 토픽 연결, 또는 그대로 반환을 결정한다.
        LLM 호출 실패 시 원본 메시지를 그대로 반환한다.
        """
        rewrite_context = rewrite_context or self.session_repository.build_rewrite_context(
            session_id, user_message,
        )
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

    async def _prepare_retrieval_state(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
    ) -> dict:
        builder = RetrievalStateBuilder(self._build_retrieval_state_deps())
        return await builder.run(session_id, user_message, allowed_source_paths)

    async def inspect_retrieval(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
    ) -> dict:
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
        """소스별로 청크를 인터리브하여 char_limit 내에서 다양한 소스가 앞 순서에 오도록 한다.

        [RBAC1, RBAC2, SCC1] → [RBAC1, SCC1, RBAC2]
        RBAC가 char_limit을 독점해서 SCC 내용이 잘리는 문제를 방지한다.
        """
        source_groups: dict[str, list[dict]] = {}
        for item in items:
            src = item["chunk"]["source_path"]
            source_groups.setdefault(src, []).append(item)
        interleaved: list[dict] = []
        max_len = max((len(v) for v in source_groups.values()), default=0)
        sources = list(source_groups.keys())
        for i in range(max_len):
            for src in sources:
                group = source_groups[src]
                if i < len(group):
                    interleaved.append(group[i])
        return interleaved

    def _build_context_blocks(self, context_items: list[dict]) -> tuple[list[str], list[str]]:
        """검색된 문맥 아이템에서 LLM 프롬프트용 텍스트 블록과 청크 ID 목록을 생성한다."""
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

    def _finalize_answer(
        self,
        answer: str,
        rewritten_query: str,
        use_retrieved_context: bool,
        top_score: float,
        selected_context_items: list[dict],
        grounded_pages: list[dict],
        preferred_preview_source: str | None,
        response_mode: str,
        policy_decision: TurnPolicyDecision,
        query_interpretation: dict | None,
        answer_route: str,
    ) -> tuple[str, list[dict], dict]:
        """답변을 정제하고, 인용·미리보기·최종 페이로드를 구성한다.

        Returns:
            (final_answer, answer_citations, final_payload) 튜플.
        """
        return self.answer_service.finalize_answer(
            answer=answer,
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
            retrieval_min_score=self.settings.retrieval_min_score,
        )

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        append_user_turn: bool = True,
    ) -> AsyncIterator[dict]:
        orchestrator = ChatTurnOrchestrator(self._build_chat_turn_deps())
        async for event in orchestrator.run(
            session_id=session_id,
            user_message=user_message,
            allowed_source_paths=allowed_source_paths,
            append_user_turn=append_user_turn,
        ):
            yield event

    def _build_retrieval_state_deps(self) -> RetrievalStateDeps:
        return RetrievalStateDeps(
            resolve_turn_context=self._resolve_turn_context,
            build_non_retrieval_state=self._build_non_retrieval_state,
            build_rewrite_context_from_topic=self._build_rewrite_context_from_topic,
            rewrite_query_with_llm=self._rewrite_query_with_llm,
            index_repository=self.index_repository,
            query_agent=self.query_agent,
            query_interpreter=self.query_interpreter,
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

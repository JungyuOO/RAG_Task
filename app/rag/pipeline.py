from __future__ import annotations

import logging
import re
import time

from app.config import Settings
from app.llm import AnswerRewriteAgent, IntentAgent, RetrievalAgent
from app.rag.answer import AnswerGenerator
from app.rag.bge_embedding_server import BGETEIEmbedder
from app.rag.bge_embeddings import BGEOllamaEmbedder
from app.rag.cache import JsonFileCache
from app.rag.chunking_markdown import StructuredMarkdownChunker
from app.rag.context import TurnContextResolver
from app.rag.index import VectorIndex
from app.rag.indexing import IndexingService
from app.rag.ingestion_pdf import DocumentIngestor
from app.rag.llm import LlmClient
from app.rag.memory import SessionStore
from app.rag.pipeline_scoring import PipelineRetrievalMixin
from app.rag.pipeline_runtime_support import PipelineRuntimeMixin
from app.rag.prompting import PromptComposer
from app.rag.retrieval import BGEReranker, HybridRetriever
from app.rag.retrieval_service import RetrievalService
from app.rag.types import TurnPolicyDecision
from app.rag.utils import normalize_text, tokenize
from app.session.repository import SessionRepository
from app.storage import CacheRepository, IndexRepository

logger = logging.getLogger("rag.pipeline")


class PipelineContextMixin:
    @staticmethod
    def _turn_value(turn, key: str, default=None):
        if hasattr(turn, key):
            return getattr(turn, key)
        if hasattr(turn, "get"):
            return turn.get(key, default)
        return default

    async def _resolve_turn_context(self, session_id: str, user_message: str) -> dict:
        t_total = time.perf_counter()

        t_repo = time.perf_counter()
        recent_turns = self.session_repository.recent_turns(session_id)
        structured_summary = self.session_repository.structured_summary(session_id)
        session_topic_state = self.session_repository.topic_state(session_id)
        session_topics = self.session_repository.list_topics(session_id)
        logger.info(
            "[Timing][TurnContext] repo_reads=%.3fs recent_turns=%d topics=%d",
            time.perf_counter() - t_repo,
            len(recent_turns),
            len(session_topics),
        )
        detected_version = self._extract_version_from_text(user_message)
        last_assistant_text = ""
        if recent_turns:
            last_turn = recent_turns[-1]
            if str(self._turn_value(last_turn, "role", "")) == "assistant":
                last_assistant_text = str(self._turn_value(last_turn, "content", "") or "")

        awaiting_version_clarification = (
            "어떤 버전의 openshift container platform을 기준으로 답변할까요?" in last_assistant_text.lower()
            or "지원 버전:" in last_assistant_text.lower()
        )

        if detected_version and awaiting_version_clarification:
            logger.info(
                "[VersionClarification] detected_version=%s user_message=%r",
                detected_version,
                user_message,
            )
            session_topic_state = {
                **session_topic_state,
                "selected_versions": [detected_version],
                "pending_version_clarification": False,
            }

        current_topic_id = str(session_topic_state.get("last_active_topic_id") or "")
        resolver = getattr(self, "turn_context_resolver", TurnContextResolver())

        t_resolver = time.perf_counter()
        resolution = resolver.resolve(
            user_message=user_message,
            session_topics=session_topics,
            recent_turns=recent_turns,
            current_topic_id=current_topic_id or None,
        )
        logger.info(
            "[Timing][TurnContext] resolver=%.3fs resolution_type=%s topic_id=%s needs_clarification=%s",
            time.perf_counter() - t_resolver,
            getattr(resolution, "resolution_type", None),
            getattr(resolution, "topic_id", None),
            getattr(resolution, "needs_clarification", None),
        )

        resolved_topic = self.session_repository.get_topic(resolution.topic_id) if resolution.topic_id else None
        resolved_topic_state = self._topic_to_topic_state(resolved_topic)
        topic_state = resolved_topic_state or session_topic_state
        scoped_recent_turns = self.session_repository.recent_topic_turns(session_id, resolution.topic_id) if resolution.topic_id else recent_turns

        t_intent = time.perf_counter()
        intent_result = await self.intent_agent.classify(
            user_message,
            context={
                "active_topic": topic_state.get("active_topic"),
                "selected_sources": topic_state.get("selected_sources", []),
                "procedure_state": topic_state.get("procedure_state", {}),
                "recent_turn_count": len(scoped_recent_turns),
                "summary_topic": structured_summary.get("topic", ""),
            },
        )
        logger.info(
            "[Timing][TurnContext] intent_classify=%.3fs intent=%s",
            time.perf_counter() - t_intent,
            intent_result.get("intent"),
        )

        policy = self._policy_from_intent(
            intent_result,
            resolution.resolution_type,
            bool(topic_state.get("active_topic") or topic_state.get("selected_sources")),
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

        logger.info(
            "[Timing][TurnContext] total=%.3fs policy_turn_type=%s response_mode=%s",
            time.perf_counter() - t_total,
            policy.turn_type,
            policy.response_mode,
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
            "intent_result": intent_result,
        }
    
    def _extract_version_from_text(self, text: str) -> str | None:
        if not text:
            return None
        match = re.search(r"\b(4\.(?:15|16|17|18|19|20|21))\b", text)
        return match.group(1) if match else None
    
    def _policy_from_intent(self, intent_result: dict, resolution_type: str, has_prior_context: bool) -> TurnPolicyDecision:
        intent = str(intent_result.get("intent", "general") or "general").casefold()
        if intent == "greeting":
            return TurnPolicyDecision("greeting", "general", False, False, False, False)
        if intent == "unsupported_language":
            return TurnPolicyDecision("general_chat", "general", False, False, False, False)
        if intent == "step_navigation":
            return TurnPolicyDecision("conversational_ack", "conversational", False, False, False, False)
        if intent in {"rag", "clarification"}:
            use_memory_rewrite = resolution_type in {"continue", "switch_existing"}
            turn_type = "document_followup" if resolution_type == "continue" else "document_query"
            return TurnPolicyDecision(turn_type, "rag", True, use_memory_rewrite, True, True)
        return TurnPolicyDecision("general_chat", "general", False, False, False, False)

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
        logger.info("[InputGuard] type=%s mode=%s use_retrieval=%s", policy.turn_type, policy.response_mode, policy.use_retrieval)
        return self._build_non_retrieval_state(user_message, turn_context)

    def _should_skip_procedure_shortcut(self, user_message: str, session_topics: list[dict], current_topic_id: str | None) -> bool:
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

    def _expand_query_with_resource_aliases(self, query: str, query_interpretation: dict | None) -> str:
        query_interpretation = query_interpretation or {}
        lowered_query = query.casefold()
        
        # Only expand aliases for resources that were explicitly identified in keywords
        # or were inherited (which we can infer if the resource name is in lowered_query or normalized_keywords)
        normalized_keywords = {str(k).casefold() for k in query_interpretation.get("normalized_keywords", [])}
        
        extra_tokens: list[str] = []
        for resource in query_interpretation.get("resources", []) or []:
            normalized = str(resource).casefold().strip()
            
            # Skip hallucinated resources that aren't in the query or keywords
            is_explicit = normalized in lowered_query or normalized in normalized_keywords
            # Check if any alias of this resource is in the query
            has_alias_in_query = False
            for alias in self.RESOURCE_KIND_ALIASES.get(normalized, {normalized}):
                if alias in lowered_query or alias in normalized_keywords:
                    has_alias_in_query = True
                    break
                    
            if not is_explicit and not has_alias_in_query:
                continue

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
        updated_state = {**procedure_state, "steps": steps, "total_steps": total_steps}
        if followup["type"] == "outline":
            lines = ["이전 답변 기준 단계별 정리입니다."]
            for step in steps:
                lines.append(f"{step['step_number']}. {step['title']}")
            updated_state["current_step"] = current_step
            return "\n".join(lines).strip(), updated_state
        requested_step = min(current_step + 1, total_steps) if followup["type"] == "next" else int(followup.get("step_number") or 1)
        matched = next((step for step in steps if int(step["step_number"]) == requested_step), None)
        if matched is None:
            return f"이전 답변 기준으로는 {requested_step}단계가 없습니다. 현재 정리된 단계는 1단계부터 {total_steps}단계까지입니다.", updated_state
        updated_state["current_step"] = requested_step
        parts = [f"{requested_step}단계: {matched['title']}"]
        if matched.get("body"):
            parts.append(matched["body"])
        return "\n\n".join(parts).strip(), updated_state

    def _prefer_block_type_items(self, items: list[dict], *, block_type: str, limit: int | None = None) -> list[dict]:
        if not items:
            return []
        preferred = [item for item in items if block_type in str(item["chunk"].get("metadata", {}).get("block_types", "")).split(",")]
        if not preferred:
            return []
        return preferred[:limit] if limit is not None else preferred

    def _heading_overlap_score(self, user_message: str, metadata: dict) -> float:
        query_tokens = {token for token in tokenize(user_message) if len(token) >= 2 and token not in {"yaml", "manifest", "code", "example", "sample", "demo"}}
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
        self.structured_chunker = StructuredMarkdownChunker(
            chunk_size=settings.structured_chunk_size,
            overlap=settings.structured_chunk_overlap,
            min_chunk_chars=settings.structured_chunk_min_chars,
        )
        if settings.embedding_backend == "tei":
            self.embedder = BGETEIEmbedder(
                base_url=settings.tei_base_url,
                model=settings.tei_embedding_model,
                timeout=settings.tei_timeout,
            )
        else:
            self.embedder = BGEOllamaEmbedder(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embedding_model,
                timeout=settings.ollama_timeout,
            )
        self.index = VectorIndex(settings.db_dsn)
        self.retriever = HybridRetriever(
            top_k=settings.retrieval_top_k,
            candidate_pool_size=settings.candidate_pool_size,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
            rerank_base_weight=settings.rerank_base_weight,
            rerank_overlap_weight=settings.rerank_overlap_weight,
            rerank_title_weight=settings.rerank_title_weight,
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
        self.index_repository = IndexRepository(self.index, rag_source_dir=settings.rag_source_dir)
        self.embedding_cache_repository = CacheRepository(self.embedding_cache)
        self.answer_cache_repository = CacheRepository(self.answer_cache)
        self.session_repository = SessionRepository(self.session_store)
        self.indexing_service = IndexingService(
            settings=settings,
            ingestor=self.ingestor,
            structured_chunker=self.structured_chunker,
            embedder=self.embedder,
            index_repository=self.index_repository,
            embedding_cache_repository=self.embedding_cache_repository,
        )
        self.intent_agent = IntentAgent(self.llm)
        self.retrieval_agent = RetrievalAgent(self.llm)
        self.answer_rewrite_agent = AnswerRewriteAgent(self.llm)
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
        versions = topic_state.get("selected_versions", [])
        resources = topic_state.get("last_explicit_resources", [])
        user_focus = str(topic_state.get("last_user_focus") or "").strip()
        document_group = str(topic_state.get("active_document_group") or topic_state.get("last_document_group_preference") or "auto")
        if not entities and not sources and not versions and not resources and not user_focus and document_group == "auto":
            return query
        uppercase_re = re.compile(r"[A-Z]{2,}")
        has_explicit_keyword = bool(uppercase_re.search(query))
        query_lower = query.lower()
        expansion_tokens: list[str] = []
        if not has_explicit_keyword:
            if document_group == "customer_generated" and "customer guide" not in query_lower:
                expansion_tokens.extend(["customer", "guide"])
            elif document_group == "official_ocp" and "official docs" not in query_lower:
                expansion_tokens.extend(["official", "docs"])
            for version in versions[:2]:
                if version and version.lower() not in query_lower:
                    expansion_tokens.append(version)
            for source in sources[:2]:
                stem = re.sub(r"\.[^.]+$", "", source)
                if stem.lower() not in query_lower:
                    expansion_tokens.append(stem)
            for resource in resources[:2]:
                if resource and resource.lower() not in query_lower:
                    expansion_tokens.append(resource)
            if user_focus and user_focus.lower() not in query_lower:
                expansion_tokens.append(user_focus)
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
            return "문서 기준으로 정리하면 다음과 같습니다.\n\n" + fallback_excerpt
        if policy.turn_type == "greeting":
            return "안녕하세요! 무엇을 도와드릴까요?"
        if policy.needs_clarification and policy.clarification_prompt:
            return policy.clarification_prompt
        if policy.response_mode == "conversational":
            return "문서와 관련된 내용이 더 필요하시면 이어서 질문해 주세요."
        return "현재 문서 기준으로 바로 답할 수 있는 근거를 찾지 못했습니다. 질문 범위를 조금 더 구체적으로 적어 주세요."

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
                cleaned_parts.append(compact[:260])
        if not cleaned_parts:
            return context_blocks[0][:400] if context_blocks else "관련 문맥을 요약하지 못했습니다."
        if len(cleaned_parts) == 1:
            return cleaned_parts[0]
        return "\n\n".join(f"{idx + 1}. {part}" for idx, part in enumerate(cleaned_parts[:3]))

    @staticmethod
    def _detect_non_korean_query(text: str) -> str | None:
        result = IntentAgent._detect_unsupported_language(text)
        if result is not None:
            return result.get("message", "한국어로 질문해 주세요.")
        return None

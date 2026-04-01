from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

logger = logging.getLogger("rag.pipeline")

from app.config import Settings
from app.rag.cache import JsonFileCache
from app.rag.chunking import StructuredMarkdownChunker, TextChunker
from app.rag.bge_embeddings import BGEOllamaEmbedder, EmbeddingModelUnavailableError
from app.rag.reranker import BGEReranker
from app.rag.index import VectorIndex
from app.rag.ingestion import DocumentIngestor
from app.rag.llm import LlmClient
from app.rag.memory import SessionStore
from app.rag.retrieval import HybridRetriever
from app.rag.utils import normalize_text, stable_hash, tokenize
from app.repositories.cache_repository import CacheRepository
from app.repositories.index_repository import IndexRepository
from app.repositories.session_repository import SessionRepository
from app.services.agent_service import JudgeAgent, QueryAgent
from app.services.answer_service import AnswerService
from app.services.indexing_service import IndexingService
from app.services.query_interpreter import QueryInterpreter
from app.services.retrieval_service import RetrievalService
from app.services.turn_context_resolver import TurnContextResolver
from app.services.turn_policy_service import TurnPolicyDecision, TurnPolicyInput, TurnPolicyService


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
        self.answer_service = AnswerService(self.retrieval_service)
        self.query_interpreter = QueryInterpreter()
        self.turn_policy_service = TurnPolicyService()
        self.turn_context_resolver = TurnContextResolver()
        self.reranker = BGEReranker(top_k=5)

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

    def _strip_code_blocks_for_non_code_route(self, answer: str, answer_route: str) -> str:
        if answer_route == "extractive_code":
            return answer
        return re.sub(r"```(?:[\w+-]+)?\n.*?```", "", answer, flags=re.DOTALL).strip()

    def _build_example_anchor(
        self,
        context_items: list[dict],
        query_interpretation: dict | None,
    ) -> dict:
        query_interpretation = query_interpretation or {}
        fields: list[str] = []
        context_ids: list[str] = []
        page_numbers: list[int] = []
        section_paths: list[str] = []
        source_path = ""
        resource_kind = str(query_interpretation.get("resources", [""])[0] or "")
        for item in context_items[:3]:
            chunk = item["chunk"]
            if not source_path:
                source_path = str(chunk.get("source_path") or "")
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id and chunk_id not in context_ids:
                context_ids.append(chunk_id)
            page_number = int(chunk.get("page_number") or chunk.get("metadata", {}).get("page_start") or 0)
            if page_number and page_number not in page_numbers:
                page_numbers.append(page_number)
            section_path = str(chunk.get("metadata", {}).get("section_path", "") or "")
            if section_path and section_path not in section_paths:
                section_paths.append(section_path)
            for field in re.findall(r"(?im)^\s*([a-z][a-z0-9_-]*)\s*:", str(chunk.get("text", "") or "")):
                lowered = field.casefold()
                if lowered not in fields:
                    fields.append(lowered)
        return {
            "resource_kind": resource_kind,
            "context_ids": context_ids[:6],
            "source_path": source_path,
            "page_numbers": page_numbers[:6],
            "section_paths": section_paths[:4],
            "fields": fields[:12],
        }

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

    def _extract_procedure_state(self, answer: str) -> dict:
        if not answer:
            return {}

        steps: list[dict] = []
        current_step: dict | None = None
        for raw_line in answer.replace("\r\n", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            normalized_line = re.sub(r"^\*+|\*+$", "", line).strip()
            match = re.match(r"^(?:\*\*)?(\d+)(?:\.\s+|\s*단계[:\s]+)(.+?)(?:\*\*)?$", normalized_line)
            if match:
                current_step = {
                    "step_number": int(match.group(1)),
                    "title": match.group(2).strip(),
                    "body_lines": [],
                }
                steps.append(current_step)
                continue
            if current_step is not None:
                current_step["body_lines"].append(line)

        if len(steps) < 2:
            return {}

        normalized_steps = []
        for step in steps:
            body = "\n".join(step["body_lines"]).strip()
            normalized_steps.append(
                {
                    "step_number": step["step_number"],
                    "title": step["title"],
                    "body": body,
                }
            )
        return {
            "mode": "procedure",
            "steps": normalized_steps,
            "current_step": 1,
            "total_steps": len(normalized_steps),
        }

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

    def _code_intent_score(self, user_message: str, item: dict) -> float:
        normalized = (user_message or "").casefold()
        metadata = item["chunk"].get("metadata", {})
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        code_language = str(metadata.get("code_language", "")).casefold()
        code_subtype = str(metadata.get("code_subtype", "")).casefold()
        code_signals = {str(signal).casefold() for signal in metadata.get("code_signals", []) or []}
        explicit_kind_match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$", lowered_text)
        explicit_resource_kind = explicit_kind_match.group(1).casefold() if explicit_kind_match else ""
        known_resource_kinds = {
            "configmap",
            "secret",
            "pod",
            "deployment",
            "service",
            "persistentvolume",
            "persistentvolumeclaim",
        }
        requested_resource_kinds = set()
        for resource_key, aliases in self.RESOURCE_KIND_ALIASES.items():
            if any(alias in normalized for alias in aliases):
                requested_resource_kinds.update(aliases)
        candidate_resource_kinds = code_signals & known_resource_kinds

        score = 0.0
        yaml_requested = any(marker in normalized for marker in ("yaml", "manifest", "매니페스트"))
        cli_requested = any(marker in normalized for marker in ("oc ", "kubectl", "cli", "command", "명령어", "커맨드"))
        create_requested = any(marker in normalized for marker in ("create", "생성", "만들", "작성"))

        if yaml_requested:
            if code_language in {"yaml", "yml"}:
                score += 1.2
            if code_subtype == "k8s_manifest":
                score += 1.0
        if cli_requested:
            if code_subtype == "cli_command":
                score += 1.1
            if code_language in {"bash", "sh", "shell"}:
                score += 0.8
        if create_requested and any(marker in normalized for marker in ("configmap", "secret", "deployment", "pod", "service")):
            if "create" in lowered_text or "생성" in lowered_text:
                score += 0.3
        if requested_resource_kinds:
            if explicit_resource_kind:
                if explicit_resource_kind in requested_resource_kinds:
                    score += 0.9
                else:
                    score -= 0.35
            elif requested_resource_kinds & candidate_resource_kinds:
                score += 0.9
            elif candidate_resource_kinds:
                score -= 0.35

        for token in tokenize(user_message):
            if len(token) < 2:
                continue
            token_casefold = token.casefold()
            if token_casefold in {"yaml", "manifest", "code", "example", "sample", "demo"}:
                continue
            if token_casefold in code_signals:
                score += 0.45
            elif token_casefold in lowered_text:
                score += 0.18

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
        return {
            "heading_overlap_score": heading_score,
            "resource_match_score": resource_score,
            "action_match_score": action_score,
            "format_match_score": format_score,
            "shape_match_score": shape_score,
            "lexical_match_score": lexical_score,
            "completeness_score": completeness_score,
            "metadata_score": metadata_score,
            "metadata_final_score": float(item.get("rerank_score", 0.0)) + metadata_score,
        }

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
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        if response_shape not in {"code", "table"} and not format_constraints.intersection({"yaml", "cli", "table"}):
            return ranked_items

        anchor_pages = {int(page) for page in topic_state.get("last_example_source_pages", []) if str(page).isdigit()}
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
        if not anchor_pages and not anchor_sections:
            return ranked_items

        seen_chunk_ids = {item["chunk"]["chunk_id"] for item in ranked_items}
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
            if not same_page_band and not same_section:
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

    def _looks_like_example_anchor_followup(
        self,
        user_message: str,
        query_interpretation: dict | None,
        topic_state: dict | None,
    ) -> bool:
        query_interpretation = query_interpretation or {}
        topic_state = topic_state or {}
        if not topic_state.get("last_example_source_pages"):
            return False
        normalized = normalize_text(user_message).lower()
        if not normalized:
            return False
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
        if format_constraints.intersection({"yaml", "cli"}) or response_shape == "code":
            return True
        return any(marker in normalized for marker in ("그 yaml", "그 코드", "field", "필드", "속성", "selector", "port", "host"))

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

    def _resolve_answer_route(self, query_interpretation: dict | None) -> str:
        query_interpretation = query_interpretation or {}
        intent = str(query_interpretation.get("intent", "") or "").casefold()
        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()

        if intent in {"yaml_example", "cli_example", "code_example"} or response_shape == "code":
            return "extractive_code"
        if intent == "table" or response_shape == "table":
            return "extractive_table"
        return "grounded_generation"

    def _should_capture_procedure_state(
        self,
        query_interpretation: dict | None,
        answer_route: str,
        policy_decision: TurnPolicyDecision,
    ) -> bool:
        if answer_route == "procedure_state_followup":
            return True
        query_interpretation = query_interpretation or {}
        intent = str(query_interpretation.get("intent", "") or "").casefold()
        response_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
        if intent == "procedure_followup" or response_shape == "procedure":
            return True
        if policy_decision.response_mode == "conversational" and policy_decision.turn_type != "conversational_ack":
            return True
        return False

    def _build_missing_extractive_answer(self, answer_route: str) -> str:
        if answer_route == "extractive_code":
            return (
                "업로드된 문서에서 요청하신 YAML/코드 예시를 직접 찾지 못했습니다. "
                "문서에 실제 예시 블록이 있는지 다시 확인할 수 있도록 더 구체적인 범위나 페이지를 지정해 주세요."
            )
        if answer_route == "extractive_table":
            return "업로드된 문서에서 요청하신 표/비교 정보를 직접 찾지 못했습니다. 키워드를 조금 더 구체적으로 적어 다시 질문해 주세요."
        return "업로드된 문서에서 관련 내용을 찾을 수 없습니다."

    def _build_policy_answer(self, turn_type: str, top_score: float) -> str:
        if turn_type == "conversational_ack":
            return "네. 문서와 관련된 질문이 있으면 이어서 질문해 주세요."
        if turn_type == "greeting":
            return "안녕하세요. 업로드된 문서에 대해 질문해 주세요."
        if turn_type == "general_chat":
            return "죄송합니다. 업로드된 문서와 관련된 질문만 답변할 수 있습니다. 문서 내용에 대한 질문을 남겨 주세요."
        if turn_type == "document_query":
            if top_score >= self.settings.retrieval_retry_min_score:
                return (
                    "관련 내용을 찾기 어렵습니다. 질문을 조금 더 구체적으로 적어 주세요.\n"
                    "예: `스토리지에서 PV 설명해줘`, `Service 종류를 표로 정리해줘`"
                )
            return "업로드된 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 하시거나 관련 문서를 업로드해 주세요."
        return "업로드된 문서에서 관련 내용을 찾을 수 없습니다."

    def _build_answer_cache_key(
        self,
        session_id: str,
        rewritten_query: str,
        context_ids: list[str],
        answer_route: str,
        query_interpretation: dict | None,
        topic_id: str | None,
    ) -> str:
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
        normalized_keywords = [str(value).lower() for value in query_interpretation.get("normalized_keywords", []) if value]
        has_referential_marker = any(
            marker in normalized
            for marker in ("그거", "그건", "그중", "그 yaml", "그 코드", "다시", "그럼", "that", "this", "again")
        )
        if query_interpretation.get("resources") and normalized_keywords and not has_referential_marker:
            return " ".join(sorted(dict.fromkeys(normalized_keywords)))
        return rewritten_query

    def _public_context_payload(self, payload: dict) -> dict:
        public_payload = dict(payload or {})
        public_payload.pop("_stored_procedure_state", None)
        return public_payload

    def _build_prompt_memory_snapshot(self, session_id: str, topic_id: str | None = None) -> dict:
        snapshot = self.session_repository.memory_snapshot(session_id)
        summary = snapshot.get("session_summary", {})
        topic_state = snapshot.get("topic_state", {})
        recent_turns = snapshot.get("recent_turns", [])
        if topic_id:
            topic = self.session_repository.get_topic(topic_id)
            if topic is not None:
                topic_summary = self.session_repository.topic_memory_snapshot(session_id, topic_id)
                topic_state = self._topic_to_topic_state(topic)
                summary = {
                    "topic": topic_summary.get("topic_label", ""),
                    "user_goal": topic_summary.get("last_user_focus", ""),
                    "recent_documents": topic_summary.get("sources", [])[:3],
                    "recent_pages": topic_summary.get("important_pages", [])[:4],
                }
                recent_turns = [turn.to_dict() for turn in self.session_repository.recent_topic_turns(session_id, topic_id)]
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
            "last_explicit_resource": topic_state.get("last_explicit_resource", ""),
            "last_explicit_resources": topic_state.get("last_explicit_resources", [])[:4],
            "last_intent": topic_state.get("last_intent", ""),
            "last_response_shape": topic_state.get("last_response_shape", ""),
            "last_answer_route": topic_state.get("last_answer_route", ""),
            "last_format_constraints": topic_state.get("last_format_constraints", [])[:4],
            "last_code_resource_kind": topic_state.get("last_code_resource_kind", ""),
            "recent_turns": compact_recent_turns,
        }

    def _build_prompt_recent_turns(self, session_id: str, topic_id: str | None = None) -> list[dict]:
        prompt_recent_turns = max(int(self.settings.llm_prompt_recent_turns), 1)
        if topic_id:
            recent_turns = self.session_repository.recent_topic_turns(session_id, topic_id)[-prompt_recent_turns:]
        else:
            recent_turns = self.session_repository.recent_turns(session_id)[-prompt_recent_turns:]
        return [
            {
                "role": turn.role,
                "content": str(turn.content)[:500],
            }
            for turn in recent_turns
        ]

    @staticmethod
    def _topic_to_topic_state(topic: dict | None) -> dict:
        if not topic:
            return {}
        summary = topic.get("summary", {})
        return {
            "active_topic": topic.get("topic_label") or summary.get("topic_label") or "",
            "active_entities": topic.get("entities", [])[:6],
            "selected_sources": topic.get("sources", [])[:3],
            "selected_pages": summary.get("important_pages", [])[:5],
            "last_retrieval_mode": topic.get("last_retrieval_mode", ""),
            "last_answer_citations": [],
            "last_user_focus": topic.get("last_user_focus", ""),
            "recent_user_topics": [topic.get("topic_label") or summary.get("topic_label") or ""],
            "last_explicit_resource": summary.get("last_explicit_resource", ""),
            "last_explicit_resources": summary.get("last_explicit_resources", [])[:4],
            "last_intent": summary.get("last_intent", ""),
            "last_response_shape": summary.get("last_response_shape", ""),
            "last_answer_route": summary.get("last_answer_route", ""),
            "last_format_constraints": summary.get("last_format_constraints", [])[:4],
            "last_code_resource_kind": summary.get("last_code_resource_kind", ""),
            "last_grounded_chunk_ids": summary.get("last_grounded_chunk_ids", [])[:6],
            "last_grounded_section_paths": summary.get("last_grounded_section_paths", [])[:4],
            "last_example_source_pages": summary.get("last_example_source_pages", [])[:6],
            "last_example_anchor": summary.get("last_example_anchor", {}),
        }

    def _build_rewrite_context_from_topic(self, topic: dict | None, topic_turns: list) -> dict | None:
        if not topic:
            return None
        topic_state = self._topic_to_topic_state(topic)
        conversation_history: list[dict] = []
        last_assistant_turn = None
        for turn in topic_turns[-4:]:
            entry = {"role": turn.role, "content": str(turn.content)[:200]}
            if turn.role == "assistant" and turn.metadata:
                sources = [
                    str(item.get("file_name", ""))
                    for item in turn.metadata.get("source_grounding", [])[:2]
                    if item.get("file_name")
                ]
                if sources:
                    entry["sources"] = sources
                last_assistant_turn = turn
            conversation_history.append(entry)

        # 마지막 assistant 응답의 포맷/형태 정보 추출 (LLM 재작성에 활용)
        last_response_shape = ""
        last_response_intent = ""
        if last_assistant_turn and last_assistant_turn.metadata:
            qi = last_assistant_turn.metadata.get("query_interpretation") or {}
            last_response_shape = str(qi.get("response_shape") or "")
            last_response_intent = str(qi.get("intent") or "")

        return {
            "conversation_history": conversation_history,
            "active_topic": str(topic_state.get("active_topic") or ""),
            "active_entities": topic_state.get("active_entities", [])[:6],
            "selected_sources": topic_state.get("selected_sources", [])[:3],
            "selected_pages": topic_state.get("selected_pages", [])[:5],
            "last_retrieval_mode": str(topic_state.get("last_retrieval_mode") or ""),
            "last_response_shape": last_response_shape,
            "last_response_intent": last_response_intent,
            "last_explicit_resources": topic_state.get("last_explicit_resources", [])[:4],
            "last_code_resource_kind": str(topic_state.get("last_code_resource_kind") or ""),
            "last_example_anchor": topic_state.get("last_example_anchor", {}),
        }

    def _build_prompt_recent_turns_clean(self, session_id: str, topic_id: str | None = None) -> list[dict]:
        """새로운 토픽 전환 시, assistant 답변에서 소스 인용 라인을 제거한 최근 대화를 반환한다.

        이전 RAG 답변의 '[파일.pdf] p.X' 형태 인용이 무관한 새 질문에 bleeding되는 것을 방지한다.
        """
        citation_pattern = re.compile(r'\[[^\]]+\.(?:pdf|PDF)[^\]]*\][^\n]*')
        turns = self._build_prompt_recent_turns(session_id, topic_id=topic_id)
        cleaned = []
        for turn in turns:
            if turn["role"] == "assistant":
                content = citation_pattern.sub('', turn["content"]).strip()
                cleaned.append({**turn, "content": content})
            else:
                cleaned.append(turn)
        return cleaned

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

    def rebuild_index(self, source_paths: list[Path], progress_callback=None) -> dict:
        return self.indexing_service.rebuild_index(source_paths, progress_callback=progress_callback)

    def index_single_file(self, source_path: Path, progress_callback=None) -> dict:
        return self.indexing_service.index_single_file(source_path, progress_callback=progress_callback)

    def _should_use_retrieved_context(
        self,
        policy: TurnPolicyDecision,
        retrieved: list[dict],
        top_score: float,
        query_interpretation: dict | None = None,
    ) -> bool:
        if top_score < self.settings.retrieval_min_score or not retrieved:
            query_interpretation = query_interpretation or {}
            lowered_shape = str(query_interpretation.get("response_shape", "") or "").casefold()
            lowered_intent = str(query_interpretation.get("intent", "") or "").casefold()
            has_explicit_resources = bool(query_interpretation.get("resources"))
            relaxed_threshold = self.settings.retrieval_min_score
            if has_explicit_resources or lowered_shape in {"code", "table", "procedure", "comparison"} or lowered_intent in {
                "yaml_example", "cli_example", "code_example", "table", "compare", "procedure_followup", "explain"
            }:
                relaxed_threshold = min(self.settings.retrieval_min_score, max(self.settings.retrieval_retry_min_score, 0.15))
            if top_score < relaxed_threshold or not retrieved:
                return False

        top_item = retrieved[0]
        lexical_signal = (
            float(top_item.get("sparse_score", 0.0))
            + float(top_item.get("title_score", 0.0))
            + float(top_item.get("title_match_bonus", 0.0))
            + float(top_item.get("compact_match_bonus", 0.0))
        )

        # Dense similarity alone can surface semantically adjacent but irrelevant chunks.
        # For first-turn document queries, require either stronger overall evidence or some lexical match.
        if (
            policy.turn_type == "document_query"
            and top_score < 0.2
            and lexical_signal <= 0.0
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
        turn_context = self._resolve_turn_context(session_id, user_message)
        resolution = turn_context["resolution"]
        resolved_topic = turn_context["resolved_topic"]
        topic_state = turn_context["topic_state"]
        scoped_recent_turns = turn_context["scoped_recent_turns"]
        policy: TurnPolicyDecision = turn_context["policy"]
        if not policy.use_retrieval:
            return self._build_non_retrieval_state(user_message, turn_context)
        rewrite_context = (
            self._build_rewrite_context_from_topic(resolved_topic, scoped_recent_turns)
            if resolved_topic is not None
            else None
        )
        rewritten_query = (
            await self._rewrite_query_with_llm(session_id, user_message, rewrite_context=rewrite_context)
            if policy.use_memory_rewrite
            else user_message.strip()
        )

        # --- QueryAgent: 검색 쿼리 최적화 및 대안 쿼리 생성 ---
        index_items_all = self.index_repository.load()
        # 전체 소스 목록은 인덱스에서 유니크하게 추출
        all_sources: list[str] = []
        seen_sources: set[str] = set()
        for item in index_items_all:
            src = item["chunk"]["source_path"]
            if src not in seen_sources:
                seen_sources.add(src)
                all_sources.append(src)
        query_result = await self.query_agent.refine_query(
            rewritten_query,
            context={"active_topic": topic_state.get("active_topic"), "selected_sources": topic_state.get("selected_sources", [])},
            available_sources=all_sources,
        )
        refined_query = query_result["refined_query"]
        alternative_queries = query_result.get("alternative_queries", [])
        logger.info(
            "[QueryAgent] 원본=%r → 최적화=%r | 대안=%r | 키워드=%r",
            rewritten_query, refined_query, alternative_queries, query_result.get("search_keywords", []),
        )

        query_interpretation = self.query_interpreter.interpret(
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
        aliased_query = self._expand_query_with_resource_aliases(refined_query, query_interpretation.to_dict())
        expanded_query = self._expand_query_with_context(aliased_query, topic_state)
        query_vector = self.embedder.encode(expanded_query)
        index_items = self.retrieval_service.filter_index_items(index_items_all, allowed_source_paths)
        retrieved = self.retriever.search_rrf(
            expanded_query, query_vector, index_items, rrf_k=60,
        )

        # 대안 쿼리 결과를 원본과 병합하여 recall을 높인다.
        # chunk_id 기준 중복 제거 후 rerank_score 내림차순으로 top_k개를 선택한다.
        seen_chunk_ids: set[str] = {r["chunk"]["chunk_id"] for r in retrieved}
        merged_extras: list[dict] = []
        for alt_query in alternative_queries[:2]:
            alt_aliased = self._expand_query_with_resource_aliases(alt_query, query_interpretation.to_dict())
            alt_expanded = self._expand_query_with_context(alt_aliased, topic_state)
            alt_vector = self.embedder.encode(alt_expanded)
            alt_retrieved = self.retriever.search_rrf(
                alt_expanded, alt_vector, index_items, rrf_k=60,
            )
            for item in alt_retrieved:
                cid = item["chunk"]["chunk_id"]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    merged_extras.append(item)
        if merged_extras:
            combined = retrieved + merged_extras
            combined.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            retrieved = combined[: self.retriever.top_k]

        for i, item in enumerate(retrieved[:5]):
            chunk = item["chunk"]
            logger.info(
                "[Retrieval] #%d %s p.%s | rerank=%.4f dense=%.4f sparse=%.4f",
                i + 1,
                Path(chunk["source_path"]).name,
                chunk.get("page_number", "?"),
                item.get("rerank_score", 0),
                item.get("dense_score", 0),
                item.get("sparse_score", 0),
            )

        # --- Reranker: RRF 후보 top-N을 cross-encoder로 재순위 ---
        if retrieved:
            extended = self.retriever.search_rrf(
                expanded_query, query_vector, index_items,
                rrf_k=60,
            )
            extended = extended[:20]
            retrieved = self.reranker.rerank(expanded_query, extended)
            retrieved = self._metadata_aware_rerank(
                user_message,
                query_interpretation.to_dict(),
                retrieved,
            )
            retrieved = self._expand_local_context_items(
                user_message,
                query_interpretation.to_dict(),
                index_items,
                retrieved,
            )
            retrieved = self._expand_topic_anchor_context_items(
                user_message,
                query_interpretation.to_dict(),
                index_items,
                retrieved,
                topic_state,
            )

        retrieval_metrics = self.retriever.compute_retrieval_metrics(
            retrieved, min_score=self.settings.retrieval_min_score,
        )
        top_score = retrieval_metrics["top_score"]
        use_retrieved_context = self._should_use_retrieved_context(
            policy,
            retrieved,
            top_score,
            query_interpretation.to_dict(),
        )
        logger.info(
            "[Retrieval] top_score=%.4f use_context=%s min_score=%.4f",
            top_score, use_retrieved_context, self.settings.retrieval_min_score,
        )
        context_items = retrieved if use_retrieved_context else []
        grounded_pages = self.retrieval_service.aggregate_page_grounding(context_items)
        ordered_context_items = self.retrieval_service.order_context_items_by_grounded_pages(context_items, grounded_pages)
        selected_context_items = self.retrieval_service.select_context_items_by_grounded_pages(ordered_context_items, grounded_pages)
        selected_context_items = self._apply_precision_filter(
            selected_context_items,
            query_interpretation.to_dict() if hasattr(query_interpretation, "to_dict") else query_interpretation,
        )
        if not selected_context_items and str(query_interpretation.response_shape or "").casefold() == "code":
            fallback_code_items = self._find_fallback_code_context_items(
                user_message,
                query_interpretation.to_dict(),
                index_items,
                topic_state,
            )
            if fallback_code_items:
                selected_context_items = fallback_code_items
                ordered_context_items = fallback_code_items
                grounded_pages = self.retrieval_service.aggregate_page_grounding(fallback_code_items)
                top_score = max(top_score, max(float(item.get("rerank_score", 0.0)) for item in fallback_code_items))
                use_retrieved_context = True
        preferred_preview_source = self.retrieval_service.select_grounded_preview_source(grounded_pages)
        preview_pages = self.retrieval_service.build_grounded_preview_pages(preferred_preview_source, grounded_pages)
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
            "query_interpretation": query_interpretation.to_dict(),
        }

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

    def list_library_documents(self) -> dict:
        return self.indexing_service.list_library_documents()

    def delete_library_document(self, source_path: Path) -> dict:
        return self.indexing_service.delete_library_document(source_path)

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
        answer = self._strip_code_blocks_for_non_code_route(answer, answer_route)
        answer = self.answer_service.sanitize_answer(answer, use_retrieved_context)
        answer_citations = (
            self.answer_service.build_answer_citation_payload(
                answer, selected_context_items, grounded_pages, preferred_preview_source,
            )
            if policy_decision.allow_citations
            else []
        )
        final_answer = self.answer_service.ensure_answer_source_line(answer, answer_citations, use_retrieved_context)
        # 검색 점수가 충분히 높을 때만 자료보기를 표시한다.
        # 점수가 낮은 애매한 결과에 자료보기를 보여주면 신뢰도가 떨어진다.
        show_preview = policy_decision.allow_preview and top_score >= self.settings.retrieval_min_score
        if show_preview:
            final_source, final_preview_pages = self.answer_service.build_answer_aligned_preview_pages(
                answer_citations, selected_context_items, preferred_preview_source, grounded_pages,
            )
        else:
            final_source, final_preview_pages = None, []
        final_payload = self.answer_service.build_context_payload(
            rewritten_query, response_mode, top_score,
            final_source, final_preview_pages,
            selected_context_items, grounded_pages, answer_citations,
            preview_finalized=True,
        )
        final_payload["query_interpretation"] = query_interpretation or {}
        final_payload["answer_route"] = answer_route
        if answer_route == "extractive_code":
            final_payload["last_example_anchor"] = self._build_example_anchor(selected_context_items, query_interpretation)
        if self._should_capture_procedure_state(query_interpretation, answer_route, policy_decision):
            procedure_state = self._extract_procedure_state(final_answer)
        else:
            procedure_state = {}
        if procedure_state:
            if answer_route == "procedure_state_followup":
                final_payload["procedure_state"] = procedure_state
            else:
                final_payload["_stored_procedure_state"] = procedure_state
        return final_answer, answer_citations, final_payload

    def _build_llm_messages(
        self,
        session_id: str,
        user_message: str,
        code_example_request: bool,
        response_mode: str,
        turn_policy: dict,
        top_score: float,
        context_blocks: list[str],
        is_new_topic: bool = False,
        topic_id: str | None = None,
    ) -> list[dict]:
        """LLM에 전달할 시스템 프롬프트, 세션 메모리, 최근 대화, 문맥을 조합한 메시지 목록을 구성한다."""
        system_prompt = (
            "You are a document-grounded RAG assistant. "
            "You ONLY answer questions based on the retrieved document context provided below. "
            "If retrieved context is provided, answer from that context and mention source file names and page numbers when possible. "
            "If retrieved context is weak, missing, or irrelevant to the user's question, do NOT answer the question. "
            "Instead, respond with: '업로드된 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 해주시거나, 관련 문서를 업로드해 주세요.' "
            "Do NOT answer general knowledge questions, trivia, or anything not grounded in the retrieved context. "
            "If the user is simply reacting, acknowledging, or thanking you after a document-grounded answer, respond conversationally without reusing document citations. "
            "Do not mention unrelated prior questions or prior document topics unless the current user message explicitly asks for them. "
            "When retrieved context is used, end the answer with a short source line such as '[file.pdf] p.5' or '[file.pdf] p.5-6'. "
            "Keep answers concise but grounded. "
            "반드시 한국어로 답변하라. 사용자가 어떤 언어로 질문하더라도 항상 한국어로만 답변하라."
        )
        if code_example_request:
            system_prompt += (
                " The user is asking for code/YAML examples. "
                "ONLY include code blocks that are directly relevant to the user's specific question. "
                "Do NOT include unrelated code from the same page or nearby sections. "
                "Preserve resource names, field names, values, and command syntax exactly as written in the source. "
                "Format code blocks with proper ```yaml or ```bash fencing. "
                "Briefly explain what each code block does before showing it."
            )
        summary = self.session_repository.summary(session_id)
        prompt_memory = self._build_prompt_memory_snapshot(session_id, topic_id=topic_id)
        recent_turns = (
            self._build_prompt_recent_turns_clean(session_id, topic_id=topic_id)
            if is_new_topic
            else self._build_prompt_recent_turns(session_id, topic_id=topic_id)
        )
        context_text = self._build_prompt_context_text(context_blocks)
        return [
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

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        append_user_turn: bool = True,
    ) -> AsyncIterator[dict]:
        """사용자 메시지를 받아 검색·LLM 생성·인용 구성을 거쳐 SSE 토큰을 스트리밍한다.

        흐름: 언어 감지 → 검색 상태 준비 → 문맥 이벤트 → 응답 생성(명확화/추출/캐시/LLM) → 최종 이벤트.
        """
        # --- 언어 감지: 한글 없이 중국어/일본어가 주된 질문은 조기 차단 ---
        lang_notice = self._detect_non_korean_query(user_message)
        if lang_notice:
            yield {"type": "token", "content": lang_notice, "cached": False}
            yield {"type": "done", "cached": False}
            return

        topic_state_before = self.session_repository.topic_state(session_id)
        current_topic_id_before = str(topic_state_before.get("last_active_topic_id") or "") if isinstance(topic_state_before, dict) else ""
        session_topics_before = self.session_repository.list_topics(session_id)
        procedure_followup = None
        if not self._should_skip_procedure_shortcut(user_message, session_topics_before, current_topic_id_before or None):
            procedure_followup = self._detect_procedure_followup(
                user_message,
                topic_state_before.get("procedure_state", {}) if isinstance(topic_state_before, dict) else {},
            )
        if procedure_followup:
            built = self._build_procedure_followup_answer(
                procedure_followup,
                topic_state_before.get("procedure_state", {}),
            )
            if built is not None:
                procedure_answer, updated_procedure_state = built
                if append_user_turn:
                    self.session_repository.add_turn(session_id, "user", user_message)
                final_payload = self.answer_service.build_context_payload(
                    user_message.strip(),
                    "conversational",
                    0.0,
                    None,
                    [],
                    [],
                    [],
                    [],
                    preview_finalized=True,
                )
                final_payload["procedure_state"] = updated_procedure_state
                yield {"type": "context", **final_payload}
                yield {"type": "token", "content": procedure_answer, "cached": False}
                resolved_topic_id = topic_state_before.get("last_active_topic_id") if isinstance(topic_state_before, dict) else None
                self._store_assistant_turn(session_id, procedure_answer, final_payload, resolved_topic_id)
                yield {"type": "done", "cached": False}
                return
        if self._looks_like_step_navigation_without_state(
            user_message,
            topic_state_before.get("procedure_state", {}) if isinstance(topic_state_before, dict) else {},
        ):
            guidance_answer = (
                "어떤 절차의 몇 단계인지 조금 더 구체적으로 알려 주세요. "
                "예를 들어 `ConfigMap 생성 2단계`, `RBAC 설정 2단계`처럼 다시 적어 주시면 바로 이어서 설명하겠습니다."
            )
            if append_user_turn:
                self.session_repository.add_turn(session_id, "user", user_message)
            final_payload = self.answer_service.build_context_payload(
                user_message.strip(),
                "general",
                0.0,
                None,
                [],
                [],
                [],
                [],
                preview_finalized=True,
            )
            yield {"type": "context", **final_payload}
            yield {"type": "token", "content": guidance_answer, "cached": False}
            resolved_topic_id = topic_state_before.get("last_active_topic_id") if isinstance(topic_state_before, dict) else None
            self._store_assistant_turn(session_id, guidance_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}
            return

        turn_context = self._resolve_turn_context(session_id, user_message)
        state = self._domain_guard_state(user_message, turn_context)
        try:
            if state is None:
                state = await self._prepare_retrieval_state(session_id, user_message, allowed_source_paths)
        except EmbeddingModelUnavailableError:
            logger.exception("[Embedding] model unavailable")
            error_message = "임베딩 모델이 아직 준비되지 않았습니다. 잠시 후 다시 시도해 주세요."
            yield {"type": "token", "content": error_message, "cached": False, "error": "embedding_model_unavailable"}
            yield {"type": "done", "cached": False}
            return
        rewritten_query = state["rewritten_query"]
        top_score = state["top_score"]
        use_retrieved_context = state["use_retrieved_context"]
        grounded_pages = state["grounded_pages"]
        ordered_context_items = state.get("ordered_context_items", [])
        selected_context_items = state["selected_context_items"]
        preferred_preview_source = state["preferred_preview_source"]
        preview_pages = state["preview_pages"]
        response_mode = state.get("response_mode", "rag" if use_retrieved_context else "general")
        turn_policy = state.get("turn_policy", {})
        query_interpretation = state.get("query_interpretation", {})
        resolved_topic_id = state.get("resolved_topic_id")
        policy_decision = TurnPolicyDecision(**turn_policy) if turn_policy else TurnPolicyDecision(
            turn_type=response_mode,
            response_mode=response_mode,
            use_retrieval=use_retrieved_context,
            use_memory_rewrite=False,
            allow_preview=use_retrieved_context,
            allow_citations=use_retrieved_context,
        )
        logger.info(
            "[TurnPolicy] type=%s mode=%s use_retrieval=%s",
            policy_decision.turn_type, policy_decision.response_mode, policy_decision.use_retrieval,
        )
        answer_route = self._resolve_answer_route(query_interpretation)
        code_example_request = answer_route == "extractive_code"
        interleaved_context_items = self._interleave_context_items_by_source(selected_context_items)
        context_blocks, context_ids = self._build_context_blocks(interleaved_context_items)

        user_turn_id: int | None = None
        if append_user_turn:
            user_turn_id = self.session_repository.add_turn(session_id, "user", user_message)
            resolved_topic_id = self._ensure_topic_for_resolution(session_id, user_message, state, user_turn_id)
        cache_key = self._build_answer_cache_key(
            session_id=session_id,
            rewritten_query=self._canonical_cache_query(user_message, rewritten_query, query_interpretation),
            context_ids=context_ids,
            answer_route=answer_route,
            query_interpretation=query_interpretation,
            topic_id=resolved_topic_id,
        )
        cached_answer = self.answer_cache_repository.get(cache_key)
        if not policy_decision.allow_preview:
            preview_pages = []
            preferred_preview_source = None
        context_payload = self.answer_service.build_context_payload(
            rewritten_query, response_mode, top_score,
            preferred_preview_source, preview_pages,
            selected_context_items, grounded_pages, [],
            preview_finalized=False,
        )
        yield {"type": "context", **self._public_context_payload(context_payload)}
        if cached_answer is not None:
            full_text = self.answer_service.sanitize_answer(cached_answer["answer"], use_retrieved_context)
            streamed = ""
            for token in full_text.split(" "):
                chunk = token + " "
                streamed += chunk
                yield {"type": "token", "content": chunk, "cached": True}
            final_answer, answer_citations, final_payload = self._finalize_answer(
                streamed.strip(), rewritten_query, use_retrieved_context, top_score,
                selected_context_items, grounded_pages, preferred_preview_source,
                response_mode, policy_decision, query_interpretation, answer_route,
            )
            if final_answer != streamed.strip():
                suffix = final_answer[len(streamed.strip()):]
                if suffix:
                    yield {"type": "token", "content": suffix, "cached": True}
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": True}
            return
        if policy_decision.turn_type == "conversational_ack":
            ack_answer = self._build_policy_answer("conversational_ack", top_score)
            yield {"type": "token", "content": ack_answer, "cached": False}
            final_payload = self.answer_service.build_context_payload(
                rewritten_query, "conversational", top_score,
                None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, ack_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}
            return

        if policy_decision.turn_type == "greeting":
            greeting_answer = self._build_policy_answer("greeting", top_score)
            for char in greeting_answer:
                yield {"type": "token", "content": char, "cached": False}
                await asyncio.sleep(0.03)
            final_payload = self.answer_service.build_context_payload(
                rewritten_query, "greeting", top_score,
                None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, greeting_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}
            return

        if policy_decision.turn_type == "general_chat":
            reject_answer = self._build_policy_answer("general_chat", top_score)
            for char in reject_answer:
                yield {"type": "token", "content": char, "cached": False}
                await asyncio.sleep(0.03)
            final_payload = self.answer_service.build_context_payload(
                rewritten_query, "general", top_score,
                None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, reject_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}
            return

        if policy_decision.turn_type == "document_query" and not use_retrieved_context:
            no_result_answer = self._build_policy_answer("document_query", top_score)
            for char in no_result_answer:
                yield {"type": "token", "content": char, "cached": False}
                await asyncio.sleep(0.03)
            final_payload = self.answer_service.build_context_payload(
                rewritten_query, "general", top_score,
                None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, no_result_answer, final_payload, resolved_topic_id)
            self.answer_cache_repository.set(cache_key, {"answer": no_result_answer})
            yield {"type": "done", "cached": False}
            return

        # --- JudgeAgent: 검색 결과 적합성 판단 ---
        if use_retrieved_context and context_blocks and self._should_run_judge_agent(
            policy_decision,
            query_interpretation,
            top_score,
        ):
            judge_result = await self.judge_agent.evaluate(
                user_message, context_blocks, top_score,
            )
            logger.info(
                "[JudgeAgent] relevant=%s confidence=%s message=%r",
                judge_result["relevant"], judge_result["confidence"],
                judge_result.get("clarification_message", "")[:100],
            )
            if not judge_result["relevant"]:
                clarification = judge_result["clarification_message"]
                for char in clarification:
                    yield {"type": "token", "content": char, "cached": False}
                    await asyncio.sleep(0.03)
                final_payload = self.answer_service.build_context_payload(
                    rewritten_query, "clarification", top_score,
                    None, [], [], [], [],
                    preview_finalized=True,
                )
                yield {"type": "context", **self._public_context_payload(final_payload)}
                self._store_assistant_turn(session_id, clarification, final_payload, resolved_topic_id)
                self.answer_cache_repository.set(cache_key, {"answer": clarification})
                yield {"type": "done", "cached": False}
                return

        # --- 명확화 응답: 모호한 지시어에 대해 추가 질문으로 응답 ---
        if policy_decision.needs_clarification and policy_decision.clarification_prompt:
            clarification_answer = policy_decision.clarification_prompt.strip()
            for char in clarification_answer:
                yield {"type": "token", "content": char, "cached": False}
                await asyncio.sleep(0.03)
            final_payload = self.answer_service.build_context_payload(
                rewritten_query, response_mode, top_score,
                None, [], [], [], [],
                preview_finalized=True,
            )
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, clarification_answer, final_payload, resolved_topic_id)
            self.answer_cache_repository.set(cache_key, {"answer": clarification_answer})
            yield {"type": "done", "cached": False}
            return

        if answer_route == "extractive_code" and use_retrieved_context:
            code_context_items = self._select_code_example_context_items(
                user_message,
                query_interpretation,
                ordered_context_items,
                selected_context_items,
            )
            extractive_code_answer = self.answer_service.build_extractive_code_answer(
                code_context_items,
                requested_resource_kinds=self._resolve_requested_resource_kinds(query_interpretation),
            )
            if extractive_code_answer:
                final_answer, answer_citations, final_payload = self._finalize_answer(
                    extractive_code_answer,
                    rewritten_query,
                    use_retrieved_context,
                    top_score,
                    code_context_items,
                    grounded_pages,
                    preferred_preview_source,
                    response_mode,
                    policy_decision,
                    query_interpretation,
                    answer_route,
                )
                final_payload["last_example_anchor"] = self._build_example_anchor(
                    code_context_items,
                    query_interpretation,
                )
                yield {"type": "token", "content": final_answer, "cached": False}
                yield {"type": "context", **self._public_context_payload(final_payload)}
                self._store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
                self.answer_cache_repository.set(cache_key, {"answer": final_answer})
                yield {"type": "done", "cached": False}
                return

            no_code_answer = self._build_missing_extractive_answer(answer_route)
            yield {"type": "token", "content": no_code_answer, "cached": False}
            final_payload = self.answer_service.build_context_payload(
                rewritten_query,
                "clarification",
                top_score,
                None,
                [],
                [],
                [],
                [],
                preview_finalized=True,
            )
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, no_code_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}
            return

        if answer_route == "extractive_table" and use_retrieved_context:
            table_context_items = self._prefer_block_type_items(
                ordered_context_items or selected_context_items,
                block_type="table",
                limit=max(len(selected_context_items), 3),
            ) or selected_context_items
            extractive_table_answer = self.answer_service.build_extractive_table_answer(table_context_items)
            if extractive_table_answer:
                final_answer, answer_citations, final_payload = self._finalize_answer(
                    extractive_table_answer,
                    rewritten_query,
                    use_retrieved_context,
                    top_score,
                    table_context_items,
                    grounded_pages,
                    preferred_preview_source,
                    response_mode,
                    policy_decision,
                    query_interpretation,
                    answer_route,
                )
                yield {"type": "token", "content": final_answer, "cached": False}
                yield {"type": "context", **self._public_context_payload(final_payload)}
                self._store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
                self.answer_cache_repository.set(cache_key, {"answer": final_answer})
                yield {"type": "done", "cached": False}
                return

            no_table_answer = self._build_missing_extractive_answer(answer_route)
            yield {"type": "token", "content": no_table_answer, "cached": False}
            final_payload = self.answer_service.build_context_payload(
                rewritten_query,
                "clarification",
                top_score,
                None,
                [],
                [],
                [],
                [],
                preview_finalized=True,
            )
            yield {"type": "context", **self._public_context_payload(final_payload)}
            self._store_assistant_turn(session_id, no_table_answer, final_payload, resolved_topic_id)
            yield {"type": "done", "cached": False}
            return
        is_new_topic = not use_retrieved_context and response_mode != "rag"
        messages = self._build_llm_messages(
            session_id, user_message, code_example_request,
            response_mode, turn_policy, top_score, context_blocks,
            is_new_topic=is_new_topic, topic_id=resolved_topic_id,
        )
        parts: list[str] = []
        try:
            async for token in self.llm.stream_chat(messages):
                parts.append(token)
                yield {"type": "token", "content": token, "cached": False}
        except Exception as exc:
            fallback = self._build_llm_failure_fallback(
                user_message, use_retrieved_context, context_blocks,
                self._build_prompt_context_text(context_blocks), policy_decision,
            )
            parts = [fallback]
            yield {"type": "token", "content": fallback, "cached": False, "error": str(exc)}

        final_answer, answer_citations, final_payload = self._finalize_answer(
            "".join(parts).strip(), rewritten_query, use_retrieved_context, top_score,
            selected_context_items, grounded_pages, preferred_preview_source,
            response_mode, policy_decision, query_interpretation, answer_route,
        )
        if final_answer != "".join(parts).strip():
            suffix = final_answer[len("".join(parts).strip()):]
            if suffix:
                yield {"type": "token", "content": suffix, "cached": False}
        yield {"type": "context", **self._public_context_payload(final_payload)}
        self._store_assistant_turn(session_id, final_answer, final_payload, resolved_topic_id)
        self.answer_cache_repository.set(cache_key, {"answer": final_answer})
        logger.info(
            "[Cache] embedding %s | answer %s",
            self.embedding_cache.stats(),
            self.answer_cache.stats(),
        )
        yield {"type": "done", "cached": False}

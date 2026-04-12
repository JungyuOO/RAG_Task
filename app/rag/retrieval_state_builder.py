"""Build retrieval state for a chat turn before answer generation."""

from __future__ import annotations

import logging
import time
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

    SOURCE_FAMILY_HINTS: dict[str, tuple[str, ...]] = {
        "auth": ("oauth", "token", "identity", "provider", "ldap", "htpasswd", "authentication", "authorization", "auth", "rbac", "인증", "권한", "신원", "아이덴티티", "토큰", "흐름"),
        "network": ("mtu", "network", "networking", "advanced_networking", "ovn", "multus", "route", "ingress", "네트워크", "라우트", "인그레스"),
        "storage": ("storage", "persistent", "volume", "pv", "pvc", "ceph", "odf", "ocs", "스토리지", "볼륨"),
        "install": ("install", "installer", "bootstrap", "baremetal", "aws", "vsphere", "cluster", "설치", "클러스터"),
        "appdev": ("deploy", "deployment", "service", "route", "build", "pipeline", "gitops", "app", "project", "배포", "서비스", "프로젝트", "파이프라인"),
    }

    def __init__(self, deps: RetrievalStateDeps) -> None:
        self.deps = deps

    @staticmethod
    def _is_code_or_command_query(query_interpretation: dict | None) -> bool:
        query_interpretation = query_interpretation or {}
        intent = str(query_interpretation.get("intent") or "").casefold()
        response_shape = str(query_interpretation.get("response_shape") or "").casefold()
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        return (
            intent in {"yaml_example", "cli_example", "code_example"}
            or response_shape == "code"
            or bool(format_constraints & {"yaml", "cli"})
        )

    @classmethod
    def _build_retrieval_profile(cls, query_interpretation: dict | None) -> dict[str, float | int | bool]:
        if cls._is_code_or_command_query(query_interpretation):
            return {
                "dense_rrf_weight": 1.35,
                "sparse_rrf_weight": 0.35,
                "rerank_limit": 6,
                "skip_cross_encoder": True,
            }
        return {
            "dense_rrf_weight": 1.0,
            "sparse_rrf_weight": 1.0,
            "rerank_limit": 10,
            "skip_cross_encoder": False,
        }

    @staticmethod
    def _effective_target_versions(target_versions: list[str], document_group_preference: str) -> list[str]:
        if document_group_preference in {"official_ocp", "mixed"}:
            return []
        return list(target_versions or [])

    @staticmethod
    def _build_format_focused_queries(query_interpretation: dict | None) -> list[str]:
        query_interpretation = query_interpretation or {}
        resources = [str(value).casefold().strip() for value in query_interpretation.get("resources", []) if value]
        format_constraints = {str(value).casefold().strip() for value in query_interpretation.get("format_constraints", []) if value}
        response_shape = str(query_interpretation.get("response_shape") or "").casefold()
        queries: list[str] = []

        def _add(value: str) -> None:
            normalized = value.strip()
            if normalized and normalized not in queries:
                queries.append(normalized)

        if "yaml" in format_constraints or response_shape == "code":
            for resource in resources[:1]:
                plural = resource if resource.endswith("s") else f"{resource}s"
                _add(f"oc get {resource} -o yaml")
                _add(f"oc get {plural} -o yaml")
                _add(f"oc describe {resource}")
        elif "cli" in format_constraints:
            for resource in resources[:1]:
                plural = resource if resource.endswith("s") else f"{resource}s"
                _add(f"oc get {resource}")
                _add(f"oc get {plural}")
        return queries[:3]

    @staticmethod
    def _should_skip_expand_with_llm(policy: TurnPolicyDecision, topic_state: dict, user_message: str) -> bool:
        lowered = str(user_message or "").casefold()
        has_followup_signal = any(marker in lowered for marker in ("그 ", "그때", "그다음", "그 다음", "다시", "이어서", "이번에는", "방금", "that", "again", "continue"))
        compare_or_summary_signal = any(marker in lowered for marker in ("비교", "차이", "요약", "정리", "summary", "compare", "difference"))
        has_topic_anchor = (
            bool(topic_state.get("selected_sources"))
            or bool(topic_state.get("selected_versions"))
            or bool(topic_state.get("active_topic"))
        )
        if policy.turn_type == "document_followup":
            return has_followup_signal or compare_or_summary_signal
        if has_topic_anchor and (has_followup_signal or compare_or_summary_signal):
            return True
        return False

    @staticmethod
    def _build_followup_fast_query_result(rewritten_query: str, topic_state: dict, user_message: str) -> dict:
        lowered = str(user_message or "").casefold()
        resources: list[str] = []
        for resource in topic_state.get("last_explicit_resources", []) or []:
            normalized = str(resource).casefold().strip()
            if normalized and normalized not in resources:
                resources.append(normalized)
        anchor = topic_state.get("last_example_anchor") or {}
        anchor_resource = str(anchor.get("resource_kind") or "").casefold().strip()
        if anchor_resource and anchor_resource not in resources:
            resources.append(anchor_resource)
        last_code_resource_kind = str(topic_state.get("last_code_resource_kind") or "").casefold().strip()
        if last_code_resource_kind and last_code_resource_kind not in resources:
            resources.append(last_code_resource_kind)

        format_constraints: list[str] = []
        if any(marker in lowered for marker in ("yaml", "manifest")):
            format_constraints.append("yaml")
        if any(marker in lowered for marker in ("명령어", "command", "cli", "oc ", "kubectl")):
            format_constraints.append("cli")

        refined_terms: list[str] = []
        refined_terms.extend(resources[:2])
        if "yaml" in format_constraints:
            refined_terms.extend(["yaml", "oc", "-o yaml"])
        elif "cli" in format_constraints:
            refined_terms.append("oc")
        refined_query = " ".join(dict.fromkeys(term for term in refined_terms if term)).strip() or rewritten_query
        return {
            "refined_query": refined_query,
            "alternative_queries": [],
            "translated_keywords": [],
            "target_versions": list(topic_state.get("selected_versions", []) or []),
            "resources": resources,
            "actions": [],
            "format_constraints": format_constraints,
            "response_shape": "code" if format_constraints else "",
            "normalized_keywords": refined_terms,
        }

    @staticmethod
    def _should_skip_reranker(
        policy: TurnPolicyDecision,
        document_group_preference: str,
        source_filter_strategy: str,
        retrieved: list[dict],
    ) -> bool:
        if not retrieved:
            return True
        if document_group_preference == "official_ocp" and source_filter_strategy == "keyword_scoped":
            return True
        if document_group_preference == "mixed":
            return False
        if source_filter_strategy not in {"selected_sources", "keyword_scoped"}:
            return False
        top_sources = {
            Path(str(item["chunk"].get("source_path") or "")).name
            for item in retrieved[:4]
        }
        top_score = float(retrieved[0].get("retrieval_score", retrieved[0].get("rerank_score", 0.0)))
        if policy.turn_type == "document_followup":
            return len(top_sources) <= 3 and top_score >= 0.10
        if policy.turn_type == "document_query" and source_filter_strategy == "selected_sources":
            return len(top_sources) <= 3 and top_score >= 0.14
        if policy.turn_type == "document_query" and source_filter_strategy == "keyword_scoped":
            return len(top_sources) <= 2 and top_score >= 0.20
        return False

    @staticmethod
    def _source_group_from_path(source_path: str) -> str:
        normalized = str(source_path or "").replace("\\", "/").casefold()
        base_name = Path(str(source_path or "")).name.casefold()
        if "/generated_pdf/" in normalized or "/generated/" in normalized or "/chat_uploads/" in normalized:
            return "customer_generated"
        if "customer-guide" in base_name or "customer_guide" in base_name:
            return "customer_generated"
        return "official_ocp"

    @staticmethod
    def _is_html_single_source_path(source_path: str) -> bool:
        normalized = str(source_path or "").replace("\\", "/").casefold()
        return "/ocp-html-single-" in normalized and normalized.endswith(".md")

    def _prefer_official_html_single_sources(self, source_paths: list[str]) -> list[str]:
        html_single = [source_path for source_path in source_paths if self._is_html_single_source_path(source_path)]
        return html_single or source_paths

    def _build_uploaded_source_filter(
        self,
        indexed_source_paths: set[str],
        query_interpretation: dict,
        target_versions: list[str],
        document_group_preference: str,
        uploaded_source_paths: set[str] | None,
    ) -> tuple[list[str] | None, str]:
        uploaded = sorted({str(path) for path in (uploaded_source_paths or set()) if path})
        if not uploaded:
            return None, "none"
        if document_group_preference != "mixed":
            return uploaded, "uploaded_only"

        all_sources = sorted(indexed_source_paths)
        if document_group_preference == "official_ocp":
            all_sources = self._prefer_official_html_single_sources(all_sources)
        interesting_tokens: list[str] = []
        for value in query_interpretation.get("resources", []) or []:
            token = str(value).casefold().strip()
            if len(token) >= 2 and token not in interesting_tokens:
                interesting_tokens.append(token)
        for value in query_interpretation.get("normalized_keywords", []) or []:
            token = str(value).casefold().strip()
            min_len = 2 if any("\uac00" <= ch <= "\ud7a3" for ch in token) else 3
            if len(token) >= min_len and token not in interesting_tokens:
                interesting_tokens.append(token)

        target_version_set = {str(value).strip().casefold() for value in target_versions if value}
        scored: list[tuple[int, str]] = []
        for source_path in all_sources:
            if source_path in uploaded:
                continue
            if self._source_group_from_path(source_path) != "official_ocp":
                continue
            normalized = str(source_path).replace("\\", "/").casefold()
            basename = Path(str(source_path)).name.casefold()
            score = 0
            if target_version_set and any(version in normalized for version in target_version_set):
                score += 3
            for token in interesting_tokens:
                if token in basename:
                    score += 2
                elif token in normalized:
                    score += 1
            if score > 0:
                scored.append((score, source_path))

        scored.sort(key=lambda item: (-item[0], str(item[1])))
        official_sources = [source_path for _score, source_path in scored[:8]]
        if not official_sources and target_version_set:
            for source_path in all_sources:
                if self._source_group_from_path(source_path) != "official_ocp":
                    continue
                normalized = str(source_path).replace("\\", "/").casefold()
                if any(version in normalized for version in target_version_set):
                    official_sources.append(source_path)
            official_sources = sorted(set(official_sources))[:16]
        return sorted(set(uploaded + official_sources)), "uploaded_mixed"

    def _build_source_filter(
        self,
        indexed_source_paths: set[str],
        topic_state: dict,
        query_interpretation: dict,
        target_versions: list[str],
        document_group_preference: str,
        turn_type: str,
        allowed_source_paths: set[str] | None,
        uploaded_source_paths: set[str] | None,
    ) -> tuple[list[str] | None, str]:
        if allowed_source_paths:
            return sorted(allowed_source_paths), "allowed"
        uploaded_filter, uploaded_strategy = self._build_uploaded_source_filter(
            indexed_source_paths,
            query_interpretation,
            target_versions,
            document_group_preference,
            uploaded_source_paths,
        )
        if uploaded_filter:
            return uploaded_filter, uploaded_strategy

        all_sources = sorted(indexed_source_paths)
        if document_group_preference == "official_ocp":
            all_sources = self._prefer_official_html_single_sources(all_sources)
        selected_source_names = [
            str(source).casefold().strip()
            for source in topic_state.get("selected_sources", []) or []
            if source
        ]
        if selected_source_names:
            matched: list[str] = []
            for selected_name in selected_source_names:
                for source_path in all_sources:
                    if Path(str(source_path)).name.casefold() == selected_name and source_path not in matched:
                        matched.append(source_path)
            if document_group_preference == "customer_generated":
                matched = [source_path for source_path in matched if str(source_path).replace("\\", "/").lower().endswith(".pdf")]
            if matched:
                if turn_type == "document_followup" and document_group_preference != "mixed":
                    return matched[:1], "selected_sources_followup"
                return matched[:8], "selected_sources"

        interesting_tokens: list[str] = []
        for value in query_interpretation.get("resources", []) or []:
            token = str(value).casefold().strip()
            if len(token) >= 2 and token not in interesting_tokens:
                interesting_tokens.append(token)
        for value in query_interpretation.get("normalized_keywords", []) or []:
            token = str(value).casefold().strip()
            min_len = 2 if any("\uac00" <= ch <= "\ud7a3" for ch in token) else 3
            if len(token) >= min_len and token not in interesting_tokens:
                interesting_tokens.append(token)

        matched_families = [
            family
            for family, hints in self.SOURCE_FAMILY_HINTS.items()
            if any(token in hints for token in interesting_tokens)
        ]

        scored: list[tuple[int, str]] = []
        target_version_set = {str(value).strip() for value in target_versions if value}
        for source_path in all_sources:
            group = self._source_group_from_path(source_path)
            if document_group_preference in {"customer_generated", "official_ocp"} and group != document_group_preference:
                continue
            normalized = str(source_path).replace("\\", "/").casefold()
            if document_group_preference == "customer_generated" and not normalized.endswith(".pdf"):
                continue
            basename = Path(str(source_path)).name.casefold()
            score = 0
            if target_version_set and any(version.casefold() in normalized for version in target_version_set):
                score += 3
            for family in matched_families:
                if any(hint in basename or hint in normalized for hint in self.SOURCE_FAMILY_HINTS[family]):
                    score += 4
            for token in interesting_tokens:
                if token in basename:
                    score += 2
                elif token in normalized:
                    score += 1
            if score > 0:
                scored.append((score, source_path))

        scored.sort(key=lambda item: (-item[0], str(item[1])))
        if scored:
            limit = 4 if matched_families else 8
            return [source_path for _score, source_path in scored[:limit]], "keyword_scoped"
        return None, "broad"

    def _scoped_topic_state_for_query(self, topic_state: dict, document_group_preference: str) -> dict:
        if not topic_state:
            return {}
        scoped = dict(topic_state)
        scoped["active_document_group"] = document_group_preference
        if document_group_preference not in {"customer_generated", "official_ocp"}:
            return scoped
        selected_sources = []
        for source in topic_state.get("selected_sources", []) or []:
            if self._source_group_from_path(str(source)) == document_group_preference:
                selected_sources.append(source)
        scoped["selected_sources"] = selected_sources
        return scoped

    @staticmethod
    def _candidate_prefilter_limit(retriever, settings) -> int:
        explicit_limit = int(getattr(settings, "pgvector_prefilter_limit", 0) or 0)
        if explicit_limit > 0:
            return explicit_limit
        return max(int(getattr(retriever, "candidate_pool_size", 15)) * 4, int(getattr(retriever, "top_k", 5)) * 6, 40)

    @staticmethod
    def _candidate_prefilter_min_results(retriever) -> int:
        return max(int(getattr(retriever, "candidate_pool_size", 15)), int(getattr(retriever, "top_k", 5)) * 3, 20)

    @staticmethod
    def _strong_query_tokens_for_code(user_message: str, query_interpretation: dict | None) -> list[str]:
        qi = query_interpretation or {}
        tokens: list[str] = []
        for token in qi.get("normalized_keywords", tokenize(user_message)) or []:
            normalized = str(token).casefold().strip()
            if len(normalized) < 4:
                continue
            if normalized in {"namespace", "project", "status", "command", "yaml", "pod"}:
                continue
            if normalized not in tokens:
                tokens.append(normalized)
        return tokens

    @classmethod
    def _select_procedure_token_matches(
        cls,
        candidate_items: list[dict],
        *,
        user_message: str,
        query_interpretation: dict | None,
        limit: int,
    ) -> list[dict]:
        strong_tokens = cls._strong_query_tokens_for_code(user_message, query_interpretation)
        if not strong_tokens:
            return []
        matched: list[tuple[int, dict]] = []
        seen: set[str] = set()
        for item in candidate_items:
            chunk = item.get("chunk") or {}
            metadata = chunk.get("metadata") or {}
            text_haystack = str(chunk.get("text") or "").casefold()
            structure_haystack = " ".join(
                [
                    str(metadata.get("section_title") or "").casefold(),
                    str(metadata.get("section_path") or "").casefold(),
                ]
            )
            text_hits = sum(1 for token in strong_tokens if token in text_haystack)
            structure_hits = sum(1 for token in strong_tokens if token in structure_haystack)
            token_score = text_hits * 10 + structure_hits
            if token_score <= 0:
                continue
            block_types = {value.strip().casefold() for value in str(metadata.get("block_types", "")).split(",") if value.strip()}
            if "code" not in block_types and not metadata.get("is_procedure"):
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            matched.append((token_score, item))
        matched.sort(
            key=lambda entry: (
                -entry[0],
                -float(entry[1].get("final_retrieval_score", entry[1].get("rerank_score", 0.0))),
            )
        )
        return [item for _, item in matched[:limit]]

    @staticmethod
    def _merge_candidate_pools(primary: list[dict], fallback: list[dict], *, limit: int) -> list[dict]:
        merged: list[dict] = []
        seen_chunk_ids: set[str] = set()
        for item in primary + fallback:
            chunk_id = str(item.get("chunk", {}).get("chunk_id") or "")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def _load_index_candidate_pool(
        self,
        deps: RetrievalStateDeps,
        *,
        query_vector: list[float],
        source_filter: list[str] | None,
        target_versions: list[str],
        doc_type: str | None,
        document_group_preference: str,
    ) -> tuple[list[dict], str]:
        search_dense = getattr(deps.index_repository, "search_dense_candidates", None)
        if callable(search_dense):
            try:
                dense_items = search_dense(
                    query_vector,
                    limit=self._candidate_prefilter_limit(deps.retriever, deps.settings),
                    source_paths=source_filter,
                    target_versions=target_versions or None,
                    doc_type=doc_type,
                    document_group_preference=document_group_preference,
                )
                if dense_items:
                    min_results = self._candidate_prefilter_min_results(deps.retriever)
                    if len(dense_items) >= min_results:
                        return dense_items, "pgvector_dense"
                    fallback_items = deps.index_repository.load(
                        source_paths=source_filter,
                        target_versions=target_versions or None,
                        doc_type=doc_type,
                        document_group_preference=document_group_preference,
                    )
                    merged = self._merge_candidate_pools(
                        dense_items,
                        fallback_items,
                        limit=max(len(dense_items), min_results),
                    )
                    return merged, "pgvector_dense_supplemented"
            except Exception as exc:
                logger.warning("[RetrievalStateBuilder] pgvector dense prefilter failed: %s", exc)

        return (
            deps.index_repository.load(
                source_paths=source_filter,
                target_versions=target_versions or None,
                doc_type=doc_type,
                document_group_preference=document_group_preference,
            ),
            "full_load",
        )

    async def run(
        self,
        session_id: str,
        user_message: str,
        allowed_source_paths: set[str] | None = None,
        *,
        uploaded_source_paths: set[str] | None = None,
        version_tag: str | None = None,
        turn_context: dict | None = None,
    ) -> dict:
        deps = self.deps
        t_total = time.perf_counter()
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

        t_source_catalog = time.perf_counter()
        indexed_source_paths = deps.index_repository.get_indexed_source_paths()
        all_sources = sorted(indexed_source_paths)
        logger.info(
            "[Timing][RetrievalStateBuilder.run] source_catalog=%.3fs sources=%d",
            time.perf_counter() - t_source_catalog,
            len(all_sources),
        )

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

        t_retrieval_expand = time.perf_counter()
        if self._should_skip_expand_with_llm(policy, topic_state, user_message):
            query_result = self._build_followup_fast_query_result(rewritten_query, topic_state, user_message)
            expand_strategy = "followup_fastpath"
        else:
            query_result = await deps.retrieval_agent.expand(
                rewritten_query,
                intent_result=intent_result,
                available_sources=all_sources,
            )
            expand_strategy = "llm_or_agent"
        logger.info(
            "[Timing][RetrievalStateBuilder.run] retrieval_agent_expand=%.3fs strategy=%s alternatives=%d target_versions=%s",
            time.perf_counter() - t_retrieval_expand,
            expand_strategy,
            len(query_result.get("alternative_queries", []) or query_result.get("alternatives", []) or []),
            query_result.get("target_versions", []),
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
        target_versions = [version_tag] if version_tag else list(query_result.get("target_versions", []) or topic_state.get("selected_versions", []) or [])
        document_group_preference = str(query_interpretation.get("document_group_preference") or "").strip() or "auto"
        if document_group_preference == "auto":
            document_group_preference = str(
                topic_state.get("last_document_group_preference")
                or topic_state.get("active_document_group")
                or "official_ocp"
            )
        effective_target_versions = self._effective_target_versions(target_versions, document_group_preference)
        retrieval_profile = self._build_retrieval_profile(query_interpretation)
        logger.info(
            "[QueryInterpretation] intent=%s resources=%s actions=%s formats=%s shape=%s keywords=%s target_versions=%s effective_target_versions=%s document_group=%s",
            query_interpretation["intent"],
            query_interpretation["resources"],
            query_interpretation["actions"],
            query_interpretation["format_constraints"],
            query_interpretation["response_shape"],
            query_interpretation["normalized_keywords"],
            target_versions,
            effective_target_versions,
            document_group_preference,
        )

        source_filter, source_filter_strategy = self._build_source_filter(
            indexed_source_paths,
            topic_state,
            query_interpretation,
            effective_target_versions,
            document_group_preference,
            policy.turn_type,
            allowed_source_paths,
            uploaded_source_paths,
        )
        scoped_topic_state = self._scoped_topic_state_for_query(topic_state, document_group_preference)
        aliased_query = deps.expand_query_with_resource_aliases(refined_query, query_interpretation)
        expanded_query = deps.expand_query_with_context(aliased_query, scoped_topic_state)
        expanded_query = self._expand_short_resource_query(
            expanded_query,
            user_message,
            query_interpretation,
        )
        t_embed = time.perf_counter()
        query_vector = deps.embedder.encode(expanded_query)
        logger.info(
            "[Timing][RetrievalStateBuilder.run] embed=%.3fs query_len=%d",
            time.perf_counter() - t_embed,
            len(expanded_query),
        )
        t_index_load = time.perf_counter()
        index_items_all, index_load_strategy = self._load_index_candidate_pool(
            deps,
            query_vector=query_vector,
            source_filter=source_filter,
            target_versions=effective_target_versions,
            doc_type=doc_type,
            document_group_preference=document_group_preference,
        )
        if source_filter and not index_items_all:
            source_filter = None
            source_filter_strategy = "broad_fallback"
            index_items_all, index_load_strategy = self._load_index_candidate_pool(
                deps,
                query_vector=query_vector,
                source_filter=None,
                target_versions=effective_target_versions,
                doc_type=doc_type,
                document_group_preference=document_group_preference,
            )
        logger.info(
            "[Timing][RetrievalStateBuilder.run] index_load=%.3fs items=%d target_versions=%s source_filter=%d source_strategy=%s load_strategy=%s doc_type=%s document_group=%s",
            time.perf_counter() - t_index_load,
            len(index_items_all),
            effective_target_versions,
            len(source_filter or []),
            source_filter_strategy,
            index_load_strategy,
            doc_type or "",
            document_group_preference,
        )
        t_filter_index = time.perf_counter()
        index_items = deps.retrieval_service.filter_index_items(
            index_items_all,
            allowed_source_paths,
            uploaded_source_paths=uploaded_source_paths,
            doc_type=doc_type,
            document_group_preference=document_group_preference,
        )
        if source_filter and not index_items and source_filter_strategy in {"selected_sources", "keyword_scoped"}:
            source_filter = None
            source_filter_strategy = "broad_post_filter_fallback"
            index_items_all = deps.index_repository.load(
                source_paths=None,
                target_versions=effective_target_versions or None,
                doc_type=doc_type,
                document_group_preference=document_group_preference,
            )
            index_items = deps.retrieval_service.filter_index_items(
                index_items_all,
                allowed_source_paths,
                uploaded_source_paths=uploaded_source_paths,
                doc_type=doc_type,
                document_group_preference=document_group_preference,
            )
        logger.info(
            "[Timing][RetrievalStateBuilder.run] filter_index_items=%.3fs before=%d after=%d doc_type=%s document_group=%s",
            time.perf_counter() - t_filter_index,
            len(index_items_all),
            len(index_items),
            doc_type or "",
            document_group_preference,
        )

        # doc_type 필터 적용 후 청크가 0개인 경우 — 해당 문서가 아직 인덱싱되지 않은 것
        no_doc_type_docs = bool(doc_type) and len(index_items) == 0 and bool(all_sources)

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
        t_retrieve_fast = time.perf_counter()
        base_rrf_pool = deps.retriever.search_rrf(
            expanded_query, query_vector, index_items, rrf_k=rrf_k,
            limit=deps.retriever.candidate_pool_size,
            target_versions=effective_target_versions,
            keyword_query=bm25_keyword_query,
            dense_weight=float(retrieval_profile["dense_rrf_weight"]),
            sparse_weight=float(retrieval_profile["sparse_rrf_weight"]),
        )
        retrieved = base_rrf_pool[: deps.retriever.top_k]
        logger.info(
            "[Timing][RetrievalStateBuilder.run] retrieve_fast=%.3fs pool=%d top_k=%d",
            time.perf_counter() - t_retrieve_fast,
            len(base_rrf_pool),
            len(retrieved),
        )

        query_interpretation_dict = dict(query_interpretation)
        query_interpretation_dict["target_versions"] = target_versions
        query_interpretation_dict["document_group_preference"] = document_group_preference

        if not query_interpretation["resources"] and topic_state.get("last_explicit_resources"):
            inherited = topic_state["last_explicit_resources"][:2]
            query_interpretation_dict["resources"] = inherited
            logger.info("[FollowupAnchor] inherited resources=%s", inherited)

        selected_source_names = {
            str(source).casefold().strip()
            for source in topic_state.get("selected_sources", []) or []
            if source
        }
        has_selected_source_hit = any(
            Path(str(item["chunk"]["source_path"] or "")).name.casefold() in selected_source_names
            for item in retrieved
        )
        lowered_shape = str(query_interpretation["response_shape"] or "").casefold()
        lowered_intent = str(query_interpretation["intent"] or "").casefold()
        should_run_selected_source_pass = (
            bool(selected_source_names)
            and not has_selected_source_hit
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
                    target_versions=effective_target_versions,
                    keyword_query=bm25_keyword_query,
                    dense_weight=float(retrieval_profile["dense_rrf_weight"]),
                    sparse_weight=float(retrieval_profile["sparse_rrf_weight"]),
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
        alt_query_passes = int(getattr(deps.settings, "retrieval_alternative_query_max_passes", 0))
        if not retrieved and alt_query_passes > 0:
            for alt_query in alternative_queries[:alt_query_passes]:
                alt_aliased = deps.expand_query_with_resource_aliases(alt_query, query_interpretation_dict)
                alt_expanded = deps.expand_query_with_context(alt_aliased, scoped_topic_state)
                alt_vector = deps.embedder.encode(alt_expanded)
                alt_bm25_kw = self._build_bm25_keyword_query(alt_query)
                alt_retrieved = deps.retriever.search_rrf(
                    alt_expanded,
                    alt_vector,
                    index_items,
                    rrf_k=rrf_k,
                    target_versions=effective_target_versions,
                    keyword_query=alt_bm25_kw,
                    dense_weight=float(retrieval_profile["dense_rrf_weight"]),
                    sparse_weight=float(retrieval_profile["sparse_rrf_weight"]),
                )
                for item in alt_retrieved:
                    cid = item["chunk"]["chunk_id"]
                    if cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        merged_extras.append(item)
        if merged_extras:
            retrieved = retrieved + merged_extras

        focused_queries = self._build_format_focused_queries(query_interpretation_dict)
        if focused_queries:
            for focused_query in focused_queries:
                focused_vector = deps.embedder.encode(focused_query)
                focused_items = deps.retriever.search_rrf(
                    focused_query,
                    focused_vector,
                    index_items,
                    rrf_k=rrf_k,
                    target_versions=effective_target_versions,
                    keyword_query=focused_query,
                    dense_weight=float(retrieval_profile["dense_rrf_weight"]),
                    sparse_weight=float(retrieval_profile["sparse_rrf_weight"]),
                )
                for item in focused_items:
                    cid = item["chunk"]["chunk_id"]
                    if cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        retrieved.append(item)

        for index, item in enumerate(retrieved[:5]):
            chunk = item["chunk"]
            logger.info(
                "[Retrieval] #%d %s p.%s | rerank=%.4f dense=%.4f sparse=%.4f",
                index + 1,
                Path(chunk["source_path"]).name,
                chunk.get("page_number", "?"),
                item.get("retrieval_score", item.get("rerank_score", 0)),
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
            extended.sort(key=lambda item: item.get("retrieval_score", item.get("rerank_score", 0)), reverse=True)
            rerank_limit = int(retrieval_profile["rerank_limit"])
            if policy.turn_type == "document_followup" and document_group_preference != "mixed":
                rerank_limit = min(rerank_limit, 3)
            extended = extended[:rerank_limit]
            lowered_intent = str(query_interpretation_dict.get("intent") or "").casefold()
            should_skip_cross_encoder = bool(retrieval_profile["skip_cross_encoder"]) or lowered_intent in {"yaml_example", "cli_example", "code_example"}
            if should_skip_cross_encoder or self._should_skip_reranker(policy, document_group_preference, source_filter_strategy, retrieved):
                logger.info(
                    "[Retrieval] skip_reranker turn_type=%s intent=%s document_group=%s source_strategy=%s top_score=%.4f profile=%s",
                    policy.turn_type,
                    lowered_intent,
                    document_group_preference,
                    source_filter_strategy,
                    float(retrieved[0].get("retrieval_score", retrieved[0].get("rerank_score", 0.0))),
                    retrieval_profile,
                )
                retrieved = extended
            else:
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
            min_score=getattr(deps.settings, "retrieval_gate_threshold", deps.settings.retrieval_min_score),
            score_field="final_retrieval_score",
        )
        top_score = retrieval_metrics["top_score"]
        use_retrieved_context = deps.should_use_retrieved_context(
            policy,
            retrieved,
            top_score,
            query_interpretation_dict,
        )
        logger.info(
            "[Retrieval] top_score=%.4f use_context=%s gate_threshold=%.4f",
            top_score,
            use_retrieved_context,
            getattr(deps.settings, "retrieval_gate_threshold", deps.settings.retrieval_min_score),
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
        selected_context_limit = min(max(int(deps.settings.grounded_chunk_top_n), 1), 3)
        selected_context_items = deps.retrieval_service.rebalance_context_items_by_document_group(
            selected_context_items,
            query_interpretation_dict.get("document_group_preference"),
            limit=selected_context_limit,
        )
        procedure_token_matches = self._select_procedure_token_matches(
            ordered_context_items,
            user_message=user_message,
            query_interpretation=query_interpretation_dict,
            limit=selected_context_limit,
        )
        if procedure_token_matches:
            selected_context_items = procedure_token_matches
        elif self._strong_query_tokens_for_code(user_message, query_interpretation_dict):
            strong_token_fallback = self._select_procedure_token_matches(
                index_items_all,
                user_message=user_message,
                query_interpretation=query_interpretation_dict,
                limit=selected_context_limit,
            )
            if strong_token_fallback:
                selected_context_items = strong_token_fallback
        if not selected_context_items and str(query_interpretation["response_shape"] or "").casefold() == "code":
            fallback_code_items = deps.find_fallback_code_context_items(
                user_message,
                query_interpretation_dict,
                index_items,
                topic_state,
            )
            if fallback_code_items:
                selected_context_items = fallback_code_items[:selected_context_limit]
                ordered_context_items = fallback_code_items[:selected_context_limit]
                grounded_pages = deps.retrieval_service.aggregate_page_grounding(fallback_code_items)
                top_score = max(
                    top_score,
                    max(float(item.get("final_retrieval_score", item.get("rerank_score", 0.0))) for item in fallback_code_items),
                )
                use_retrieved_context = True
        preferred_preview_source = deps.retrieval_service.select_grounded_preview_source(grounded_pages)
        preview_pages = deps.retrieval_service.build_grounded_preview_pages(
            preferred_preview_source,
            grounded_pages,
        )
        logger.info(
            "[Timing][RetrievalStateBuilder.run] total=%.3fs target_versions=%s selected_context=%d",
            time.perf_counter() - t_total,
            target_versions,
            len(selected_context_items),
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
            "no_doc_type_docs": no_doc_type_docs,
            "index_load_strategy": index_load_strategy,
            "source_filter_strategy": source_filter_strategy,
            "index_items_loaded": len(index_items_all),
            "index_items_filtered": len(index_items),
            "candidate_pool_size": len(base_rrf_pool),
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
        "뭐야", "무엇", "어떻게", "왜", "설명", "설명해줘", "알려줘", "개념", "역할", "특징",
        "동작", "원리", "구성", "요소", "방법", "대해", "대해서", "알고싶어", "알고", "차이", "비교",
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

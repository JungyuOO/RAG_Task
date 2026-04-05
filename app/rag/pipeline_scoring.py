from __future__ import annotations

import re
from pathlib import Path

from app.rag.utils import normalize_text, tokenize

_KIND_RE = re.compile(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$")


def _qi_str(qi: dict, key: str) -> str:
    return str(qi.get(key, "") or "").casefold()


def _qi_set(qi: dict, key: str) -> set[str]:
    return {str(v).casefold() for v in qi.get(key, []) if v}


class PipelineRetrievalMixin:
    def _metadata_aware_score(self, user_message: str, query_interpretation: dict | None, item: dict) -> dict:
        query_interpretation = query_interpretation or {}
        metadata = item["chunk"].get("metadata", {})
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        block_types = {value.strip().casefold() for value in str(metadata.get("block_types", "")).split(",") if value.strip()}
        code_language = str(metadata.get("code_language", "")).casefold()
        code_subtype = str(metadata.get("code_subtype", "")).casefold()
        code_signals = {str(signal).casefold() for signal in metadata.get("code_signals", []) or []}
        explicit_kind_match = _KIND_RE.search(lowered_text)
        explicit_resource_kind = explicit_kind_match.group(1).casefold() if explicit_kind_match else ""
        query_tokens = {token for token in query_interpretation.get("normalized_keywords", tokenize(user_message)) if len(token) >= 2 and token not in {"yaml", "manifest", "code", "example", "sample", "demo"}}
        resources = self._resolve_requested_resource_kinds(query_interpretation)
        actions = _qi_set(query_interpretation, "actions")
        format_constraints = _qi_set(query_interpretation, "format_constraints")
        response_shape = _qi_str(query_interpretation, "response_shape")
        intent = _qi_str(query_interpretation, "intent")

        heading_score = self._heading_overlap_score(user_message, metadata)
        resource_score = action_score = format_score = shape_score = lexical_score = completeness_score = 0.0
        focus_multiplier = 1.0
        is_single_resource_focus = len(resources) == 1 and (intent == "explain" or response_shape == "text")

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
                parent_headings = " ".join(str(value).casefold() for value in metadata.get("parent_headings", []) or [])
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
                if resource:
                    focus_multiplier = self._compute_focus_multiplier(resource, resources, lowered_text, section_focus_text, focus_multiplier)

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

    def _compute_focus_multiplier(self, target_resource: str, all_requested_resources: set[str], lowered_text: str, section_focus_text: str, current_multiplier: float) -> float:
        if target_resource in section_focus_text.split():
            return max(current_multiplier, 0.95)
        target_count = lowered_text.count(target_resource)
        if target_count == 0:
            return current_multiplier
        sibling_count = 0
        for resource in all_requested_resources:
            if resource != target_resource:
                sibling_count += lowered_text.count(resource)
        total_mentions = target_count + sibling_count
        if total_mentions == 0:
            return current_multiplier
        focus_ratio = target_count / total_mentions
        if focus_ratio >= 0.6:
            return max(current_multiplier, 0.95)
        if focus_ratio >= 0.3:
            return min(current_multiplier, 0.85)
        return min(current_multiplier, 0.65)

    def _code_completeness_score(self, lowered_text: str) -> float:
        field_lines = re.findall(r"(?im)^\s*([a-z][a-z0-9_-]*)\s*:", lowered_text)
        unique_fields = {field.casefold() for field in field_lines}
        if not unique_fields:
            return 0.0
        return min(len(unique_fields) * 0.05, 0.35)

    def _metadata_aware_rerank(self, user_message: str, query_interpretation: dict | None, items: list[dict]) -> list[dict]:
        if not items:
            return []
        rescored = [{**item, **self._metadata_aware_score(user_message, query_interpretation, item)} for item in items]
        rescored.sort(key=lambda item: (-float(item.get("metadata_final_score", 0.0)), -float(item.get("metadata_score", 0.0)), -float(item.get("rerank_score", 0.0))))
        return rescored

    def _has_code_content(self, item: dict) -> bool:
        text = str(item["chunk"].get("text", "") or "")
        if not text.strip():
            return False
        return len(self.answer_service._extract_code_candidates(text)) > 0

    def _select_code_example_context_items(self, user_message: str, query_interpretation: dict | None, ordered_context_items: list[dict], selected_context_items: list[dict]) -> list[dict]:
        candidates = ordered_context_items or selected_context_items
        code_candidates = self._prefer_block_type_items(candidates, block_type="code", limit=None)
        if not code_candidates:
            code_candidates = [item for item in candidates if self._has_code_content(item)]
        if not code_candidates:
            return selected_context_items

        requested_resource_kinds = self._resolve_requested_resource_kinds(query_interpretation)
        if requested_resource_kinds:
            explicit_kind_matches = [item for item in code_candidates if self._extract_explicit_resource_kind(item) in requested_resource_kinds]
            if explicit_kind_matches:
                code_candidates = explicit_kind_matches
            exact_resource_matches = [item for item in code_candidates if self._infer_item_resource_kinds(item) & requested_resource_kinds]
            if exact_resource_matches:
                code_candidates = exact_resource_matches

        rescored = self._metadata_aware_rerank(user_message, query_interpretation, code_candidates)
        has_positive_match = any(item.get("resource_match_score", 0) > 0 for item in rescored)
        if has_positive_match:
            rescored = [item for item in rescored if item.get("resource_match_score", 0) >= 0]
        for item in rescored:
            item["code_selection_score"] = float(item.get("metadata_final_score", 0.0))
            item["code_intent_score"] = float(item.get("resource_match_score", 0.0)) + float(item.get("action_match_score", 0.0)) + float(item.get("format_match_score", 0.0)) + float(item.get("shape_match_score", 0.0)) + float(item.get("lexical_match_score", 0.0))
        return rescored

    def _extract_explicit_resource_kind(self, item: dict) -> str:
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        explicit_kind_match = _KIND_RE.search(lowered_text)
        return explicit_kind_match.group(1).casefold() if explicit_kind_match else ""

    def _should_expand_local_context(self, query_interpretation: dict | None) -> bool:
        qi = query_interpretation or {}
        return _qi_str(qi, "intent") in {"yaml_example", "cli_example", "code_example", "table"} or _qi_str(qi, "response_shape") in {"code", "table"}

    def _expand_local_context_items(self, user_message: str, query_interpretation: dict | None, index_items: list[dict], ranked_items: list[dict]) -> list[dict]:
        if not ranked_items or not self._should_expand_local_context(query_interpretation):
            return ranked_items
        anchor_items = ranked_items[:2]
        anchors: list[dict] = []
        for item in anchor_items:
            chunk = item["chunk"]
            metadata = chunk.get("metadata", {})
            section_path = str(metadata.get("section_path", "") or "")
            section_prefix = section_path.split(">", 1)[0].strip().casefold() if section_path else ""
            anchors.append({"source_path": chunk["source_path"], "page_number": int(chunk.get("page_number") or metadata.get("page_start") or 0), "section_prefix": section_prefix})

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
            expanded.append({"chunk": chunk, "dense_score": 0.0, "sparse_score": 0.0, "rerank_score": 0.0})
            seen_chunk_ids.add(chunk_id)
        return self._metadata_aware_rerank(user_message, query_interpretation, expanded)

    def _expand_topic_anchor_context_items(self, user_message: str, query_interpretation: dict | None, index_items: list[dict], ranked_items: list[dict], topic_state: dict) -> list[dict]:
        query_interpretation = query_interpretation or {}
        topic_state = topic_state or {}
        if not index_items:
            return ranked_items
        response_shape = _qi_str(query_interpretation, "response_shape")
        intent = _qi_str(query_interpretation, "intent")
        format_constraints = _qi_set(query_interpretation, "format_constraints")
        normalized_user = normalize_text(user_message).lower()
        followup_markers = ("그거", "그건", "그 문서", "그 페이지", "그 yaml", "그 코드", "그 타입", "타입", "종류", "특징", "자세히", "더 설명", "설치", "과정", "구성", "차이", "비교", "다음", "계속", "that", "this", "those", "again", "next", "continue")
        text_followup_expansion = intent in {"explain", "compare"} and bool(topic_state.get("selected_sources")) and (any(marker in normalized_user for marker in followup_markers) or len(normalized_user) <= 28)
        if response_shape not in {"code", "table"} and not format_constraints.intersection({"yaml", "cli", "table"}) and not text_followup_expansion:
            return ranked_items

        anchor_pages = {int(page) for page in topic_state.get("selected_pages", []) if str(page).isdigit()}
        anchor_pages.update(int(page) for page in topic_state.get("last_example_source_pages", []) if str(page).isdigit())
        anchor_pages.update(int(item.get("page_number")) for item in topic_state.get("last_answer_citations", []) or [] if str(item.get("page_number", "")).isdigit())
        anchor_sections = {str(path).casefold().strip() for path in topic_state.get("last_grounded_section_paths", []) if path}
        anchor_sources = {str(source).casefold().strip() for source in topic_state.get("selected_sources", []) if source}
        if not anchor_pages and not anchor_sections and not anchor_sources:
            return ranked_items

        seen_chunk_ids = {item["chunk"]["chunk_id"] for item in ranked_items}
        source_anchor_scores: dict[str, float] = {}
        for item in ranked_items:
            source_path = str(item["chunk"].get("source_path") or "")
            source_anchor_scores[source_path] = max(source_anchor_scores.get(source_path, 0.0), float(item.get("rerank_score", 0.0)))

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
                same_section = bool(section_path and any(section_path.startswith(anchor_section) or anchor_section.startswith(section_path) for anchor_section in anchor_sections))
            if not same_page_band and not same_section:
                continue
            continuity_score = source_anchor_scores.get(str(chunk.get("source_path") or ""), 0.0) * 0.35 if text_followup_expansion else 0.0
            expanded.append({"chunk": chunk, "dense_score": 0.0, "sparse_score": 0.0, "rerank_score": continuity_score})
            seen_chunk_ids.add(chunk_id)
        return self._metadata_aware_rerank(user_message, query_interpretation, expanded)

    def _find_fallback_code_context_items(self, user_message: str, query_interpretation: dict | None, index_items: list[dict], topic_state: dict | None) -> list[dict]:
        query_interpretation = query_interpretation or {}
        topic_state = topic_state or {}
        if _qi_str(query_interpretation, "response_shape") != "code":
            return []
        selected_source_names = {str(source).casefold().strip() for source in topic_state.get("selected_sources", []) if source}
        candidate_pool: list[dict] = []
        for item in index_items:
            chunk = item["chunk"]
            source_name = Path(str(chunk.get("source_path") or "")).name.casefold()
            if selected_source_names and source_name not in selected_source_names:
                continue
            if self._has_code_content(item) or "code" in str(chunk.get("metadata", {}).get("block_types", "")).split(","):
                candidate_pool.append({"chunk": chunk, "rerank_score": float(item.get("rerank_score", 0.0)), "dense_score": float(item.get("dense_score", 0.0)), "sparse_score": float(item.get("sparse_score", 0.0)), "score": float(item.get("score", 0.0))})
        if not candidate_pool:
            return []
        selected = self._select_code_example_context_items(user_message, query_interpretation, candidate_pool, candidate_pool)
        return selected[: max(int(self.settings.grounded_chunk_top_n), 1)]

    def _apply_precision_filter(self, items: list[dict], query_interpretation: dict | None) -> list[dict]:
        if not items:
            return items
        query_interpretation = query_interpretation or {}
        resources = _qi_set(query_interpretation, "resources")
        if resources:
            has_positive = any(item.get("resource_match_score", 0) > 0 for item in items)
            if has_positive:
                items = [item for item in items if item.get("resource_match_score", 0) >= 0]
        if len(items) > 1:
            top_score = max(float(item.get("metadata_final_score", 0)) for item in items)
            if top_score > 0:
                threshold = top_score * 0.2
                items = [item for item in items if float(item.get("metadata_final_score", 0)) >= threshold]
        return items

    def _apply_focus_filter(self, items: list[dict], query_interpretation: dict | None) -> list[dict]:
        if not items:
            return items
        qi = query_interpretation or {}
        resources = [str(value).casefold() for value in qi.get("resources", []) if value]
        intent = _qi_str(qi, "intent")
        response_shape = _qi_str(qi, "response_shape")
        if len(resources) != 1 or (intent != "explain" and response_shape != "text"):
            return items
        focused = [item for item in items if item.get("focus_multiplier", 1.0) >= 0.85]
        sibling = [item for item in items if item.get("focus_multiplier", 1.0) < 0.85]
        if focused and len(sibling) / max(len(items), 1) >= 0.3:
            items = focused
        return items[:4]

    def _resolve_answer_route(self, query_interpretation: dict | None) -> str:
        query_interpretation = query_interpretation or {}
        intent = _qi_str(query_interpretation, "intent")
        response_shape = _qi_str(query_interpretation, "response_shape")
        if intent in {"yaml_example", "cli_example", "code_example"} or response_shape == "code":
            return "extractive_code"
        if intent == "table" or response_shape == "table":
            return "extractive_table"
        return "grounded_generation"

    def _build_missing_extractive_answer(self, answer_route: str) -> str:
        if answer_route == "extractive_code":
            return "업로드한 문서에서 요청하신 YAML/코드 예시를 직접 찾지 못했습니다. 문서에 실제 예시 블록이 있는지 다시 확인하시거나, 더 구체적인 범위나 페이지를 지정해 주세요."
        if answer_route == "extractive_table":
            return "업로드한 문서에서 요청하신 표나 비교 정보를 직접 찾지 못했습니다. 키워드를 조금 더 구체적으로 적어 다시 질문해 주세요."
        return "업로드한 문서에서 관련 내용을 찾을 수 없습니다."

    def _build_policy_answer(self, turn_type: str, top_score: float) -> str:
        if turn_type == "conversational_ack":
            return "네. 문서와 관련된 질문이 있으시면 이어서 질문해 주세요."
        if turn_type == "greeting":
            return "안녕하세요! 무엇을 도와드릴까요?"
        if turn_type == "general_chat":
            return "죄송합니다. 업로드한 문서와 관련된 질문만 답변할 수 있습니다. 문서 내용에 대해 질문해 주세요."
        if turn_type == "document_query":
            if top_score >= self.settings.retrieval_retry_min_score:
                return "관련 내용을 찾기 어렵습니다. 질문을 조금 더 구체적으로 적어 주세요.\n예: `스토리지 문서에서 PV 설명해줘`, `Service 종류를 서로 비교해줘`"
            return "업로드한 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 하시거나 관련 문서를 업로드해 주세요."
        return "업로드한 문서에서 관련 내용을 찾을 수 없습니다."

    def _should_use_retrieved_context(self, policy, retrieved: list[dict], top_score: float, query_interpretation: dict | None = None) -> bool:
        query_interpretation = query_interpretation or {}
        top_item = retrieved[0] if retrieved else {}
        lexical_signal = float(top_item.get("sparse_score", 0.0)) + float(top_item.get("title_score", 0.0)) + float(top_item.get("title_match_bonus", 0.0)) + float(top_item.get("compact_match_bonus", 0.0))
        resource_match_score = float(top_item.get("resource_match_score", 0.0))
        lexical_match_score = float(top_item.get("lexical_match_score", 0.0))
        query_tokens = {token for token in query_interpretation.get("normalized_keywords", []) if len(token) >= 2 and token not in {"pdf", "설명", "explain"}}
        metadata = top_item.get("chunk", {}).get("metadata", {}) if top_item else {}
        structure_text = " ".join([str(metadata.get("section_title", "") or ""), str(metadata.get("section_path", "") or ""), " ".join(str(value) for value in metadata.get("parent_headings", []) or [])])
        structure_tokens = set(tokenize(structure_text))
        has_structural_anchor = bool(query_tokens and query_tokens & structure_tokens)
        strong_resource_anchor = resource_match_score >= 0.9 or lexical_match_score >= 0.2 or has_structural_anchor

        if top_score < self.settings.retrieval_min_score or not retrieved:
            lowered_shape = _qi_str(query_interpretation, "response_shape")
            lowered_intent = _qi_str(query_interpretation, "intent")
            has_explicit_resources = bool(query_interpretation.get("resources"))
            relaxed_threshold = self.settings.retrieval_min_score
            if has_explicit_resources:
                relaxed_threshold = min(relaxed_threshold, max(self.settings.retrieval_retry_min_score, 0.10))
            if lowered_shape in {"code", "table", "procedure", "comparison"} or lowered_intent in {"yaml_example", "cli_example", "code_example", "table", "compare", "procedure_followup", "explain"}:
                relaxed_threshold = min(self.settings.retrieval_min_score, max(self.settings.retrieval_retry_min_score, 0.15))
            if policy.turn_type in {"document_query", "document_followup"} and lowered_intent == "explain" and (lexical_signal >= 0.06 or has_structural_anchor):
                relaxed_threshold = min(relaxed_threshold, max(self.settings.retrieval_retry_min_score, 0.08 if has_explicit_resources else 0.12))
            if policy.turn_type in {"document_query", "document_followup"} and lowered_intent == "explain" and has_explicit_resources and strong_resource_anchor:
                relaxed_threshold = min(relaxed_threshold, max(self.settings.retrieval_retry_min_score * 0.8, 0.04))
            if top_score < relaxed_threshold or not retrieved:
                return False

        if len(retrieved) >= 2:
            second_score = float(retrieved[1].get("rerank_score", 0.0))
            if top_score - second_score > 0.15 and top_score >= 0.08:
                return True

        resources = _qi_set(query_interpretation, "resources")
        if resources and top_item:
            chunk_meta = top_item.get("chunk", {}).get("metadata", {})
            code_signals = {str(signal).casefold() for signal in chunk_meta.get("code_signals", []) or []}
            chunk_text = str(top_item.get("chunk", {}).get("text", "")).casefold()
            explicit_kind = ""
            kind_match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)", chunk_text)
            if kind_match:
                explicit_kind = kind_match.group(1).casefold()
            for resource in resources:
                if resource == explicit_kind or resource in code_signals:
                    return True

        if policy.turn_type == "document_query" and top_score < 0.2 and lexical_signal <= 0.0 and not strong_resource_anchor:
            return False
        return True

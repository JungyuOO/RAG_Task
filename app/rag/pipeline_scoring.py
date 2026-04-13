from __future__ import annotations

import logging
import re
import time
from pathlib import Path

logger = logging.getLogger("rag.pipeline")

from app.rag.utils import normalize_text, tokenize

_KIND_RE = re.compile(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$")


def _qi_str(qi: dict, key: str) -> str:
    return str(qi.get(key, "") or "").casefold()


def _qi_set(qi: dict, key: str) -> set[str]:
    return {str(v).casefold() for v in qi.get(key, []) if v}


class PipelineRetrievalMixin:
    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _item_primary_score(item: dict) -> float:
        return float(item.get("final_retrieval_score", item.get("rerank_score", 0.0)))

    def _normalize_metadata_component(self, metadata_score: float, focus_multiplier: float) -> float:
        normalized = self._clamp_unit(max(metadata_score, 0.0) / 3.0)
        return self._clamp_unit(normalized * max(focus_multiplier, 0.0))

    def _build_anchor_score(
        self,
        *,
        item: dict,
        has_structural_anchor: bool,
        resource_match_score: float,
        lexical_match_score: float,
    ) -> float:
        lexical_signal = (
            self._clamp_unit(float(item.get("sparse_score", 0.0)))
            + self._clamp_unit(float(item.get("title_score", 0.0)))
        ) / 2.0
        resource_anchor = self._clamp_unit(max(resource_match_score, 0.0))
        lexical_anchor = self._clamp_unit(max(lexical_match_score, 0.0) / 0.35)
        structural_anchor = 1.0 if has_structural_anchor else 0.0
        continuity_anchor = self._clamp_unit(float(item.get("continuity_score", 0.0)))
        return max(lexical_signal, resource_anchor, lexical_anchor, structural_anchor, continuity_anchor)

    def _compose_final_retrieval_score(
        self,
        *,
        ce_score: float,
        metadata_component: float,
        anchor_score: float,
        query_interpretation: dict | None = None,
    ) -> float:
        query_interpretation = query_interpretation or {}
        intent = _qi_str(query_interpretation, "intent")
        ce_weight = float(getattr(self.settings, "retrieval_final_ce_weight", 0.65))
        metadata_weight = float(getattr(self.settings, "retrieval_final_metadata_weight", 0.25))
        anchor_weight = float(getattr(self.settings, "retrieval_final_anchor_weight", 0.10))
        if intent in {"yaml_example", "cli_example", "code_example"} or _qi_str(query_interpretation, "response_shape") == "code":
            ce_weight = 0.50
            metadata_weight = 0.35
            anchor_weight = 0.15
        weight_sum = ce_weight + metadata_weight + anchor_weight
        if weight_sum <= 0:
            return self._clamp_unit(ce_score)
        return self._clamp_unit(
            (
                self._clamp_unit(ce_score) * ce_weight
                + self._clamp_unit(metadata_component) * metadata_weight
                + self._clamp_unit(anchor_score) * anchor_weight
            )
            / weight_sum
        )

    def _metadata_aware_score(self, user_message: str, query_interpretation: dict | None, item: dict) -> dict:
        query_interpretation = query_interpretation or {}
        metadata = item["chunk"].get("metadata", {})
        lowered_text = str(item["chunk"].get("text", "")).casefold()
        source_path = str(item["chunk"].get("source_path", "") or "")
        source_name = Path(source_path).name.casefold()
        block_types = {value.strip().casefold() for value in str(metadata.get("block_types", "")).split(",") if value.strip()}
        code_language = str(metadata.get("code_language", "")).casefold()
        code_subtype = str(metadata.get("code_subtype", "")).casefold()
        code_signals = {str(signal).casefold() for signal in metadata.get("code_signals", []) or []}
        block_code_languages = {str(value).casefold() for value in metadata.get("block_code_languages", []) or []}
        block_code_resource_kinds = {str(value).casefold() for value in metadata.get("block_code_resource_kinds", []) or []}
        table_headers = {str(value).casefold() for value in metadata.get("table_headers", []) or []}
        list_item_count = int(metadata.get("list_item_count") or 0)
        table_row_count = int(metadata.get("table_row_count") or 0)
        table_column_count = int(metadata.get("table_column_count") or 0)
        has_cli_block = bool(metadata.get("has_cli_block"))
        is_toc = bool(metadata.get("is_toc"))
        is_intro = bool(metadata.get("is_intro"))
        is_overview = bool(metadata.get("is_overview"))
        is_procedure = bool(metadata.get("is_procedure"))
        explicit_kind_match = _KIND_RE.search(lowered_text)
        explicit_resource_kind = explicit_kind_match.group(1).casefold() if explicit_kind_match else ""
        query_tokens = {token for token in query_interpretation.get("normalized_keywords", tokenize(user_message)) if len(token) >= 2 and token not in {"yaml", "manifest", "code", "example", "sample", "demo"}}
        source_query_tokens = set(query_tokens)
        for token in list(query_tokens):
            source_query_tokens.update(
                part
                for part in re.split(r"[^a-z0-9]+", token.casefold())
                if len(part) >= 3
            )
        resources = self._resolve_requested_resource_kinds(query_interpretation)
        actions = _qi_set(query_interpretation, "actions")
        format_constraints = _qi_set(query_interpretation, "format_constraints")
        response_shape = _qi_str(query_interpretation, "response_shape")
        intent = _qi_str(query_interpretation, "intent")
        is_code_request = (
            intent in {"yaml_example", "cli_example", "code_example"}
            or response_shape == "code"
            or bool(format_constraints & {"yaml", "cli"})
        )
        generic_command_query = bool(query_interpretation.get("generic_command_query"))
        normalized_user = normalize_text(user_message).casefold()
        asks_for_toc = any(marker in normalized_user for marker in ("목차", "contents", "table of contents", "섹션", "절"))
        asks_for_overview = any(marker in normalized_user for marker in ("개요", "소개", "overview", "introduction", "주제", "설명"))

        heading_score = self._heading_overlap_score(user_message, metadata)
        resource_score = action_score = format_score = shape_score = lexical_score = completeness_score = group_score = source_score = 0.0
        focus_multiplier = 1.0
        is_single_resource_focus = len(resources) == 1 and (intent == "explain" or response_shape == "text")
        document_group_preference = _qi_str(query_interpretation, "document_group_preference")
        item_document_group = str(metadata.get("document_group") or ("customer_generated" if metadata.get("doc_type") == "operation_manual" else "official_ocp")).casefold()

        if resources:
            matched_resources = 0
            for resource in resources:
                if resource == explicit_resource_kind:
                    matched_resources += 1
                    resource_score += 1.0
                elif resource in block_code_resource_kinds:
                    matched_resources += 1
                    resource_score += 0.95
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

        explain_focus_bonus = 0.0
        if is_single_resource_focus:
            resource = next(iter(resources), "")
            section_title = str(metadata.get("section_title", "") or "").casefold()
            section_path = str(metadata.get("section_path", "") or "").casefold()
            parent_headings = " ".join(str(value).casefold() for value in metadata.get("parent_headings", []) or [])
            focus_text = " ".join([section_title, section_path, parent_headings]).strip()
            boost = float(getattr(self.settings, "retrieval_explain_focus_boost", 0.08))
            if resource and focus_text:
                if resource in section_title:
                    explain_focus_bonus += boost
                elif resource in focus_text:
                    explain_focus_bonus += boost * 0.65

        if "create" in actions:
            if any(marker in lowered_text for marker in ("create", "생성", "만들", "작성")):
                action_score += 0.4
            if any(marker in str(metadata.get("section_title", "")).casefold() for marker in ("create", "생성")):
                action_score += 0.5
        if "compare" in actions and "table" in block_types:
            action_score += 0.5
        if "explain" in actions and "code" not in block_types:
            action_score += 0.25
        if intent == "procedure_followup" and list_item_count >= 2:
            action_score += 0.25
        if is_toc and not asks_for_toc:
            action_score -= 1.0
        if is_intro and response_shape == "code":
            action_score -= 0.35
        if is_overview and response_shape == "code":
            action_score -= 0.25
        if is_procedure and intent == "procedure_followup":
            action_score += 0.35

        if "yaml" in format_constraints:
            if code_language in {"yaml", "yml"}:
                format_score += 1.2
            if block_code_languages & {"yaml", "yml"}:
                format_score += 0.8
            if code_subtype == "k8s_manifest":
                format_score += 0.9
            if is_toc or is_intro or is_overview:
                format_score -= 0.45
        if "cli" in format_constraints:
            if code_subtype == "cli_command":
                format_score += 1.1
            if code_language in {"bash", "sh", "shell"}:
                format_score += 0.8
            if has_cli_block or (block_code_languages & {"bash", "sh", "shell"}):
                format_score += 0.55
            if is_toc or is_intro or is_overview:
                format_score -= 0.45
        if "table" in format_constraints and "table" in block_types:
            format_score += 1.0
        if "table" in format_constraints and table_headers:
            header_overlap = len(query_tokens & table_headers)
            if header_overlap:
                format_score += min(header_overlap * 0.18, 0.54)

        if response_shape == "code":
            if "code" in block_types:
                shape_score += 0.75
            elif "table" in block_types:
                shape_score -= 0.15
            if block_code_languages or has_cli_block:
                shape_score += 0.20
            if is_toc or is_intro or is_overview:
                shape_score -= 0.45
        elif response_shape == "table":
            if "table" in block_types:
                shape_score += 0.75
            elif "code" in block_types:
                shape_score -= 0.2
            if table_row_count > 0 and table_column_count > 0:
                shape_score += 0.20
        elif response_shape == "procedure":
            if list_item_count >= 2:
                shape_score += 0.35
        elif response_shape in {"text", "comparison"} and "code" in block_types:
            shape_score -= 0.15
        if is_toc and not asks_for_toc:
            shape_score -= 0.45
        if is_intro and not asks_for_overview and intent == "explain":
            shape_score -= 0.20

        for token in query_tokens:
            token_casefold = token.casefold()
            if token_casefold in code_signals:
                lexical_score += 0.35
            elif token_casefold in lowered_text:
                lexical_score += 0.12
            if token_casefold in source_name:
                source_score += 0.18

        strong_query_tokens = {
            token.casefold()
            for token in query_tokens
            if len(token) >= 4 and token.casefold() not in {"namespace", "project", "status", "command"}
        }
        structure_text_lower = " ".join(
            [
                str(metadata.get("section_title", "") or "").casefold(),
                str(metadata.get("section_path", "") or "").casefold(),
                " ".join(str(value).casefold() for value in metadata.get("parent_headings", []) or []),
            ]
        )
        strong_token_hits = sum(
            1 for token in strong_query_tokens
            if token in lowered_text or token in structure_text_lower
        )
        if strong_token_hits:
            lexical_score += min(strong_token_hits * 0.28, 0.84)
            if is_procedure:
                action_score += min(strong_token_hits * 0.12, 0.24)
        elif strong_query_tokens and (has_cli_block or explicit_resource_kind or is_procedure):
            source_score -= 0.22

        source_name_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", source_name)
            if len(token) >= 3 and token not in {"openshift", "container", "platform", "customer", "guide", "official", "generated", "pdf", "en", "us", "ocp"}
        }
        source_overlap = len(source_query_tokens & source_name_tokens)
        if source_overlap:
            source_score += min(source_overlap * 0.12, 0.48)

        section_title_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", str(metadata.get("section_title", "")).casefold())
            if len(token) >= 3
        }
        section_path_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", str(metadata.get("section_path", "")).casefold())
            if len(token) >= 3
        }
        section_overlap = len(source_query_tokens & (section_title_tokens | section_path_tokens))
        if section_overlap:
            source_score += min(section_overlap * 0.14, 0.56)

        if generic_command_query:
            if source_name == "cli_tools.md":
                source_score += 0.90
            if any(token in query_tokens for token in {"namespace", "project"}):
                if "oc project" in lowered_text or "oc projects" in lowered_text:
                    format_score += 0.80
                if "deployment" in lowered_text and "project" not in lowered_text and "namespace" not in lowered_text:
                    source_score -= 0.40
            if any(token in query_tokens for token in {"pod"}):
                if re.search(r"\boc get pods?\b(?!.*(?:-l|--selector|jsonpath|grep|jq))", lowered_text):
                    format_score += 0.65
                if "-o yaml" in lowered_text and "yaml" not in query_tokens:
                    format_score -= 0.30
            if "yaml" in query_tokens:
                if "-o yaml" in lowered_text or "oc describe pod" in lowered_text:
                    format_score += 0.70
                if "oc create -f" in lowered_text or "oc apply -f" in lowered_text:
                    format_score -= 0.45
            if any(marker in lowered_text for marker in ("-l ", "--selector", "jsonpath", "app.kubernetes.io", "cert-manager", "workshop")):
                source_score -= 0.40

        if is_code_request and "code" in block_types:
            shape_score += 0.25
        if "code" in block_types:
            completeness_score += self._code_completeness_score(lowered_text)

        if document_group_preference in {"customer_generated", "official_ocp"}:
            if item_document_group == document_group_preference:
                group_score += 0.6
            else:
                group_score -= 0.2
        elif document_group_preference == "mixed" and "compare" in actions:
            if item_document_group in {"customer_generated", "official_ocp"}:
                group_score += 0.15

        metadata_score = heading_score + resource_score + action_score + format_score + shape_score + lexical_score + completeness_score + explain_focus_bonus + group_score + source_score
        ce_score = float(item.get("ce_score", item.get("rerank_score", 0.0)))
        dense_score = self._clamp_unit(float(item.get("dense_score", 0.0)))
        sparse_score = self._clamp_unit(float(item.get("sparse_score", 0.0)))
        if is_code_request:
            command_dense_prior = self._clamp_unit(dense_score * 0.80 + sparse_score * 0.05)
            ce_score = max(ce_score, command_dense_prior)
        structure_text = " ".join(
            [
                str(metadata.get("section_title", "") or ""),
                str(metadata.get("section_path", "") or ""),
                " ".join(str(value) for value in metadata.get("parent_headings", []) or []),
            ]
        )
        structure_tokens = set(tokenize(structure_text))
        has_structural_anchor = bool(query_tokens and query_tokens & structure_tokens)
        metadata_component = self._normalize_metadata_component(metadata_score, focus_multiplier)
        anchor_score = self._build_anchor_score(
            item=item,
            has_structural_anchor=has_structural_anchor,
            resource_match_score=resource_score,
            lexical_match_score=lexical_score,
        )
        final_retrieval_score = self._compose_final_retrieval_score(
            ce_score=ce_score,
            metadata_component=metadata_component,
            anchor_score=anchor_score,
            query_interpretation=query_interpretation,
        )
        return {
            "ce_score": ce_score,
            "heading_overlap_score": heading_score,
            "resource_match_score": resource_score,
            "action_match_score": action_score,
            "format_match_score": format_score,
            "shape_match_score": shape_score,
            "lexical_match_score": lexical_score,
            "completeness_score": completeness_score,
            "group_match_score": group_score,
            "source_match_score": source_score,
            "explain_focus_bonus": explain_focus_bonus,
            "focus_multiplier": focus_multiplier,
            "metadata_score": metadata_score,
            "metadata_component_score": metadata_component,
            "structural_anchor_score": 1.0 if has_structural_anchor else 0.0,
            "anchor_score": anchor_score,
            "metadata_final_score": final_retrieval_score,
            "final_retrieval_score": final_retrieval_score,
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
        t_total = time.perf_counter()
        if not items:
            logger.info("[Timing][PipelineRetrievalMixin._metadata_aware_rerank] total=%.3fs items=0", time.perf_counter() - t_total)
            return []
        rescored = [{**item, **self._metadata_aware_score(user_message, query_interpretation, item)} for item in items]
        rescored.sort(
            key=lambda item: (
                -float(item.get("final_retrieval_score", 0.0)),
                -float(item.get("anchor_score", 0.0)),
                -float(item.get("metadata_component_score", 0.0)),
                -float(item.get("ce_score", item.get("rerank_score", 0.0))),
            )
        )
        logger.info(
            "[Timing][PipelineRetrievalMixin._metadata_aware_rerank] total=%.3fs items=%d top_score=%.4f",
            time.perf_counter() - t_total,
            len(rescored),
            float(rescored[0].get("final_retrieval_score", 0.0)),
        )
        return rescored

    def _has_code_content(self, item: dict) -> bool:
        text = str(item["chunk"].get("text", "") or "")
        if not text.strip():
            return False
        if len(self.answer_service._extract_code_candidates(text)) > 0:
            return True
        return len(self.answer_service._extract_command_candidates(text)) > 0

    def _select_code_example_context_items(self, user_message: str, query_interpretation: dict | None, ordered_context_items: list[dict], selected_context_items: list[dict]) -> list[dict]:
        query_interpretation = query_interpretation or {}
        candidates = ordered_context_items or selected_context_items
        code_candidates = self._prefer_block_type_items(candidates, block_type="code", limit=None)
        if not code_candidates:
            code_candidates = [item for item in candidates if self._has_code_content(item)]
        if not code_candidates:
            return selected_context_items

        requested_resource_kinds = self._resolve_requested_resource_kinds(query_interpretation)
        generic_command_query = bool(query_interpretation.get("generic_command_query"))
        if requested_resource_kinds and not generic_command_query:
            explicit_kind_matches = [item for item in code_candidates if self._extract_explicit_resource_kind(item) in requested_resource_kinds]
            if explicit_kind_matches:
                code_candidates = explicit_kind_matches
            exact_resource_matches = [item for item in code_candidates if self._infer_item_resource_kinds(item) & requested_resource_kinds]
            if exact_resource_matches:
                code_candidates = exact_resource_matches

        rescored = self._metadata_aware_rerank(user_message, query_interpretation, code_candidates)
        has_positive_match = any(item.get("resource_match_score", 0) > 0 for item in rescored)
        if has_positive_match and not generic_command_query:
            rescored = [item for item in rescored if item.get("resource_match_score", 0) >= 0]
        for item in rescored:
            item["code_selection_score"] = float(item.get("final_retrieval_score", 0.0))
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
            expanded.append({"chunk": chunk, "dense_score": 0.0, "sparse_score": 0.0, "rerank_score": 0.0, "ce_score": 0.0})
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
        followup_markers = (
            "그거",
            "그건",
            "그 문서",
            "그 페이지",
            "그 yaml",
            "그 코드",
            "그 예시",
            "예시",
            "종류",
            "특징",
            "자세히",
            "설명",
            "설치",
            "과정",
            "구성",
            "차이",
            "비교",
            "다음",
            "계속",
            "that",
            "this",
            "those",
            "again",
            "next",
            "continue",
        )
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
            source_anchor_scores[source_path] = max(source_anchor_scores.get(source_path, 0.0), self._item_primary_score(item))

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
            expanded.append(
                {
                    "chunk": chunk,
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "rerank_score": 0.0,
                    "ce_score": 0.0,
                    "continuity_score": continuity_score,
                }
            )
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
                candidate_pool.append(
                    {
                        "chunk": chunk,
                        "rerank_score": float(item.get("rerank_score", 0.0)),
                        "ce_score": float(item.get("ce_score", item.get("rerank_score", 0.0))),
                        "dense_score": float(item.get("dense_score", 0.0)),
                        "sparse_score": float(item.get("sparse_score", 0.0)),
                        "score": float(item.get("score", 0.0)),
                        "final_retrieval_score": float(item.get("final_retrieval_score", item.get("rerank_score", 0.0))),
                    }
                )
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
            top_score = max(float(item.get("final_retrieval_score", 0.0)) for item in items)
            if top_score > 0:
                threshold = top_score * 0.2
                items = [item for item in items if float(item.get("final_retrieval_score", 0.0)) >= threshold]
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
        actions = [str(value).casefold() for value in query_interpretation.get("actions", []) if value]
        if intent in {"yaml_example", "cli_example", "code_example"} or response_shape == "code":
            return "extractive_code"
        if intent == "table" or response_shape == "table":
            return "extractive_table"
        if "compare" in actions:
            return "extractive_compare"
        if "compare" not in actions and intent in {"explain", "procedure_followup"} and response_shape in {"", "text", "procedure"}:
            return "extractive_text"
        return "grounded_generation"

    def _build_missing_extractive_answer(self, answer_route: str) -> str:
        if answer_route == "extractive_code":
            return "업로드한 문서에서 요청하신 YAML/코드 예시를 직접 찾지 못했습니다. 문서에 실제 예시 블록이 있는지 다시 확인하시거나, 더 구체적인 범위나 페이지를 지정해 주세요."
        if answer_route == "extractive_table":
            return "업로드한 문서에서 요청하신 표나 비교 정보를 직접 찾지 못했습니다. 키워드를 조금 더 구체적으로 적어 다시 질문해 주세요."
        return "업로드한 문서에서 관련 내용을 찾을 수 없습니다."

    def _build_policy_answer(self, turn_type: str, top_score: float) -> str:
        if turn_type == "conversational_ack":
            return "문서와 관련된 질문이 있으면 이어서 질문해 주세요."
        if turn_type == "greeting":
            return "안녕하세요! 무엇을 도와드릴까요?"
        if turn_type == "general_chat":
            return (
                "제가 바로 도와드릴 수 있는 범위는 다음과 같습니다.\n\n"
                "- 공식 문서 기반 명령어 / 절차 질문\n"
                "- 현재 demo namespace 기준 Pod / Event / YAML 조회\n"
                "- 문서 기준 명령어와 실제 OCP 결과를 함께 보여주는 mixed 질문\n"
                "- 고객사 메뉴얼 업로드 후 고객사 메뉴얼 기준 질의\n\n"
                "예: `pod 확인하는 명령어 뭐야?`, `지금 pandas 관련 pod 보여줘`, `현재 상태 확인 명령어랑 실제 결과 같이 알려줘`"
            )
        if turn_type == "document_query":
            if top_score >= self.settings.retrieval_retry_min_score:
                return "관련 내용을 찾기 어려웠습니다. 질문을 조금 더 구체적으로 적어 주세요.\n예: `스토리지 문서에서 PV 설명해줘`, `Service 종류를 서로 비교해줘`"
            return "업로드한 문서에서 관련 내용을 찾을 수 없습니다. 다른 질문을 하시거나 관련 문서를 업로드해 주세요."
        return "업로드한 문서에서 관련 내용을 찾을 수 없습니다."

    def _should_use_retrieved_context(self, policy, retrieved: list[dict], top_score: float, query_interpretation: dict | None = None) -> bool:
        query_interpretation = query_interpretation or {}
        del top_score
        if not retrieved:
            logger.info("[RetrievalGate] no retrieved items -> use_context=False")
            return False
        threshold = float(getattr(self.settings, "retrieval_gate_threshold", getattr(self.settings, "retrieval_min_score", 0.25)))
        if _qi_str(query_interpretation, "intent") == "explain" and _qi_str(query_interpretation, "response_shape") in {"", "text"}:
            threshold = min(threshold, float(getattr(self.settings, "retrieval_explain_gate_threshold", threshold)))
        if _qi_str(query_interpretation, "intent") in {"yaml_example", "cli_example", "code_example"} or _qi_str(query_interpretation, "response_shape") == "code":
            threshold = min(threshold, 0.08)
        top_item_score = self._item_primary_score(retrieved[0])
        decision = top_item_score >= threshold
        logger.info(
            "[RetrievalGate] turn_type=%s top_score=%.4f threshold=%.4f use_context=%s",
            getattr(policy, "turn_type", ""),
            top_item_score,
            threshold,
            decision,
        )
        return decision

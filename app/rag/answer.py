from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from app.rag.retrieval_service import RetrievalService


class AnswerGenerator:
    """LLM 응답에서 인용을 추출하고, 출처 라인과 프리뷰 페이지를 구성하는 서비스.

    답변 텍스트에서 [파일명] p.N 형식의 인용을 파싱하여 context 아이템과
    매칭하고, 코드 예시 요청 시 문서에서 코드 블록을 추출하여 응답한다.
    """

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    @staticmethod
    def public_context_payload(payload: dict) -> dict:
        public_payload = dict(payload or {})
        public_payload.pop("_stored_procedure_state", None)
        return public_payload

    @staticmethod
    def strip_code_blocks_for_non_code_route(answer: str, answer_route: str) -> str:
        if answer_route == "extractive_code":
            return answer
        return re.sub(r"```(?:[\w+-]+)?\n.*?```", "", answer, flags=re.DOTALL).strip()

    @staticmethod
    def build_example_anchor(context_items: list[dict], query_interpretation: dict | None) -> dict:
        query_interpretation = query_interpretation or {}
        fields: list[str] = []
        context_ids: list[str] = []
        page_numbers: list[int] = []
        section_paths: list[str] = []
        source_path = ""
        resources = [str(value) for value in query_interpretation.get("resources", []) if value]
        resource_kind = resources[0] if resources else ""
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
    def extract_procedure_state(answer: str) -> dict:
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

    @staticmethod
    def should_capture_procedure_state(query_interpretation, answer_route, policy_decision) -> bool:
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

    def finalize_answer(
        self,
        answer: str,
        rewritten_query: str,
        use_retrieved_context: bool,
        top_score: float,
        selected_context_items: list[dict],
        grounded_pages: list[dict],
        preferred_preview_source: str | None,
        response_mode: str,
        policy_decision,
        query_interpretation: dict | None,
        answer_route: str,
        retrieval_min_score: float,
    ) -> tuple[str, list[dict], dict]:
        answer = self.strip_code_blocks_for_non_code_route(answer, answer_route)
        answer = self.sanitize_answer(answer, use_retrieved_context)
        answer_citations = (
            self.build_answer_citation_payload(
                answer, selected_context_items, grounded_pages, preferred_preview_source,
            )
            if policy_decision.allow_citations
            else []
        )
        final_answer = self.ensure_answer_source_line(answer, answer_citations, use_retrieved_context)
        show_preview = (
            policy_decision.allow_preview
            and use_retrieved_context
            and bool(answer_citations or grounded_pages)
        )
        if show_preview:
            final_source, final_preview_pages = self.build_answer_aligned_preview_pages(
                answer_citations, selected_context_items, preferred_preview_source, grounded_pages,
            )
        else:
            final_source, final_preview_pages = None, []
        final_payload = self.build_context_payload(
            rewritten_query,
            response_mode,
            top_score,
            final_source,
            final_preview_pages,
            selected_context_items,
            grounded_pages,
            answer_citations,
            preview_finalized=True,
        )
        final_payload["query_interpretation"] = query_interpretation or {}
        final_payload["answer_route"] = answer_route
        if answer_route == "extractive_code":
            final_payload["last_example_anchor"] = self.build_example_anchor(selected_context_items, query_interpretation)
        if self.should_capture_procedure_state(query_interpretation, answer_route, policy_decision):
            procedure_state = self.extract_procedure_state(final_answer)
        else:
            procedure_state = {}
        if procedure_state:
            if answer_route == "procedure_state_followup":
                final_payload["procedure_state"] = procedure_state
            else:
                final_payload["_stored_procedure_state"] = procedure_state
        return final_answer, answer_citations, final_payload

    def build_extractive_table_answer(self, context_items: list[dict]) -> str | None:
        tables: list[dict[str, str]] = []
        seen_tables: set[str] = set()

        for item in context_items:
            chunk = item["chunk"]
            text = str(chunk.get("text", "") or "")
            if not text.strip():
                continue
            for table in self._extract_markdown_tables(text):
                normalized = re.sub(r"\s+", " ", table).strip().casefold()
                if len(normalized) < 20 or normalized in seen_tables:
                    continue
                seen_tables.add(normalized)
                tables.append(
                    {
                        "file_name": Path(chunk["source_path"]).name,
                        "page_start": str(chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                        "page_end": str(chunk["metadata"].get("page_end") or chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                        "table": table.strip(),
                    }
                )
                if len(tables) >= 2:
                    break
            if len(tables) >= 2:
                break

        if not tables:
            return None

        parts = ["문서에서 확인된 표를 그대로 정리하면 아래와 같습니다."]
        for table in tables:
            page_label = (
                f"p.{table['page_start']}"
                if table["page_start"] == table["page_end"]
                else f"p.{table['page_start']}-{table['page_end']}"
            )
            parts.append(f"[{table['file_name']}] {page_label}")
            parts.append(table["table"])
        return "\n\n".join(parts).strip()

    def build_extractive_code_answer(
        self,
        context_items: list[dict],
        requested_resource_kinds: set[str] | None = None,
    ) -> str | None:
        snippets: list[dict[str, str]] = []
        seen_blocks: set[str] = set()
        requested_resource_kinds = {str(value).casefold() for value in (requested_resource_kinds or set()) if value}

        for item in context_items:
            chunk = item["chunk"]
            text = str(chunk.get("text", "") or "")
            if not text.strip():
                continue
            for block in self._extract_code_candidates(text):
                normalized_blocks = self._split_and_filter_code_blocks(block, requested_resource_kinds)
                for normalized_block in normalized_blocks:
                    normalized = re.sub(r"\s+", " ", normalized_block).strip().casefold()
                    if len(normalized) < 24 or normalized in seen_blocks:
                        continue
                    seen_blocks.add(normalized)
                    snippets.append(
                        {
                            "file_name": Path(chunk["source_path"]).name,
                            "page_start": str(chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                            "page_end": str(chunk["metadata"].get("page_end") or chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                            "code": normalized_block.strip(),
                            "code_language": str(chunk["metadata"].get("code_language", "") or ""),
                        }
                    )
                    if len(snippets) >= 3:
                        break
                if len(snippets) >= 3:
                    break
            if len(snippets) >= 3:
                break

        if not snippets:
            return None

        parts = ["문서에 나온 예시 코드를 그대로 정리하면 아래와 같습니다."]
        for snippet in snippets:
            page_label = (
                f"p.{snippet['page_start']}"
                if snippet["page_start"] == snippet["page_end"]
                else f"p.{snippet['page_start']}-{snippet['page_end']}"
            )
            fence_language = snippet["code_language"] if snippet["code_language"] in {"yaml", "yml", "bash", "sh", "shell", "json"} else "yaml"
            parts.append(f"[{snippet['file_name']}] {page_label}")
            parts.append(f"```{fence_language}")
            parts.append(snippet["code"])
            parts.append("```")
        return "\n\n".join(parts).strip()

    def _split_and_filter_code_blocks(self, block: str, requested_resource_kinds: set[str]) -> list[str]:
        if not requested_resource_kinds:
            return [block]
        yaml_docs = [part.strip() for part in re.split(r"(?m)^\s*---\s*$", block) if part.strip()]
        detected_kinds = {
            kind
            for kind in (self._extract_code_block_kind(doc) for doc in yaml_docs)
            if kind
        }
        matched_docs = [
            doc for doc in yaml_docs
            if self._extract_code_block_kind(doc) in requested_resource_kinds
        ]
        if matched_docs:
            return matched_docs
        if detected_kinds:
            return []
        return [block]

    def _extract_code_block_kind(self, block: str) -> str:
        match = re.search(r"(?im)^\s*kind:\s*([a-z0-9_-]+)\s*$", block)
        return match.group(1).casefold() if match else ""

    def sanitize_answer(self, answer: str, use_retrieved_context: bool) -> str:
        if not answer:
            return answer
        if use_retrieved_context:
            return self._sanitize_retrieved_answer(answer)
        return self._sanitize_general_answer(answer)

    def _sanitize_retrieved_answer(self, answer: str) -> str:
        sanitized = answer.replace("\r\n", "\n").strip()
        lines: list[str] = []
        for raw_line in sanitized.split("\n"):
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            if self._is_retrieved_mode_negative_artifact(line):
                continue
            lines.append(raw_line.rstrip())

        compacted: list[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            compacted.append(line)
            previous_blank = is_blank
        return "\n".join(compacted).strip()

    def _sanitize_general_answer(self, answer: str) -> str:
        sanitized = answer.replace("\r\n", "\n").strip()
        lines: list[str] = []
        for raw_line in sanitized.split("\n"):
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            if self._is_general_mode_source_artifact(line):
                continue
            lines.append(raw_line.rstrip())

        compacted: list[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            compacted.append(line)
            previous_blank = is_blank
        return "\n".join(compacted).strip()

    def _is_general_mode_source_artifact(self, line: str) -> bool:
        normalized = line.casefold()
        if normalized in {"[general knowledge]", "general knowledge"}:
            return True
        if "uploaded library" in normalized:
            return True
        if "no specific content was retrieved" in normalized:
            return True
        if "general knowledge" in normalized and line.strip().startswith("[") and line.strip().endswith("]"):
            return True
        if "uploaded documents" in normalized and "supporting evidence" in normalized:
            return True
        if "retrieval mode:" in normalized:
            return True
        if re.fullmatch(r"\[[^\]\n]+\]\s*p\.\d+(?:-\d+)?(?:\s*\(.*\))?", line, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"sources:\s*.+", line, flags=re.IGNORECASE):
            return True
        return False

    def _is_retrieved_mode_negative_artifact(self, line: str) -> bool:
        normalized = line.casefold()
        patterns = (
            "제공된 문서에는",
            "문서에는",
            "포함되어 있지 않습니다",
            "구체적인 정의나 설명이 포함되어 있지 않습니다",
            "is not included in the provided document",
            "is not specifically described in the provided document",
        )
        if any(pattern.casefold() in normalized for pattern in patterns):
            return True
        return False

    def _extract_code_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        for fenced in re.findall(r"```(?:[\w+-]+)?\n(.*?)```", text, flags=re.DOTALL):
            block = self._clean_code_block(fenced)
            if self._looks_like_code_block(block):
                candidates.append(block)

        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        current: list[str] = []
        in_code = False

        for raw_line in lines:
            line = raw_line.strip()
            if self._is_code_break_line(line):
                if current:
                    block = self._clean_code_block("\n".join(current))
                    if self._looks_like_code_block(block):
                        candidates.append(block)
                    current = []
                in_code = False
                continue

            if not line:
                if current:
                    block = self._clean_code_block("\n".join(current))
                    if self._looks_like_code_block(block):
                        candidates.append(block)
                    current = []
                    in_code = False
                continue

            if self._is_code_line(line) or self._looks_like_code_continuation(line):
                current.append(raw_line.rstrip())
                in_code = True
                continue

            if in_code and (self._looks_like_code_comment(line) or self._looks_like_code_continuation(line)):
                current.append(raw_line.rstrip())
                continue

            if current:
                block = self._clean_code_block("\n".join(current))
                if self._looks_like_code_block(block):
                    candidates.append(block)
                current = []
            in_code = False

        if current:
            block = self._clean_code_block("\n".join(current))
            if self._looks_like_code_block(block):
                candidates.append(block)

        return self._merge_code_candidates(candidates)

    def _extract_markdown_tables(self, text: str) -> list[str]:
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        tables: list[str] = []
        current: list[str] = []

        def flush() -> None:
            nonlocal current
            if len(current) >= 2:
                tables.append("\n".join(current).strip())
            current = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                current.append(stripped)
                continue
            if current:
                flush()
        if current:
            flush()
        return [table for table in tables if self._looks_like_markdown_table(table)]

    def _looks_like_markdown_table(self, table: str) -> bool:
        lines = [line.strip() for line in table.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        separator_re = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
        return any(separator_re.match(line) for line in lines[1:3])

    def _is_code_line(self, line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.casefold()
        if stripped.startswith("# "):
            return True
        if lowered.startswith(("apiversion:", "kind:", "metadata:", "spec:", "data:", "stringdata:")):
            return True
        if lowered.startswith(("oc ", "kubectl ", "helm ", "cat ", "echo ")):
            return True
        if re.match(r"^[A-Za-z0-9_.-]+\.(ya?ml|json)$", stripped):
            return True
        if lowered.startswith(("host:", "to:", "provisioner:", "parameters:", "allowvolumeexpansion:")):
            return True
        if re.match(r"^[A-Za-z0-9_.\"'/-]+\s*:\s*", stripped):
            return True
        return False

    def _looks_like_code_comment(self, line: str) -> bool:
        return line.strip().startswith("#")

    def _looks_like_code_continuation(self, line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.casefold()
        if lowered.startswith((
            "-", "name:", "image:", "path:", "storage:", "accessmodes:", "resources:", "requests:",
            "claimname:", "mountpath:", "containers:", "volumes:", "templ", "selector:", "matchlabels:",
            "replicas:", "provisioner:", "parameters:", "type:", "allowvolumeexpansion:", "mountoptions:",
            "persistentvolumeclaim:", "volumemounts:", "volumemode:", "persistentvolumereclaimpolicy:",
            "host:", "to:", "port:", "targetport:", "tls:", "weight:", "reclaimpolicy:",
            "http:", "paths:", "backend:", "annotations:", "pathtype:", "number:",
        )):
            return True
        if re.match(r"^[A-Za-z0-9_.\"'/-]+\s*:\s*", stripped):
            return True
        return line.startswith(("  ", "\t"))

    def _is_code_break_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.match(r"^##\s*Page\s+\d+\s*$", stripped, flags=re.IGNORECASE):
            return True
        if re.match(r"^-\s*(loader|chars):", stripped, flags=re.IGNORECASE):
            return True
        return False

    def _clean_code_block(self, block: str) -> str:
        lines = [line.rstrip() for line in block.replace("\r\n", "\n").split("\n")]
        cleaned: list[str] = []
        started = False

        for raw_line in lines:
            stripped = raw_line.strip()
            if self._is_code_break_line(stripped):
                continue
            is_codeish = (
                self._is_code_line(stripped)
                or self._looks_like_code_comment(stripped)
                or self._looks_like_code_continuation(stripped)
            )

            if not started:
                if not is_codeish:
                    continue
                started = True

            if started and stripped and not is_codeish:
                break

            cleaned.append(raw_line)

        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return "\n".join(cleaned).strip()

    def _merge_code_candidates(self, candidates: list[str]) -> list[str]:
        merged: list[str] = []
        for candidate in candidates:
            normalized_candidate = self._clean_code_block(candidate)
            if not normalized_candidate:
                continue
            if not merged:
                merged.append(normalized_candidate)
                continue

            prev = merged[-1]
            if self._is_subsequence_block(normalized_candidate, prev):
                continue
            if self._should_merge_code_blocks(prev, normalized_candidate):
                merged[-1] = self._merge_two_code_blocks(prev, normalized_candidate)
                continue
            merged.append(normalized_candidate)
        return merged

    def _should_merge_code_blocks(self, previous: str, current: str) -> bool:
        prev_kind = self._extract_code_block_kind(previous)
        current_kind = self._extract_code_block_kind(current)
        current_first = next((line.strip() for line in current.splitlines() if line.strip()), "")
        continuation_prefixes = (
            "spec:", "rules:", "- ", "path:", "pathtype:", "backend:", "service:", "port:",
            "number:", "http:", "paths:", "tls:", "to:", "weight:",
        )
        if current_first.casefold().startswith(continuation_prefixes):
            return True
        if prev_kind and current_kind and prev_kind == current_kind:
            return True
        if prev_kind and not current_kind:
            return True
        return False

    def _merge_two_code_blocks(self, previous: str, current: str) -> str:
        prev_lines = previous.splitlines()
        current_lines = current.splitlines()
        overlap = 0
        max_overlap = min(len(prev_lines), len(current_lines))
        for size in range(max_overlap, 0, -1):
            if prev_lines[-size:] == current_lines[:size]:
                overlap = size
                break
        merged = prev_lines + current_lines[overlap:]
        deduped: list[str] = []
        for line in merged:
            if deduped and deduped[-1] == line and not line.strip():
                continue
            deduped.append(line)
        return "\n".join(deduped).strip()

    def _is_subsequence_block(self, candidate: str, existing: str) -> bool:
        candidate_lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        existing_lines = [line.strip() for line in existing.splitlines() if line.strip()]
        if not candidate_lines:
            return True
        candidate_text = "\n".join(candidate_lines)
        existing_text = "\n".join(existing_lines)
        return candidate_text in existing_text

    def _looks_like_code_block(self, block: str) -> bool:
        lowered = block.casefold()
        indicators = (
            "apiversion:",
            "kind:",
            "metadata:",
            "spec:",
            "oc apply -f",
            "oc get ",
            "kubectl ",
            "route.openshift.io",
            "storage.k8s.io",
            "rbac.authorization.k8s.io",
            "host:",
            "provisioner:",
            "- path:",
            "backend:",
            "service:",
            "port:",
            "pathtype:",
            "number:",
        )
        return any(indicator in lowered for indicator in indicators)

    def build_answer_citation_payload(
        self,
        answer: str,
        context_items: list[dict],
        grounded_pages: list[dict],
        preferred_preview_source: str | None,
    ) -> list[dict]:
        if self.should_suppress_citations(answer):
            return []

        payload: list[dict] = []
        seen: set[tuple[str, int]] = set()
        items_by_name: dict[str, list[dict]] = defaultdict(list)
        for item in context_items:
            source_path = item["chunk"]["source_path"]
            items_by_name[Path(source_path).name].append(item)

        for file_name, cited_start, cited_end in self.extract_answer_citations(answer):
            for item in items_by_name.get(file_name, []):
                chunk = item["chunk"]
                source_path = chunk["source_path"]
                page_start = int(chunk["metadata"].get("page_start") or chunk["page_number"] or 1)
                page_end = int(chunk["metadata"].get("page_end") or page_start)
                if page_end < page_start:
                    page_end = page_start
                overlap_start = max(page_start, cited_start)
                overlap_end = min(page_end, cited_end)
                for page_number in range(overlap_start, overlap_end + 1):
                    self.append_citation_entry(
                        payload,
                        seen,
                        source_path=source_path,
                        page_number=page_number,
                        score=float(item["rerank_score"]),
                        chunk_id=chunk["chunk_id"],
                        origin="answer_text",
                    )

        if payload:
            return payload

        fallback_pages = [
            item
            for item in grounded_pages
            if item["source_path"] == preferred_preview_source
        ] or grounded_pages
        for item in fallback_pages[:3]:
            self.append_citation_entry(
                payload,
                seen,
                source_path=item["source_path"],
                page_number=int(item["page_number"]),
                score=float(item["score"]),
                chunk_id=None,
                origin="grounded_page",
            )
        return payload

    def should_suppress_citations(self, answer: str) -> bool:
        normalized = (answer or "").strip().casefold()
        if not normalized:
            return True
        negative_markers = (
            "업로드된 문서에서 관련 내용을 찾을 수 없습니다",
            "관련 내용을 찾기 어렵습니다",
            "다른 질문을 해주시거나",
            "관련 문서를 업로드해 주세요",
            "찾을 수 없습니다",
            "unable to find relevant content",
        )
        return any(marker.casefold() in normalized for marker in negative_markers)

    def append_citation_entry(
        self,
        payload: list[dict],
        seen: set[tuple[str, int]],
        source_path: str,
        page_number: int,
        score: float,
        chunk_id: str | None,
        origin: str,
    ) -> None:
        key = (source_path, page_number)
        if key in seen:
            return
        seen.add(key)
        payload.append(
            {
                "source_path": source_path,
                "file_name": Path(source_path).name,
                "page_number": page_number,
                "score": round(score, 4),
                "chunk_id": chunk_id,
                "origin": origin,
            }
        )

    def extract_answer_citations(self, answer: str) -> list[tuple[str, int, int]]:
        if not answer:
            return []

        citations: list[tuple[str, int, int]] = []
        grouped_pattern = re.compile(
            r"\[([^\[\]\n]+?\.pdf)\]\s*((?:p\.\d+(?:-\d+)?)(?:\s*,\s*p\.\d+(?:-\d+)?)*)",
            flags=re.IGNORECASE,
        )
        for match in grouped_pattern.finditer(answer):
            file_name = Path(match.group(1).strip()).name
            page_tokens = re.findall(r"p\.(\d+)(?:-(\d+))?", match.group(2), flags=re.IGNORECASE)
            for page_start_raw, page_end_raw in page_tokens:
                page_start = int(page_start_raw)
                page_end = int(page_end_raw or page_start)
                if page_end < page_start:
                    page_end = page_start
                citations.append((file_name, page_start, page_end))

        patterns = (
            r"\[([^\[\]\n]+?\.pdf)\s+p\.(\d+)(?:-(\d+))?\]",
            r"\[([^\[\]\n]+?\.pdf)\]\s*p\.(\d+)(?:-(\d+))?",
            r"\[([^\[\]\n]+?)\]\s*p\.(\d+)(?:-(\d+))?",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
                file_name = Path(match.group(1).strip()).name
                page_start = int(match.group(2))
                page_end = int(match.group(3) or page_start)
                if page_end < page_start:
                    page_end = page_start
                citations.append((file_name, page_start, page_end))
        deduped: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int, int]] = set()
        for citation in citations:
            if citation in seen:
                continue
            seen.add(citation)
            deduped.append(citation)
        return deduped

    def build_answer_aligned_preview_pages(
        self,
        answer_citations: list[dict],
        context_items: list[dict],
        preferred_preview_source: str | None,
        grounded_pages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        fallback_pages = self.retrieval_service.build_grounded_preview_pages(preferred_preview_source, grounded_pages)
        if not context_items or not answer_citations:
            return preferred_preview_source, fallback_pages
        preview_pages: list[dict] = []
        seen_pages: set[tuple[str, int]] = set()
        chosen_source: str | None = None

        for citation in answer_citations:
            source_path = citation["source_path"]
            if chosen_source is None:
                chosen_source = source_path
            if source_path != chosen_source:
                continue
            page_key = (source_path, int(citation["page_number"]))
            if page_key in seen_pages:
                continue
            preview_pages.append(
                {
                    "source_path": source_path,
                    "page_number": int(citation["page_number"]),
                    "score": round(float(citation["score"]), 4),
                }
            )
            seen_pages.add(page_key)
            if len(preview_pages) >= 3:
                return chosen_source, preview_pages

        if preview_pages:
            return chosen_source, preview_pages
        # fallback: grounded_pages에서 직접 preview 생성
        if fallback_pages:
            return preferred_preview_source, fallback_pages
        # 최종 fallback: grounded_pages top 3
        if grounded_pages:
            top_grounded = sorted(grounded_pages, key=lambda p: float(p.get("score", 0)), reverse=True)[:3]
            source = top_grounded[0].get("source_path") if top_grounded else preferred_preview_source
            return source, top_grounded
        return preferred_preview_source, []

    def build_source_line(self, answer_citations: list[dict]) -> str:
        citations_by_source: dict[str, list[int]] = defaultdict(list)
        for citation in answer_citations:
            citations_by_source[citation["file_name"]].append(int(citation["page_number"]))

        parts: list[str] = []
        for file_name, pages in citations_by_source.items():
            unique_pages = sorted(set(pages))
            page_ranges: list[str] = []
            start = unique_pages[0]
            end = unique_pages[0]
            for page_number in unique_pages[1:]:
                if page_number == end + 1:
                    end = page_number
                    continue
                page_ranges.append(f"p.{start}" if start == end else f"p.{start}-{end}")
                start = page_number
                end = page_number
            page_ranges.append(f"p.{start}" if start == end else f"p.{start}-{end}")
            parts.append(f"[{file_name}] " + ", ".join(page_ranges))
        return "\nSources: " + " | ".join(parts) if parts else ""

    def ensure_answer_source_line(
        self,
        answer: str,
        answer_citations: list[dict],
        use_retrieved_context: bool,
    ) -> str:
        if not answer or not use_retrieved_context or not answer_citations:
            return answer
        if self.should_suppress_citations(answer):
            return answer
        if self.extract_answer_citations(answer):
            return answer
        return answer.rstrip() + self.build_source_line(answer_citations)

    def build_context_payload(
        self,
        rewritten_query: str,
        response_mode: str,
        top_score: float,
        preferred_preview_source: str | None,
        preview_pages: list[dict],
        context_items: list[dict],
        grounded_pages: list[dict],
        answer_citations: list[dict],
        preview_finalized: bool = False,
    ) -> dict:
        return {
            "query": rewritten_query,
            "mode": response_mode,
            "top_score": round(top_score, 4),
            "preferred_preview_source": preferred_preview_source,
            "preview_pages": preview_pages,
            "source_grounding": self.retrieval_service.aggregate_source_grounding(grounded_pages),
            "grounded_pages": grounded_pages,
            "answer_citations": answer_citations,
            "preview_finalized": preview_finalized,
            "items": self.retrieval_service.build_context_items_payload(context_items),
        }

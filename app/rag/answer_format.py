from __future__ import annotations

import re
from pathlib import Path


class AnswerFormatMixin:
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
        matched_docs = [doc for doc in yaml_docs if self._extract_code_block_kind(doc) in requested_resource_kinds]
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
        return any(pattern.casefold() in normalized for pattern in patterns)

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
        return bool(re.match(r"^[A-Za-z0-9_.\"'/-]+\s*:\s*", stripped))

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
            is_codeish = self._is_code_line(stripped) or self._looks_like_code_comment(stripped) or self._looks_like_code_continuation(stripped)
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
            previous = merged[-1]
            if self._is_subsequence_block(normalized_candidate, previous):
                continue
            if self._should_merge_code_blocks(previous, normalized_candidate):
                merged[-1] = self._merge_two_code_blocks(previous, normalized_candidate)
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
        return "\n".join(candidate_lines) in "\n".join(existing_lines)

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

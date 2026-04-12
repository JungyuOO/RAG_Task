from __future__ import annotations

import re
from pathlib import Path

from app.rag.utils import normalize_query_keywords


class AnswerFormatMixin:
    NEGATIVE_RETRIEVED_MARKERS = (
        "제공된 문서에는",
        "문서에는",
        "관련 내용을 찾을 수 없습니다",
        "포함되어 있지 않습니다",
        "명시적인 목록은 없습니다",
        "구체적인 설명이 포함되어 있지 않습니다",
        "is not included in the provided document",
        "is not specifically described in the provided document",
    )
    LOW_SIGNAL_RETRIEVED_MARKERS = (
        "table of contents",
        "contents",
        "legal notice",
        "copyright",
        "creative commons",
        "all rights reserved",
        "registering an additional oauth client",
        "default oauth clients",
    )
    COMMAND_LINE_RE = re.compile(r"(?:(?:^|```(?:text|bash|shell)?\s*)\$?\s*)((?:oc|kubectl)\s+[^\n`]+)", re.IGNORECASE)
    SIMPLE_COMMAND_PATTERNS = (
        re.compile(r"^\s*(?:oc|kubectl)\s+get\s+[a-z0-9./-]+s?\b", re.IGNORECASE),
        re.compile(r"^\s*(?:oc|kubectl)\s+project(?:s)?\b", re.IGNORECASE),
        re.compile(r"^\s*(?:oc|kubectl)\s+status\b", re.IGNORECASE),
    )
    COMPLEX_COMMAND_MARKERS = ("|", "jq ", "grep ", "awk ", "xargs ", "exec ", "logs ", "rsh ", "debug ", "patch ", "replace ", "apply ")

    @staticmethod
    def _prose_source_text(chunk: dict) -> str:
        metadata = chunk.get("metadata") or {}
        retrieval_text = str(metadata.get("retrieval_text") or "").strip()
        if retrieval_text:
            return retrieval_text
        return str(chunk.get("text", "") or "")

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
        *,
        user_message: str = "",
        query_interpretation: dict | None = None,
    ) -> str | None:
        query_interpretation = query_interpretation or {}
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        generic_command_query = bool(query_interpretation.get("generic_command_query"))
        if not generic_command_query:
            procedure_answer = self._build_procedure_command_answer(
                context_items,
                user_message=user_message,
                query_interpretation=query_interpretation,
            )
            if procedure_answer:
                return procedure_answer
        if generic_command_query:
            snippet_limit = 2 if "yaml" in format_constraints else 1
        else:
            snippet_limit = 3
        snippets = self._collect_code_snippets(
            context_items,
            requested_resource_kinds=requested_resource_kinds,
            limit=snippet_limit,
            user_message=user_message,
            query_interpretation=query_interpretation,
        )

        if not snippets:
            return None

        query_interpretation = query_interpretation or {}
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        parts = ["문서에 나온 예시 코드를 그대로 정리하면 아래와 같습니다."]
        if generic_command_query:
            synthesized = self._build_generic_command_templates(
                requested_resource_kinds=requested_resource_kinds,
                format_constraints=format_constraints,
                query_interpretation=query_interpretation,
            )
            if synthesized:
                lead = snippets[0]
                page_label = (
                    f"p.{lead['page_start']}"
                    if lead["page_start"] == lead["page_end"]
                    else f"p.{lead['page_start']}-{lead['page_end']}"
                )
                parts.append(f"[{lead['file_name']}] {page_label}")
                for code in synthesized:
                    parts.append("```bash")
                    parts.append(code)
                    parts.append("```")
                return "\n\n".join(parts).strip()

        for snippet in snippets:
            snippet_code = self._canonicalize_generic_command_snippet(
                str(snippet["code"] or ""),
                query_interpretation=query_interpretation,
            )
            page_label = (
                f"p.{snippet['page_start']}"
                if snippet["page_start"] == snippet["page_end"]
                else f"p.{snippet['page_start']}-{snippet['page_end']}"
            )
            fence_language = snippet["code_language"] if snippet["code_language"] in {"yaml", "yml", "bash", "sh", "shell", "json"} else "yaml"
            parts.append(f"[{snippet['file_name']}] {page_label}")
            parts.append(f"```{fence_language}")
            parts.append(snippet_code)
            parts.append("```")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _canonicalize_generic_command_snippet(snippet: str, *, query_interpretation: dict | None) -> str:
        query_interpretation = query_interpretation or {}
        if not query_interpretation.get("generic_command_query"):
            return snippet
        resources = {str(value).casefold().strip() for value in query_interpretation.get("resources", []) if value}
        normalized = str(snippet or "").strip()
        lowered = normalized.casefold()
        if resources == {"pod"}:
            if lowered == "oc get pod":
                return "oc get pods"
            if lowered == "kubectl get pod":
                return "kubectl get pods"
        return normalized

    @staticmethod
    def build_generic_yaml_compare_answer(*, resource_kind: str = "resource") -> str:
        resource_label = resource_kind or "resource"
        return (
            "공식 문서 기준\n"
            f"문서에서 {resource_label} YAML을 볼 때는 보통 아래 형태를 기준으로 확인합니다.\n\n"
            "```bash\n"
            f"oc get {resource_label} <{resource_label}_name> -o yaml\n"
            "```\n\n"
            "```bash\n"
            f"oc describe {resource_label} <{resource_label}_name>\n"
            "```\n\n"
            "현재 OCP 기준\n"
            f"실제 YAML은 비교할 {resource_label} 이름을 먼저 특정해야 확인할 수 있습니다.\n\n"
            "비교 가이드\n"
            f"같은 {resource_label} 이름을 기준으로 공식 문서 예시와 실제 YAML을 나란히 비교해 보세요."
        )

    @staticmethod
    def _build_generic_command_templates(
        *,
        requested_resource_kinds: set[str] | None,
        format_constraints: set[str],
        query_interpretation: dict | None = None,
    ) -> list[str]:
        query_interpretation = query_interpretation or {}
        resource_kinds = {str(value).casefold().strip() for value in (requested_resource_kinds or set()) if value}
        resource_priority = ("pod", "deployment", "service", "route", "namespace", "project")
        resource = next((value for value in resource_priority if value in resource_kinds), "")
        if not resource:
            resource = next(iter(resource_kinds), "")
        normalized_keywords = {str(value).casefold().strip() for value in query_interpretation.get("normalized_keywords", []) if value}
        templates: list[str] = []

        if any(token in normalized_keywords for token in {"status", "상태", "현재", "결과"}):
            templates.extend([
                "oc status",
                "oc get pods",
            ])

        if "yaml" in format_constraints:
            if resource == "pod":
                templates.extend([
                    "oc get pod <pod_name> -o yaml",
                    "oc describe pod <pod_name>",
                ])
            elif resource:
                templates.extend([
                    f"oc get {resource} <name> -o yaml",
                    f"oc describe {resource} <name>",
                ])
            else:
                templates.extend([
                    "oc get <resource> <name> -o yaml",
                    "oc describe <resource> <name>",
                ])
        elif "cli" in format_constraints:
            if resource == "pod":
                templates.extend([
                    "oc get pods",
                    "oc get pods -o wide",
                ])
            elif resource in {"namespace", "project"}:
                templates.extend([
                    "oc project",
                    "oc projects",
                ])

        unique: list[str] = []
        for template in templates:
            normalized = template.strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique[:2]

    def _build_procedure_command_answer(
        self,
        context_items: list[dict],
        *,
        user_message: str,
        query_interpretation: dict | None,
    ) -> str | None:
        query_interpretation = query_interpretation or {}
        format_constraints = {str(value).casefold().strip() for value in query_interpretation.get("format_constraints", []) if value}
        response_shape = str(query_interpretation.get("response_shape") or "").casefold().strip()
        if "cli" not in format_constraints and response_shape != "code":
            return None

        pairs: list[dict[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        lead_file = ""
        lead_page_start = ""
        lead_page_end = ""
        aggregate_texts: list[str] = []

        for item in context_items[:3]:
            chunk = item["chunk"]
            metadata = chunk.get("metadata") or {}
            if not lead_file:
                lead_file = Path(str(chunk.get("source_path") or "")).name
                lead_page_start = str(metadata.get("page_start") or chunk.get("page_number") or 1)
                lead_page_end = str(metadata.get("page_end") or lead_page_start)
            if Path(str(chunk.get("source_path") or "")).name == lead_file:
                aggregate_texts.append(str(chunk.get("text", "") or ""))

            extracted_pairs = self._extract_label_command_pairs(str(chunk.get("text", "") or ""))
            for pair in extracted_pairs:
                key = (pair["label"].casefold(), pair["command"].casefold())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append(pair)
            if len(pairs) >= 3:
                break

        if aggregate_texts and len(pairs) < 2:
            merged_pairs = self._extract_label_command_pairs("\n".join(aggregate_texts))
            for pair in merged_pairs:
                key = (pair["label"].casefold(), pair["command"].casefold())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append(pair)

        neighborhood_pairs = self._extract_neighbor_page_pairs(
            context_items,
            query_interpretation=query_interpretation,
        )
        if self._score_label_command_pairs(neighborhood_pairs, query_interpretation) > self._score_label_command_pairs(pairs, query_interpretation):
            pairs = neighborhood_pairs

        if not pairs:
            return None

        page_label = f"p.{lead_page_start}" if lead_page_start == lead_page_end else f"p.{lead_page_start}-{lead_page_end}"
        lines = ["문서 절차 기준으로 보면 다음처럼 확인할 수 있습니다.", f"[{lead_file}] {page_label}"]
        for pair in pairs[:3]:
            lines.append(f"- `{pair['label']}` 는 `{pair['command']}` 로 확인할 수 있습니다.")
        return "\n\n".join(lines).strip()

    def _extract_neighbor_page_pairs(
        self,
        context_items: list[dict],
        *,
        query_interpretation: dict | None,
    ) -> list[dict[str, str]]:
        best_pairs: list[dict[str, str]] = []
        best_score = -1.0
        for item in context_items[:4]:
            chunk = item.get("chunk") or {}
            metadata = chunk.get("metadata") or {}
            source_path = Path(str(chunk.get("source_path") or ""))
            if source_path.suffix.lower() != ".md" or not source_path.exists():
                continue
            page_start = int(metadata.get("page_start") or chunk.get("page_number") or 1)
            page_end = int(metadata.get("page_end") or page_start)
            candidate_text = self._read_markdown_page_window(source_path, page_start, page_end)
            candidate_pairs = self._extract_label_command_pairs(candidate_text)
            candidate_score = self._score_label_command_pairs(candidate_pairs, query_interpretation)
            if candidate_score > best_score:
                best_pairs = candidate_pairs
                best_score = candidate_score
        return best_pairs

    @staticmethod
    def _read_markdown_page_window(source_path: Path, page_start: int, page_end: int) -> str:
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        pages = re.split(r"(?im)^##\s*Page\s+(\d+)\s*$", text)
        if len(pages) < 3:
            return text
        collected: list[str] = []
        target_pages = set(range(max(1, page_start - 1), page_end + 2))
        for index in range(1, len(pages), 2):
            try:
                page_number = int(str(pages[index]).strip())
            except ValueError:
                continue
            if page_number in target_pages:
                collected.append(str(pages[index + 1]))
        return "\n".join(collected).strip() if collected else text

    @staticmethod
    def _score_label_command_pairs(pairs: list[dict[str, str]], query_interpretation: dict | None) -> float:
        if not pairs:
            return -1.0
        query_interpretation = query_interpretation or {}
        strong_tokens = {
            str(token).casefold().strip()
            for token in query_interpretation.get("normalized_keywords", []) or []
            if len(str(token).strip()) >= 4 and str(token).casefold().strip() not in {"namespace", "project", "status", "command", "yaml", "pod"}
        }
        score = float(len(pairs)) * 0.25
        for pair in pairs:
            haystack = f"{pair['label']} {pair['command']}".casefold()
            score += sum(1.0 for token in strong_tokens if token in haystack)
            if "-n " in pair["command"]:
                score += 0.35
            if pair["label"].startswith("openshift-"):
                score += 0.45
        return score

    @staticmethod
    def _extract_label_command_pairs(text: str) -> list[dict[str, str]]:
        lines = [line.strip() for line in str(text or "").replace("\r\n", "\n").split("\n")]
        pairs: list[dict[str, str]] = []
        skip_labels = {
            "procedure",
            "procedures",
            "example",
            "examples",
            "command",
            "commands",
            "run the following command",
            "check the status of etcd pods.",
        }

        def _is_command(value: str) -> bool:
            normalized = value.lstrip("$").strip().casefold()
            return normalized.startswith(("oc ", "kubectl "))

        def _label_from_command(value: str) -> str:
            normalized = value.lstrip("$").strip()
            namespace_match = re.search(r"(?:^|\s)-n\s+([a-z0-9][a-z0-9-]*)\b", normalized, flags=re.IGNORECASE)
            if namespace_match:
                return namespace_match.group(1)
            return ""

        def _is_label(value: str) -> bool:
            lowered = value.casefold().strip(" :")
            if not lowered or lowered in skip_labels:
                return False
            if lowered.startswith("```"):
                return False
            if lowered.startswith(("oc ", "kubectl ", "$ oc ", "$ kubectl ")):
                return False
            if lowered.startswith(("check ", "run ", "use ", "verify ")):
                return False
            if len(lowered) > 64:
                return False
            return True

        index = 0
        while index < len(lines) - 1:
            label = lines[index]
            command = lines[index + 1]
            if _is_label(label) and _is_command(command):
                pairs.append(
                    {
                        "label": label.strip(" :"),
                        "command": command.lstrip("$").strip(),
                    }
                )
                index += 2
                continue
            if _is_command(label):
                inferred_label = _label_from_command(label)
                if inferred_label:
                    pairs.append(
                        {
                            "label": inferred_label,
                            "command": label.lstrip("$").strip(),
                        }
                    )
            index += 1
        return pairs

    def build_supporting_code_example(
        self,
        context_items: list[dict],
        requested_resource_kinds: set[str] | None = None,
        *,
        user_message: str = "",
        query_interpretation: dict | None = None,
    ) -> dict | None:
        snippets = self._collect_code_snippets(
            context_items,
            requested_resource_kinds=requested_resource_kinds,
            limit=1,
            min_chars=8,
            user_message=user_message,
            query_interpretation=query_interpretation,
        )
        if not snippets:
            return None
        snippet = snippets[0]
        fence_language = snippet["code_language"] if snippet["code_language"] in {"yaml", "yml", "bash", "sh", "shell", "json"} else "yaml"
        return {
            "type": "code",
            "title": "예시 코드",
            "source_path": str(context_items[0]["chunk"].get("source_path") or ""),
            "file_name": snippet["file_name"],
            "page_start": snippet["page_start"],
            "page_end": snippet["page_end"],
            "html_anchor": str((context_items[0]["chunk"].get("metadata") or {}).get("html_anchor") or ""),
            "block_anchor": str((context_items[0]["chunk"].get("metadata") or {}).get("primary_block_anchor") or ""),
            "language": fence_language,
            "content": snippet["code"],
        }

    def build_supporting_table_example(self, context_items: list[dict]) -> dict | None:
        table_answer = self.build_extractive_table_answer(context_items)
        if not table_answer:
            return None
        lines = [line for line in table_answer.splitlines() if line.strip()]
        if len(lines) <= 1:
            return None
        body = "\n".join(lines[1:])
        file_name = ""
        page_start = ""
        page_end = ""
        first = context_items[0]["chunk"] if context_items else {}
        if first:
            file_name = Path(str(first.get("source_path") or "")).name
            page_start = str(first.get("metadata", {}).get("page_start") or first.get("page_number") or 1)
            page_end = str(first.get("metadata", {}).get("page_end") or page_start)
        return {
            "type": "table",
            "title": "관련 표",
            "source_path": str(first.get("source_path") or ""),
            "file_name": file_name,
            "page_start": page_start,
            "page_end": page_end,
            "html_anchor": str((first.get("metadata") or {}).get("html_anchor") or ""),
            "block_anchor": str((first.get("metadata") or {}).get("primary_block_anchor") or ""),
            "content": body,
        }

    def _collect_code_snippets(
        self,
        context_items: list[dict],
        *,
        requested_resource_kinds: set[str] | None = None,
        limit: int = 3,
        min_chars: int = 24,
        user_message: str = "",
        query_interpretation: dict | None = None,
    ) -> list[dict[str, str]]:
        snippets: list[dict[str, str]] = []
        seen_blocks: set[str] = set()
        requested_resource_kinds = {str(value).casefold() for value in (requested_resource_kinds or set()) if value}
        query_interpretation = query_interpretation or {}
        keyword_tokens = normalize_query_keywords(user_message, query_interpretation.get("normalized_keywords", []))
        format_constraints = {str(value).casefold() for value in query_interpretation.get("format_constraints", []) if value}
        generic_command_query = bool(query_interpretation.get("generic_command_query"))

        for item in context_items:
            chunk = item["chunk"]
            text = str(chunk.get("text", "") or "")
            if not text.strip():
                continue
            for command in self._extract_command_candidates(text, requested_resource_kinds):
                normalized = re.sub(r"\s+", " ", command).strip().casefold()
                if len(normalized) < 8 or normalized in seen_blocks:
                    continue
                seen_blocks.add(normalized)
                snippets.append(
                    {
                        "file_name": Path(chunk["source_path"]).name,
                        "page_start": str(chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                        "page_end": str(chunk["metadata"].get("page_end") or chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                        "code": command.strip(),
                        "code_language": "bash",
                        "snippet_score": self._score_code_snippet(
                            command.strip(),
                            item,
                            keyword_tokens=keyword_tokens,
                            format_constraints=format_constraints,
                            requested_resource_kinds=requested_resource_kinds,
                            generic_command_query=generic_command_query,
                        ),
                    }
                )
            for block in self._extract_code_candidates(text):
                normalized_blocks = self._split_and_filter_code_blocks(block, requested_resource_kinds)
                for normalized_block in normalized_blocks:
                    normalized = re.sub(r"\s+", " ", normalized_block).strip().casefold()
                    if "```" in normalized_block:
                        continue
                    if len(normalized) < min_chars or normalized in seen_blocks:
                        continue
                    seen_blocks.add(normalized)
                    snippets.append(
                        {
                            "file_name": Path(chunk["source_path"]).name,
                            "page_start": str(chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                            "page_end": str(chunk["metadata"].get("page_end") or chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                            "code": normalized_block.strip(),
                            "code_language": str(chunk["metadata"].get("code_language", "") or ""),
                            "snippet_score": self._score_code_snippet(
                                normalized_block.strip(),
                                item,
                                keyword_tokens=keyword_tokens,
                                format_constraints=format_constraints,
                                requested_resource_kinds=requested_resource_kinds,
                                generic_command_query=generic_command_query,
                            ),
                        }
                    )
        snippets.sort(
            key=lambda item: (
                -float(item.get("snippet_score", 0.0)),
                len(str(item.get("code") or "")),
                str(item.get("file_name") or ""),
            )
        )
        return snippets[:limit]

    def _score_code_snippet(
        self,
        snippet: str,
        item: dict,
        *,
        keyword_tokens: list[str],
        format_constraints: set[str],
        requested_resource_kinds: set[str],
        generic_command_query: bool,
    ) -> float:
        lowered = str(snippet or "").casefold()
        source_name = Path(str(item["chunk"].get("source_path") or "")).name.casefold()
        score = float(item.get("final_retrieval_score", item.get("rerank_score", 0.0)))
        if lowered.startswith(("oc ", "kubectl ")):
            score += 0.35
        if "cli" in format_constraints and lowered.startswith(("oc ", "kubectl ")):
            score += 0.25
        elif "cli" in format_constraints:
            score -= 0.20
        if any(pattern.search(snippet) for pattern in self.SIMPLE_COMMAND_PATTERNS):
            score += 0.28
        if any(marker in lowered for marker in self.COMPLEX_COMMAND_MARKERS):
            score -= 0.22
        if "yaml" in format_constraints and ("-o yaml" in lowered or " yaml" in lowered or "manifest" in lowered):
            score += 0.30
            if lowered.startswith(("oc ", "kubectl ")) and "-o yaml" in lowered:
                score += 0.32
            if lowered.startswith(("oc describe", "kubectl describe")):
                score += 0.22
        elif "yaml" in format_constraints and "cli" in format_constraints:
            score -= 0.15
        elif "yaml" in format_constraints:
            score -= 0.28
        if requested_resource_kinds:
            resource_hits = 0
            for kind in requested_resource_kinds:
                if kind and (f" {kind}" in lowered or f" {kind}s" in lowered or f"/{kind}" in lowered):
                    resource_hits += 1
            score += min(resource_hits * 0.18, 0.36)
        keyword_hits = 0
        for token in keyword_tokens:
            normalized = str(token).casefold().strip()
            if len(normalized) < 3:
                continue
            if normalized in {"yaml", "cli", "command", "example"}:
                continue
            if normalized in lowered:
                keyword_hits += 1
        score += min(keyword_hits * 0.08, 0.32)
        if any(token in keyword_tokens for token in ("namespace", "project")) and any(marker in lowered for marker in (" project", "namespace", "-n ", "--namespace")):
            score += 0.22
        if any(token in keyword_tokens for token in ("status", "상태")) and lowered.startswith(("oc status", "kubectl get all")):
            score += 0.30
        if any(token in keyword_tokens for token in ("pod", "pods")) and re.search(r"\bget\s+pods?\b", lowered):
            score += 0.22
        if generic_command_query:
            if "cli_tools" in source_name:
                score += 0.90
            if any(token in keyword_tokens for token in ("namespace", "project")):
                if lowered.startswith(("oc project", "oc projects")):
                    score += 0.60
                if re.search(r"\bdeployment\b", lowered):
                    score -= 0.35
            if any(token in keyword_tokens for token in ("pod", "pods")):
                if re.search(r"\bget\s+pods?\b", lowered):
                    score += 0.45
                    if lowered.strip() in {"oc get pods", "kubectl get pods"}:
                        score += 0.80
                    if lowered.strip() in {"oc get pod", "kubectl get pod"}:
                        score -= 0.25
                    if lowered.strip() in {"oc get pod", "oc get pods", "kubectl get pod", "kubectl get pods"}:
                        score += 0.55
                    if "-o wide" in lowered:
                        score += 0.30
                    if re.search(r"\bget\s+pods?\b\s+[a-z0-9_.-]+", lowered):
                        score -= 0.75
                if "-o yaml" in lowered and "yaml" not in keyword_tokens:
                    score -= 0.35
            if "yaml" in keyword_tokens:
                if "-o yaml" in lowered or lowered.startswith(("oc describe", "kubectl describe")):
                    score += 0.35
                if lowered.startswith(("oc create", "oc apply")):
                    score -= 0.35
                if lowered.startswith(("oc edit", "kubectl edit")):
                    score -= 0.50
                if re.search(r"\bget\s+pods?\b\s+[a-z0-9_.-]+", lowered) and "-o yaml" in lowered:
                    score += 0.20
            if any(marker in lowered for marker in ("-l ", "--selector", "jsonpath", "app.kubernetes.io", "cert-manager", "workshop")):
                score -= 0.55
            if re.search(r"-n\s+[a-z0-9-]+", lowered) and "<" not in lowered:
                score -= 0.40
            if re.search(r"\btest\b", lowered):
                score -= 0.20
            if re.search(r"\b(?:busybox|parksmap|router-default|nginx|cert-manager)\b", lowered):
                score -= 0.35
            if lowered.startswith(("oc get pod ", "oc get pods ")) and "-o yaml" not in lowered and "-o wide" not in lowered:
                extra = lowered.split()
                if len(extra) > 4:
                    score -= 0.60
            if "<" in snippet and ">" in snippet:
                score += 0.08
        if len(snippet) > 80 and lowered.startswith(("oc ", "kubectl ")):
            score -= min((len(snippet) - 80) / 180.0, 0.35)
        if len(snippet) > 220:
            score -= min((len(snippet) - 220) / 400.0, 0.25)
        return score

    def _extract_command_candidates(self, text: str, requested_resource_kinds: set[str] | None = None) -> list[str]:
        requested_resource_kinds = {str(value).casefold() for value in (requested_resource_kinds or set()) if value}
        matches: list[str] = []
        for match in self.COMMAND_LINE_RE.findall(text or ""):
            command = re.sub(r"\s+", " ", str(match).strip())
            command = command.lstrip("$").strip()
            if not command:
                continue
            lowered = command.casefold()
            if requested_resource_kinds:
                if not any(kind in lowered or f"{kind}s" in lowered for kind in requested_resource_kinds):
                    continue
            if command not in matches:
                matches.append(command)
        return matches

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

    def _strip_rewrite_meta_commentary(self, answer: str) -> str:
        sanitized = answer
        patterns = [
            r"(?im)^\s*제시해주신 초안의 핵심 의미와 내용을 유지하면서.*$",
            r"(?im)^\s*문맥이 중복되고 어색한 부분을 정리하여.*$",
            r"(?im)^\s*자연스러운 한국어로 다시 작성했습니다\.?\s*$",
            r"(?im)^\s*초안을 바탕으로.*다시 정리.*$",
            r"(?im)^\s*초안의 의미를 유지하면서.*$",
            r"(?im)^\s*요청하신 대로.*다시 정리한 답변.*$",
            r"(?im)^\s*다음은 .*다시 작성한.*$",
            r"(?im)^\s*아래는 .*재작성한.*$",
        ]
        for pattern in patterns:
            sanitized = re.sub(pattern, "", sanitized)
        return sanitized.strip()

    def looks_like_negative_retrieved_answer(self, answer: str) -> bool:
        normalized = (answer or "").strip().casefold()
        if not normalized:
            return False
        return any(marker.casefold() in normalized for marker in self.NEGATIVE_RETRIEVED_MARKERS)

    def looks_like_low_signal_retrieved_answer(self, answer: str) -> bool:
        normalized = (answer or "").strip().casefold()
        if not normalized:
            return False
        if any(marker in normalized for marker in self.LOW_SIGNAL_RETRIEVED_MARKERS):
            return True
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        toc_like = 0
        for line in lines[:8]:
            if self._looks_like_toc_line(line):
                toc_like += 1
        return toc_like >= 2

    def build_extractive_text_answer(self, context_items: list[dict]) -> str | None:
        bullets: list[str] = []
        seen: set[str] = set()

        for item in context_items[:4]:
            chunk = item["chunk"]
            metadata = chunk.get("metadata") or {}
            if metadata.get("is_toc"):
                continue
            text = self._prose_source_text(chunk)
            if not text.strip():
                continue
            cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
            cleaned = re.sub(r"(?m)^##\s*Page\s+\d+\s*$", " ", cleaned)
            cleaned = re.sub(r"(?m)^-\s*(loader|chars):.*$", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                continue

            segments = re.split(r"(?<=[.!?다요])\s+", cleaned)
            for segment in segments:
                line = segment.strip(" -*")
                if len(line) < 18:
                    continue
                if self._looks_like_toc_line(line):
                    continue
                lowered = line.casefold()
                if lowered in seen:
                    continue
                seen.add(lowered)
                bullets.append(line)
                if len(bullets) >= 6:
                    break
            if len(bullets) >= 6:
                break

        if not bullets:
            return None

        lines = ["문서 기준으로 정리하면 다음과 같습니다."]
        for bullet in bullets:
            lines.append(f"- {bullet}")
        return "\n\n".join(lines).strip()

    def build_extractive_compare_answer(self, context_items: list[dict]) -> str | None:
        grouped: dict[str, list[str]] = {"official_ocp": [], "customer_generated": []}
        seen: dict[str, set[str]] = {"official_ocp": set(), "customer_generated": set()}

        for item in context_items[:6]:
            chunk = item["chunk"]
            metadata = chunk.get("metadata") or {}
            group = str(metadata.get("document_group") or "")
            if metadata.get("is_toc"):
                continue
            if group not in grouped:
                group = "customer_generated" if str(metadata.get("doc_type") or "") == "operation_manual" else "official_ocp"
            text = self._prose_source_text(chunk)
            if not text.strip():
                continue
            cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
            cleaned = re.sub(r"(?m)^##\s*Page\s+\d+\s*$", " ", cleaned)
            cleaned = re.sub(r"(?m)^-\s*(loader|chars):.*$", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                continue
            for segment in re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=니다\.)\s+", cleaned):
                line = segment.strip(" -*")
                if len(line) < 18:
                    continue
                if self._looks_like_toc_line(line):
                    continue
                lowered = line.casefold()
                if lowered in seen[group]:
                    continue
                seen[group].add(lowered)
                grouped[group].append(line)
                if len(grouped[group]) >= 3:
                    break

        parts: list[str] = []
        if grouped["official_ocp"]:
            parts.append("공식 문서에서는 다음과 같이 설명합니다.\n\n- " + "\n- ".join(grouped["official_ocp"][:3]))
        if grouped["customer_generated"]:
            parts.append("고객사 메뉴얼에서는 다음과 같이 설명합니다.\n\n- " + "\n- ".join(grouped["customer_generated"][:3]))
        if not parts:
            return None
        return "\n\n".join(parts).strip()

    def _sanitize_retrieved_answer(self, answer: str) -> str:
        sanitized = answer.replace("\r\n", "\n").strip()
        sanitized = self._strip_rewrite_meta_commentary(sanitized)
        sanitized = re.sub(r"(?im)^\s*(?:제공해주신 초안.*|자연스러운 한국어로 정리한 답변은 다음과 같습니다\.?|문맥을 자연스럽게 연결.*답변은 다음과 같습니다\.?)\s*", "", sanitized)
        sanitized = re.sub(r"(?m)^\s*---+\s*$", "", sanitized)
        sanitized = sanitized.replace("• - ", "- ").replace("• ", "- ")
        sanitized = re.sub(r"\[\d+\]", "", sanitized)
        sanitized = re.sub(
            r"(?:^|[\s,])\[[^\]\n]+?\.pdf\]\s*p\.\d+(?:-\d+)?(?:\s*,\s*p\.\d+(?:-\d+)?)*",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        lines: list[str] = []
        for raw_line in sanitized.split("\n"):
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            if self._is_retrieved_mode_negative_artifact(line):
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

    def _sanitize_general_answer(self, answer: str) -> str:
        sanitized = answer.replace("\r\n", "\n").strip()
        sanitized = self._strip_rewrite_meta_commentary(sanitized)
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
        return any(pattern.casefold() in normalized for pattern in self.NEGATIVE_RETRIEVED_MARKERS)

    def _looks_like_toc_line(self, line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.casefold()
        if re.match(r"^\d+(?:\.\d+){1,4}\.?\s+", stripped):
            return True
        if stripped.endswith("절") or " 절" in stripped:
            return True
        if "table of contents" in lowered or lowered == "contents":
            return True
        if re.search(r"\bp\.\d+\b", lowered):
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

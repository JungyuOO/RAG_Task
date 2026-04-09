from __future__ import annotations

import re

from app.rag.chunking import MarkdownBlock
from app.rag.types import Chunk, Document
from app.rag.utils import normalize_text, stable_hash, tokenize


class StructuredMarkdownChunkerSupport:
    TOC_SECTION_MARKERS = ("table of contents", "contents", "목차")
    OVERVIEW_SECTION_MARKERS = ("overview", "introduction", "about", "개요", "소개")
    PROCEDURE_MARKERS = ("step", "steps", "procedure", "procedures", "절차", "단계", "순서")

    def _parse_annotated_markdown_blocks(self, annotated_lines: list[tuple[str, int]]) -> list[MarkdownBlock]:
        blocks: list[MarkdownBlock] = []
        for section in self._split_annotated_sections(annotated_lines):
            lines = [line.strip() for line, _page in section if line.strip()]
            pages = [page for line, page in section if line.strip()]
            if not lines or not pages:
                continue

            page_start = min(pages)
            page_end = max(pages)
            if len(lines) == 1 and (lines[0].startswith("#") or (len(lines[0]) <= 40 and lines[0].endswith(":"))):
                normalized = normalize_text(lines[0].lstrip("#").strip())
                if normalized:
                    heading_level = len(lines[0]) - len(lines[0].lstrip("#")) if lines[0].startswith("#") else 1
                    blocks.append(
                        MarkdownBlock(
                            text=normalized,
                            page_start=page_start,
                            page_end=page_end,
                            kind="heading",
                            heading_level=max(heading_level, 1),
                        )
                    )
                continue

            if self._is_code_block(lines):
                blocks.extend(self._split_code_block(lines, page_start, page_end))
                continue
            if self._is_table_block(lines):
                blocks.extend(self._split_table_block(lines, page_start, page_end))
                continue
            if all(self._is_list_line(line) for line in lines):
                blocks.append(
                    MarkdownBlock(
                        text="\n".join(lines).strip(),
                        page_start=page_start,
                        page_end=page_end,
                        kind="list",
                    )
                )
                continue

            normalized = normalize_text(" ".join(lines))
            if normalized:
                blocks.append(
                    MarkdownBlock(
                        text=normalized,
                        page_start=page_start,
                        page_end=page_end,
                        kind="paragraph",
                    )
                )
        return blocks

    def _split_annotated_sections(self, annotated_lines: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
        sections: list[list[tuple[str, int]]] = []
        current_lines: list[tuple[str, int]] = []
        in_code_block = False

        for raw_line, page_number in annotated_lines:
            line = raw_line.rstrip()
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_lines.append((line, page_number))
                continue
            if not in_code_block and not line.strip():
                if current_lines:
                    sections.append(current_lines)
                    current_lines = []
                continue
            current_lines.append((line, page_number))

        if current_lines:
            sections.append(current_lines)
        return [section for section in sections if any(line.strip() for line, _page in section)]

    def _parse_markdown_blocks(self, body_text: str, page_number: int) -> list[MarkdownBlock]:
        blocks: list[MarkdownBlock] = []
        for section in self._split_markdown_sections(body_text):
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if not lines:
                continue
            if len(lines) == 1 and (lines[0].startswith("#") or (len(lines[0]) <= 40 and lines[0].endswith(":"))):
                normalized = normalize_text(lines[0].lstrip("#").strip())
                if normalized:
                    heading_level = len(lines[0]) - len(lines[0].lstrip("#")) if lines[0].startswith("#") else 1
                    blocks.append(
                        MarkdownBlock(
                            text=normalized,
                            page_start=page_number,
                            page_end=page_number,
                            kind="heading",
                            heading_level=max(heading_level, 1),
                        )
                    )
                continue
            if self._is_code_block(lines):
                blocks.extend(self._split_code_block(lines, page_number, page_number))
                continue
            if self._is_table_block(lines):
                blocks.extend(self._split_table_block(lines, page_number, page_number))
                continue
            if all(self._is_list_line(line) for line in lines):
                blocks.append(
                    MarkdownBlock(
                        text="\n".join(lines).strip(),
                        page_start=page_number,
                        page_end=page_number,
                        kind="list",
                    )
                )
                continue

            normalized = normalize_text(" ".join(lines))
            if normalized:
                blocks.append(
                    MarkdownBlock(
                        text=normalized,
                        page_start=page_number,
                        page_end=page_number,
                        kind="paragraph",
                    )
                )
        return blocks

    def _split_markdown_sections(self, body_text: str) -> list[str]:
        sections: list[str] = []
        current_lines: list[str] = []
        in_code_block = False

        for raw_line in body_text.splitlines():
            line = raw_line.rstrip()
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue
            if not in_code_block and not line.strip():
                if current_lines:
                    sections.append("\n".join(current_lines).strip())
                    current_lines = []
                continue
            # 헤딩 라인 앞에서 강제로 섹션 분리 (빈 줄 없이 헤딩이 붙어있어도 경계 생성)
            if not in_code_block and line.lstrip().startswith("#") and current_lines:
                sections.append("\n".join(current_lines).strip())
                current_lines = []
            current_lines.append(line)

        if current_lines:
            sections.append("\n".join(current_lines).strip())
        return [section for section in sections if section]

    def _split_code_block(self, lines: list[str], page_start: int, page_end: int) -> list[MarkdownBlock]:
        max_lines = 40
        fence = lines[0]
        closing = lines[-1]
        code_lines = lines[1:-1]

        if len(code_lines) <= max_lines:
            text = "\n".join(line.rstrip() for line in lines).strip()
            return [MarkdownBlock(text=text, page_start=page_start, page_end=page_end, kind="code")]

        result = []
        for index in range(0, len(code_lines), max_lines):
            group = code_lines[index:index + max_lines]
            text = "\n".join([fence] + [line.rstrip() for line in group] + [closing]).strip()
            result.append(MarkdownBlock(text=text, page_start=page_start, page_end=page_end, kind="code"))
        return result

    def _split_table_block(self, lines: list[str], page_start: int, page_end: int) -> list[MarkdownBlock]:
        if len(lines) < 3:
            text = "\n".join(line.rstrip() for line in lines).strip()
            return [MarkdownBlock(text=text, page_start=page_start, page_end=page_end, kind="table")]

        header = lines[0].rstrip()
        sep_idx = next((index for index, line in enumerate(lines[1:], 1) if re.match(r"^\|?[\s:|\-]+\|?$", line)), None)
        if sep_idx is None:
            text = "\n".join(line.rstrip() for line in lines).strip()
            return [MarkdownBlock(text=text, page_start=page_start, page_end=page_end, kind="table")]

        separator = lines[sep_idx].rstrip()
        data_rows = [line.rstrip() for line in lines[sep_idx + 1:]]
        header_block = f"{header}\n{separator}"
        header_chars = len(header_block) + 1
        budget = max(self.chunk_size - header_chars, len(header) * 3)

        groups: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for row in data_rows:
            if current and current_chars + len(row) + 1 > budget:
                groups.append(current)
                current = [row]
                current_chars = len(row)
            else:
                current.append(row)
                current_chars += len(row) + 1
        if current:
            groups.append(current)

        if len(groups) <= 1:
            text = "\n".join(line.rstrip() for line in lines).strip()
            return [MarkdownBlock(text=text, page_start=page_start, page_end=page_end, kind="table")]

        result = []
        for group in groups:
            text = header_block + "\n" + "\n".join(group)
            result.append(MarkdownBlock(text=text, page_start=page_start, page_end=page_end, kind="table"))
        return result

    def _is_list_line(self, line: str) -> bool:
        return bool(re.match(r"^(?:[-*]\s+|\d+\.\s+)", line))

    def _is_table_block(self, lines: list[str]) -> bool:
        if len(lines) < 2:
            return False
        pipe_lines = [line for line in lines if "|" in line]
        if len(pipe_lines) != len(lines):
            return False
        separator_pattern = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
        return any(separator_pattern.match(line) for line in lines)

    def _is_code_block(self, lines: list[str]) -> bool:
        return len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```")

    def _blocks_from_documents(self, documents: list[Document]) -> list[MarkdownBlock]:
        blocks: list[MarkdownBlock] = []
        for document in documents:
            text = normalize_text(document.text)
            if not text:
                continue
            page_number = document.page_number or 0
            blocks.append(
                MarkdownBlock(
                    text=text,
                    page_start=page_number,
                    page_end=page_number,
                    kind="paragraph",
                )
            )
        return blocks

    def _build_chunk(self, doc_id: str, source_path: str, blocks: list[MarkdownBlock], order: int) -> Chunk:
        chunk_text = "\n\n".join(block.text for block in blocks).strip()
        page_start = blocks[0].page_start
        page_end = blocks[-1].page_end
        chunk_id = stable_hash(f"{doc_id}:{order}:{page_start}:{page_end}:{chunk_text[:40]}")
        section_path_parts = next((list(block.heading_path) for block in reversed(blocks) if block.heading_path), [])
        nearest_heading = section_path_parts[-1] if section_path_parts else ""
        metadata = {
            "page_start": page_start,
            "page_end": page_end,
            "block_types": ",".join(sorted({block.kind for block in blocks})),
            "block_count": len(blocks),
            "section_title": nearest_heading,
            "section_path": " > ".join(section_path_parts),
            "nearest_heading": nearest_heading,
            "parent_headings": section_path_parts[:-1],
        }
        metadata.update(self._infer_structure_flags(chunk_text, metadata))
        if any(block.kind == "code" for block in blocks):
            metadata.update(self._infer_code_metadata(chunk_text))
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_path=source_path,
            text=chunk_text,
            tokens=tokenize(chunk_text),
            page_number=page_start if page_start == page_end else None,
            metadata=metadata,
        )

    def _infer_structure_flags(self, text: str, metadata: dict) -> dict[str, bool]:
        normalized = text.replace("\r\n", "\n")
        lowered = normalized.casefold()
        section_title = str(metadata.get("section_title", "") or "").casefold()
        section_path = str(metadata.get("section_path", "") or "").casefold()
        block_types = {value.strip().casefold() for value in str(metadata.get("block_types", "")).split(",") if value.strip()}
        page_start = int(metadata.get("page_start") or 0)

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        short_lines = [line for line in lines if len(line) <= 90]
        numbered_heading_lines = [
            line
            for line in lines
            if re.match(r"^\d+(?:\.\d+){1,4}\.?\s+", line)
        ]
        has_contents_heading = any(marker in section_title or marker in section_path or marker in lowered for marker in self.TOC_SECTION_MARKERS)
        is_toc = bool(
            has_contents_heading
            or (
                len(numbered_heading_lines) >= 2
                and len(short_lines) >= max(2, len(lines) // 2)
                and "code" not in block_types
                and "table" not in block_types
            )
        )

        overview_markers = self.OVERVIEW_SECTION_MARKERS
        is_overview = bool(
            not is_toc
            and any(marker in section_title or marker in section_path for marker in overview_markers)
        )
        is_intro = bool(
            not is_toc
            and page_start <= 3
            and (
                any(marker in section_title or marker in section_path for marker in overview_markers)
                or (len(lines) <= 4 and len(short_lines) >= max(1, len(lines) - 1))
            )
        )
        is_procedure = bool(
            any(marker in section_title or marker in section_path for marker in self.PROCEDURE_MARKERS)
            or "list" in block_types
            or any(re.match(r"^(?:\d+\.\s+|[-*]\s+)", line) for line in lines[:6])
        )
        return {
            "is_toc": is_toc,
            "is_intro": is_intro,
            "is_overview": is_overview,
            "is_procedure": is_procedure,
        }

    def _infer_code_metadata(self, text: str) -> dict[str, str | list[str]]:
        normalized = text.replace("\r\n", "\n")
        lowered = normalized.casefold()
        fence_match = re.search(r"```([\w+-]+)", normalized)
        fence_language = fence_match.group(1).strip().lower() if fence_match else ""
        kind_match = re.search(r"(?im)^\s*kind:\s*([A-Za-z0-9_-]+)\s*$", normalized)
        resource_kind = kind_match.group(1) if kind_match else ""

        signals: list[str] = []
        if resource_kind:
            signals.append(resource_kind)
        for marker in ("ConfigMap", "Secret", "Pod", "Deployment", "Service", "PersistentVolume", "PersistentVolumeClaim"):
            if marker.casefold() in lowered and marker not in signals:
                signals.append(marker)
        for marker in ("apiVersion", "kind", "metadata", "oc", "kubectl"):
            if marker.casefold() in lowered and marker not in signals:
                signals.append(marker)

        code_language = fence_language
        if not code_language:
            if "oc " in lowered or "kubectl " in lowered or "helm " in lowered:
                code_language = "bash"
            elif "apiversion:" in lowered or "kind:" in lowered:
                code_language = "yaml"

        code_subtype = ""
        if "apiversion:" in lowered and "kind:" in lowered:
            code_subtype = "k8s_manifest"
        elif "oc " in lowered or "kubectl " in lowered or "helm " in lowered:
            code_subtype = "cli_command"
        elif code_language in {"yaml", "yml"}:
            code_subtype = "yaml_snippet"

        return {
            "code_language": code_language,
            "code_subtype": code_subtype,
            "code_signals": signals[:8],
        }

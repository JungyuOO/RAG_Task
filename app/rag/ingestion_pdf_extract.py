from __future__ import annotations

import re
from pathlib import Path

from app.rag.utils import extracted_markdown_path
from app.rag.types import Document
from app.rag.utils import normalize_text


_YAML_START_RE = re.compile(
    r"^(apiVersion|kind|metadata|spec|rules|subjects|roleRef|data|stringData"
    r"|items|parameters|provisioner|template|containers|volumes|ports"
    r"|env|resources|status|selector|replicas|strategy|storage"
    r"|certificate|description)\s*:",
)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+\.(?:\d+\.?)*)\s+(.+)$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")


class PdfExtractionSupport:
    @staticmethod
    def _should_merge_spans(prev: str, curr: str) -> bool:
        if not prev or not curr:
            return True
        last_char = prev[-1]
        first_char = curr[0]
        if last_char in ".,;:!?":
            return True
        if last_char in "([{" or first_char in ")]}":
            return True
        if last_char == "-" or first_char == "-":
            return True
        return False

    def _detect_footer_pattern(self, pdf) -> str | None:
        if len(pdf) < 2:
            return None

        candidates: dict[str, int] = {}
        for page in pdf:
            lines = page.get_text("text").strip().splitlines()
            if len(lines) < 3:
                continue
            last = lines[-1].strip()
            second_last = lines[-2].strip()
            if last.isdigit() and 1 <= len(second_last) <= 60:
                candidates[second_last] = candidates.get(second_last, 0) + 1

        if not candidates:
            return None

        most_common = max(candidates, key=candidates.get)
        if candidates[most_common] >= max(2, len(pdf) // 2):
            return most_common
        return None

    def _extract_structured_page(self, page, footer_pattern: str | None) -> str:
        table_rects: list[tuple] = []
        table_markdowns: list[tuple[float, str]] = []
        try:
            tables = page.find_tables()
            for table in tables.tables:
                table_rects.append(table.bbox)
                markdown = self._table_to_markdown(table)
                if markdown:
                    table_markdowns.append((table.bbox[1], markdown))
        except Exception:
            pass

        blocks = page.get_text("dict")["blocks"]
        text_segments: list[tuple[float, str]] = []
        for block in blocks:
            if "lines" not in block:
                continue

            bx0, by0, bx1, by1 = block["bbox"]
            if self._overlaps_any_table(bx0, by0, bx1, by1, table_rects):
                continue

            block_lines = self._extract_block_lines(block)
            if block_lines:
                text_segments.append((by0, "\n".join(block_lines)))

        all_segments = text_segments + table_markdowns
        all_segments.sort(key=lambda item: item[0])
        raw_text = "\n\n".join(segment[1] for segment in all_segments)

        processed = self._remove_footer(raw_text, footer_pattern)
        processed = self._restore_headings(processed)
        processed = self._detect_code_blocks(processed)
        return processed.strip()

    def _overlaps_any_table(self, x0: float, y0: float, x1: float, y1: float, table_rects: list[tuple]) -> bool:
        for _tx0, ty0, _tx1, ty1 in table_rects:
            overlap_top = max(y0, ty0)
            overlap_bottom = min(y1, ty1)
            overlap_height = max(0, overlap_bottom - overlap_top)
            block_height = max(1, y1 - y0)
            if overlap_height / block_height > 0.5:
                return True
        return False

    def _extract_block_lines(self, block: dict) -> list[str]:
        lines: list[str] = []
        page_left = 72.0

        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue

            line_text = "".join(span["text"] for span in spans).rstrip()
            if not line_text.strip():
                continue

            x_offset = spans[0]["origin"][0]
            indent_level = max(0, int((x_offset - page_left) / 18))
            stripped = line_text.strip()
            if indent_level > 0 and not re.match(r"^(\d+\.\s|[-*]\s|#)", stripped):
                line_text = "  " * indent_level + stripped
            else:
                line_text = stripped
            lines.append(line_text)

        return lines

    def _table_to_markdown(self, table) -> str:
        rows = table.extract()
        if not rows:
            return ""

        cleaned: list[list[str]] = []
        for row in rows:
            cleaned.append([(cell or "").replace("\n", " ").strip() for cell in row])

        if not cleaned:
            return ""

        col_count = max(len(row) for row in cleaned)
        for row in cleaned:
            while len(row) < col_count:
                row.append("")

        header = cleaned[0]
        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in cleaned[1:]:
            markdown_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(markdown_lines)

    def _remove_footer(self, text: str, footer_pattern: str | None) -> str:
        lines = text.splitlines()
        if len(lines) < 2:
            return text

        while lines and lines[-1].strip().isdigit():
            lines.pop()
        if footer_pattern and lines and lines[-1].strip() == footer_pattern:
            lines.pop()
        return "\n".join(lines)

    def _restore_headings(self, text: str) -> str:
        result_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                result_lines.append("")
                continue

            match = _NUMBERED_HEADING_RE.match(stripped)
            if match:
                number_part = match.group(1)
                title_part = match.group(2).strip()
                if len(title_part) <= 50 and not title_part.endswith((".", "다.", "함.")):
                    depth = number_part.count(".")
                    level = min(depth + 2, 4)
                    result_lines.append(f"{'#' * level} {number_part} {title_part}")
                    continue
            result_lines.append(line)
        return "\n".join(result_lines)

    def _detect_code_blocks(self, text: str) -> str:
        lines = text.splitlines()
        result: list[str] = []
        index = 0

        while index < len(lines):
            line = lines[index]
            if line.strip().startswith("```"):
                result.append(line)
                index += 1
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    result.append(lines[index])
                    index += 1
                if index < len(lines):
                    result.append(lines[index])
                    index += 1
                continue

            if _YAML_START_RE.match(line.strip()):
                code_lines = [line]
                in_block_scalar = bool(re.match(r".*:\s*[|>]\s*$", line.strip()))
                block_scalar_indent: int | None = None
                cursor = index + 1
                while cursor < len(lines):
                    next_line = lines[cursor]
                    next_stripped = next_line.strip()
                    if in_block_scalar:
                        if not next_stripped:
                            code_lines.append(next_line)
                            cursor += 1
                            continue
                        curr_indent = len(next_line) - len(next_line.lstrip())
                        if block_scalar_indent is None:
                            if curr_indent > 0 or next_stripped.startswith("-----"):
                                block_scalar_indent = max(curr_indent, 1)
                                code_lines.append(next_line)
                                cursor += 1
                                continue
                            in_block_scalar = False
                            block_scalar_indent = None
                        elif curr_indent >= block_scalar_indent or next_stripped.startswith("-----"):
                            code_lines.append(next_line)
                            cursor += 1
                            continue
                        else:
                            in_block_scalar = False
                            block_scalar_indent = None

                    is_yaml = (
                        not next_stripped
                        or next_line.startswith(" ")
                        or next_line.startswith("\t")
                        or _YAML_START_RE.match(next_stripped)
                        or next_stripped.startswith("- ")
                        or next_stripped.startswith("# ")
                        or re.match(r"^[\w./-]+\s*:", next_stripped)
                    )
                    if not is_yaml:
                        break
                    code_lines.append(next_line)
                    if re.match(r".*:\s*[|>]\s*$", next_stripped):
                        in_block_scalar = True
                        block_scalar_indent = None
                    cursor += 1

                while code_lines and not code_lines[-1].strip():
                    code_lines.pop()
                if len(code_lines) >= 2:
                    result.append("```yaml")
                    result.extend(code_lines)
                    result.append("```")
                    index = cursor
                    continue

            result.append(line)
            index += 1

        return "\n".join(result)


# ---------------------------------------------------------------------------
# PdfMergeSupport — cross-page merging (tables and YAML blocks)
# ---------------------------------------------------------------------------

@staticmethod
def _is_yaml_signal_line(raw_line: str) -> bool:
    stripped = raw_line.strip()
    return bool(
        _YAML_START_RE.match(stripped)
        or re.match(r"^[\w./-]+\s*:", stripped)
        or stripped.startswith("- ")
        or raw_line.startswith(" ")
        or raw_line.startswith("\t")
    )


def _is_yaml_filename(stripped: str) -> bool:
    return stripped.lower().endswith((".yaml", ".yml")) or (
        stripped.startswith("#") and any(ext in stripped.lower() for ext in (".yaml", ".yml"))
    )


class PdfMergeSupport:
    def _apply_merged_texts(
        self,
        documents: list[Document],
        markdown_sections: list[dict],
        index_a: int,
        merged_a: str,
        index_b: int,
        merged_b: str,
    ) -> None:
        page_a = markdown_sections[index_a]["page_number"]
        page_b = markdown_sections[index_b]["page_number"]
        for document in documents:
            if document.page_number == page_a:
                document.text = normalize_text(merged_a)
            elif document.page_number == page_b:
                document.text = normalize_text(merged_b)

    def _merge_cross_page_tables(self, documents: list[Document], markdown_sections: list[dict]) -> None:
        if len(markdown_sections) < 2:
            return

        for index in range(len(markdown_sections) - 1):
            curr_text = str(markdown_sections[index]["text"])
            next_text = str(markdown_sections[index + 1]["text"])
            curr_table_end = self._find_trailing_table(curr_text)
            if curr_table_end is None:
                continue
            next_table_start = self._find_leading_table(next_text)
            if next_table_start is None:
                continue

            curr_table_lines = curr_table_end["lines"]
            next_table_lines = next_table_start["lines"]
            curr_col_count = curr_table_lines[0].count("|") - 1
            next_col_count = next_table_lines[0].count("|") - 1
            if curr_col_count != next_col_count:
                continue

            body_rows = self._extract_table_body_rows(next_table_lines)
            if not body_rows:
                continue

            merged_curr_text = curr_text[:curr_table_end["end_pos"]].rstrip() + "\n" + "\n".join(body_rows) + curr_text[curr_table_end["end_pos"]:]
            merged_next_text = next_text[next_table_start["end_pos"]:].lstrip("\n")
            markdown_sections[index]["text"] = merged_curr_text.strip()
            markdown_sections[index]["chars"] = len(merged_curr_text.strip())
            markdown_sections[index + 1]["text"] = merged_next_text.strip()
            markdown_sections[index + 1]["chars"] = len(merged_next_text.strip())
            self._apply_merged_texts(documents, markdown_sections, index, merged_curr_text, index + 1, merged_next_text)

    def _merge_cross_page_yaml_blocks(self, documents: list[Document], markdown_sections: list[dict]) -> None:
        if len(markdown_sections) < 2:
            return

        max_passes = len(markdown_sections)
        for _ in range(max_passes):
            merged_any = False
            for index in range(len(markdown_sections) - 1):
                curr_text = str(markdown_sections[index]["text"])
                if not curr_text.strip():
                    continue

                next_index = index + 1
                while next_index < len(markdown_sections) and not str(markdown_sections[next_index]["text"]).strip():
                    next_index += 1
                if next_index >= len(markdown_sections):
                    continue

                next_text = str(markdown_sections[next_index]["text"])
                curr_yaml = self._find_trailing_yaml_block(curr_text)
                next_yaml = self._find_leading_yaml_block(next_text)
                if next_yaml is None:
                    if curr_yaml is not None:
                        next_yaml = self._find_leading_yaml_preamble(next_text)
                    if next_yaml is None:
                        continue
                if curr_yaml is None:
                    curr_yaml = self._find_trailing_yaml_preamble(curr_text)
                    if curr_yaml is None:
                        continue

                merged_body_lines = curr_yaml["body_lines"] + next_yaml["body_lines"]
                if len([line for line in merged_body_lines if line.strip()]) < 3:
                    continue

                merged_block = "```yaml\n" + "\n".join(merged_body_lines).rstrip() + "\n```"
                merged_curr_text = curr_text[:curr_yaml["start_pos"]].rstrip() + ("\n\n" if curr_text[:curr_yaml["start_pos"]].strip() else "") + merged_block
                merged_next_text = next_text[next_yaml["end_pos"]:].lstrip("\n")

                markdown_sections[index]["text"] = merged_curr_text.strip()
                markdown_sections[index]["chars"] = len(merged_curr_text.strip())
                markdown_sections[next_index]["text"] = merged_next_text.strip()
                markdown_sections[next_index]["chars"] = len(merged_next_text.strip())
                self._apply_merged_texts(documents, markdown_sections, index, merged_curr_text, next_index, merged_next_text)
                merged_any = True

            if not merged_any:
                break

    def _find_trailing_yaml_block(self, text: str) -> dict | None:
        stripped_text = text.rstrip()
        match = re.search(r"```yaml\s*\n(?P<body>[\s\S]*?)\n```$", stripped_text, flags=re.IGNORECASE)
        if not match:
            return None
        body_lines = [line.rstrip() for line in match.group("body").splitlines()]
        if not self._looks_like_yaml_lines(body_lines):
            return None
        return {"start_pos": match.start(), "end_pos": match.end(), "body_lines": body_lines}

    def _find_leading_yaml_block(self, text: str) -> dict | None:
        leading_whitespace = len(text) - len(text.lstrip())
        stripped_text = text.lstrip()
        match = re.match(r"```yaml\s*\n(?P<body>[\s\S]*?)\n```", stripped_text, flags=re.IGNORECASE)
        if not match:
            return None
        body_lines = [line.rstrip() for line in match.group("body").splitlines()]
        if not self._looks_like_yaml_lines(body_lines):
            return None
        return {
            "start_pos": leading_whitespace + match.start(),
            "end_pos": leading_whitespace + match.end(),
            "body_lines": body_lines,
        }

    def _find_trailing_yaml_preamble(self, text: str) -> dict | None:
        lines = text.rstrip().splitlines()
        if not lines:
            return None

        collected: list[str] = []
        start_index: int | None = None
        saw_yaml_signal = False
        for index in range(len(lines) - 1, -1, -1):
            raw_line = lines[index].rstrip()
            stripped = raw_line.strip()
            if not stripped:
                if collected:
                    break
                continue

            is_signal = _is_yaml_signal_line(raw_line)
            is_fname = _is_yaml_filename(stripped)
            if not collected and not (is_signal or is_fname):
                continue
            if collected and not (is_signal or is_fname):
                break
            if is_signal:
                saw_yaml_signal = True
            collected.insert(0, raw_line)
            start_index = index

        if start_index is None or not saw_yaml_signal:
            return None

        body_lines = [line.rstrip() for line in collected if line.strip()]
        if not body_lines:
            return None

        preamble_text = "\n".join(lines[start_index:]).rstrip()
        start_pos = text.rfind(preamble_text)
        if start_pos < 0:
            return None
        return {"start_pos": start_pos, "end_pos": start_pos + len(preamble_text), "body_lines": body_lines}

    def _find_leading_yaml_preamble(self, text: str) -> dict | None:
        lines = text.lstrip("\n").splitlines()
        if not lines:
            return None

        collected: list[str] = []
        saw_yaml_signal = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                if collected:
                    break
                continue
            if stripped.startswith("```"):
                break
            if not _is_yaml_signal_line(raw_line):
                break
            saw_yaml_signal = True
            collected.append(raw_line.rstrip())

        if not saw_yaml_signal or not collected:
            return None

        body_lines = [line for line in collected if line.strip()]
        if not body_lines:
            return None

        leading_ws = len(text) - len(text.lstrip("\n"))
        preamble_text = "\n".join(collected)
        start_pos = text.find(preamble_text, leading_ws)
        if start_pos < 0:
            start_pos = leading_ws
        end_pos = start_pos + len(preamble_text)
        return {"start_pos": 0, "end_pos": end_pos, "body_lines": body_lines}

    def _looks_like_yaml_lines(self, lines: list[str]) -> bool:
        meaningful = [line for line in lines if line.strip()]
        if len(meaningful) < 2:
            return False
        yaml_like = sum(
            1 for line in meaningful
            if _is_yaml_signal_line(line)
        )
        return yaml_like >= max(2, len(meaningful) // 2)

    def _find_trailing_table(self, text: str) -> dict | None:
        lines = text.rstrip().splitlines()
        if not lines:
            return None

        table_lines = []
        for index in range(len(lines) - 1, -1, -1):
            stripped = lines[index].strip()
            if stripped and "|" in stripped:
                table_lines.insert(0, stripped)
            elif stripped:
                break

        if len(table_lines) < 2:
            return None

        if not any(_TABLE_SEPARATOR_RE.match(line) for line in table_lines):
            return None
        start_pos = text.rfind(table_lines[0])
        end_pos = len(text.rstrip())
        return {"lines": table_lines, "start_pos": start_pos, "end_pos": end_pos}

    def _find_leading_table(self, text: str) -> dict | None:
        lines = text.lstrip().splitlines()
        if not lines:
            return None

        table_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and "|" in stripped:
                table_lines.append(stripped)
            elif stripped:
                break
        if len(table_lines) < 1:
            return None

        end_pos = 0
        remaining = text.lstrip()
        for table_line in table_lines:
            found_at = remaining.find(table_line, end_pos)
            if found_at >= 0:
                end_pos = found_at + len(table_line)
        leading_whitespace = len(text) - len(text.lstrip())
        end_pos += leading_whitespace
        return {"lines": table_lines, "end_pos": end_pos}

    def _extract_table_body_rows(self, table_lines: list[str]) -> list[str]:
        body_rows = []
        found_separator = False
        for line in table_lines:
            if _TABLE_SEPARATOR_RE.match(line):
                found_separator = True
                continue
            if found_separator:
                body_rows.append(line)

        if not found_separator:
            for line in table_lines:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if any(cell for cell in cells):
                    body_rows.append(line)
        return body_rows

    def _export_pdf_markdown(self, path: Path, documents: list[Document], markdown_sections: list[dict[str, str | int]]) -> None:
        if not self.settings.save_extracted_markdown:
            return

        output_path = extracted_markdown_path(self.settings.rag_extract_dir, path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {path.name}",
            "",
            f"- source_path: `{path}`",
            f"- extracted_pages: {len(documents)}",
            "",
        ]

        if not documents:
            lines.extend(["> No text was extracted from this PDF.", ""])
        else:
            for section in markdown_sections:
                lines.extend(
                    [
                        f"## Page {section['page_number']}",
                        "",
                        f"- loader: `{section['loader']}`",
                        f"- chars: {section['chars']}",
                        "",
                        str(section["text"]).strip(),
                        "",
                        "---",
                        "",
                    ]
                )

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

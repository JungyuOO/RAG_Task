from __future__ import annotations

import re
from pathlib import Path

from app.rag.utils import extracted_markdown_path
from app.rag.ingestion_pdf_extract import _YAML_START_RE
from app.rag.types import Document
from app.rag.utils import normalize_text


class PdfMergeSupport:
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

            for document in documents:
                if document.page_number == markdown_sections[index]["page_number"]:
                    document.text = normalize_text(merged_curr_text)
                elif document.page_number == markdown_sections[index + 1]["page_number"]:
                    document.text = normalize_text(merged_next_text)

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

                for document in documents:
                    if document.page_number == markdown_sections[index]["page_number"]:
                        document.text = normalize_text(merged_curr_text)
                    elif document.page_number == markdown_sections[next_index]["page_number"]:
                        document.text = normalize_text(merged_next_text)
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

            is_yaml_signal = bool(
                _YAML_START_RE.match(stripped)
                or re.match(r"^[\w./-]+\s*:", stripped)
                or stripped.startswith("- ")
                or raw_line.startswith(" ")
                or raw_line.startswith("\t")
            )
            is_yaml_filename = stripped.lower().endswith((".yaml", ".yml")) or (
                stripped.startswith("#") and any(ext in stripped.lower() for ext in (".yaml", ".yml"))
            )

            if not collected and not (is_yaml_signal or is_yaml_filename):
                continue
            if collected and not (is_yaml_signal or is_yaml_filename):
                break
            if is_yaml_signal:
                saw_yaml_signal = True
            collected.insert(0, raw_line)
            start_index = index

        if start_index is None or not saw_yaml_signal:
            return None

        body_lines = [line.rstrip() for line in collected if line.strip()]
        if len(body_lines) < 1:
            return None

        preamble_text = "\n".join(lines[start_index:]).rstrip()
        start_pos = text.rfind(preamble_text)
        if start_pos < 0:
            return None
        return {"start_pos": start_pos, "end_pos": start_pos + len(preamble_text), "body_lines": body_lines}

    def _looks_like_yaml_lines(self, lines: list[str]) -> bool:
        meaningful = [line for line in lines if line.strip()]
        if len(meaningful) < 2:
            return False
        yaml_like = 0
        for line in meaningful:
            stripped = line.strip()
            if _YAML_START_RE.match(stripped) or stripped.startswith("- ") or re.match(r"^[\w./-]+\s*:", stripped) or line.startswith(" ") or line.startswith("\t"):
                yaml_like += 1
        return yaml_like >= max(2, len(meaningful) // 2)

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

            is_yaml_signal = bool(
                _YAML_START_RE.match(stripped)
                or re.match(r"^[\w./-]+\s*:", stripped)
                or stripped.startswith("- ")
                or raw_line.startswith(" ")
                or raw_line.startswith("\t")
            )
            if stripped.startswith("```"):
                break
            if not is_yaml_signal:
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

        separator_re = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
        if not any(separator_re.match(line) for line in table_lines):
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
        separator_re = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
        body_rows = []
        found_separator = False
        for line in table_lines:
            if separator_re.match(line):
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

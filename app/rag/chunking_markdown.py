from __future__ import annotations

import re
from dataclasses import replace

from app.rag.chunking import MarkdownBlock, _normalize_markdown_text, _split_long_text
from app.rag.chunking_markdown_support import StructuredMarkdownChunkerSupport
from app.rag.types import Chunk, Document
from app.rag.utils import stable_hash


class StructuredMarkdownChunker(StructuredMarkdownChunkerSupport):
    """마크다운 구조를 인식해 공식 문서형 콘텐츠를 의미 단위로 청킹한다."""

    def __init__(self, *, chunk_size: int, overlap: int, max_block_chars: int = 1400) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_block_chars = max_block_chars

    def split(self, documents: list[Document], markdown_text: str | None = None) -> list[Chunk]:
        if not documents:
            return []

        blocks = self._blocks_from_markdown(markdown_text) if markdown_text else []
        if not blocks:
            blocks = self._blocks_from_documents(documents)
        if not blocks:
            return []

        chunks: list[Chunk] = []
        doc_id = stable_hash(documents[0].source_path)
        order = 0
        current_blocks: list[MarkdownBlock] = []
        current_length = 0
        heading_stack: list[tuple[int, str]] = []

        for block in blocks:
            if block.kind == "heading":
                level = block.heading_level or 1
                heading_stack = [item for item in heading_stack if item[0] < level]
                heading_stack.append((level, block.text))
                block = replace(block, heading_path=tuple(text for _, text in heading_stack))
            elif heading_stack:
                block = replace(block, heading_path=tuple(text for _, text in heading_stack))

            if len(block.text) > self.max_block_chars:
                if current_blocks:
                    chunks.append(self._build_chunk(doc_id, documents[0].source_path, current_blocks, order))
                    order += 1
                    current_blocks = []
                    current_length = 0
                for piece in _split_long_text(block.text, self.chunk_size, self.overlap):
                    piece_block = MarkdownBlock(
                        text=piece,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        kind=block.kind,
                        heading_level=block.heading_level,
                        heading_path=block.heading_path,
                    )
                    chunks.append(self._build_chunk(doc_id, documents[0].source_path, [piece_block], order))
                    order += 1
                continue

            if current_blocks and self._should_force_boundary(current_blocks, block):
                is_lone_heading = len(current_blocks) == 1 and current_blocks[0].kind == "heading"
                if not is_lone_heading:
                    chunks.append(self._build_chunk(doc_id, documents[0].source_path, current_blocks, order))
                    order += 1
                    if block.kind != "heading":
                        last_heading = next(
                            (b for b in reversed(current_blocks) if b.kind == "heading"),
                            None,
                        )
                        current_blocks = [last_heading] if last_heading else []
                        current_length = len(last_heading.text) if last_heading else 0
                    else:
                        current_blocks = []
                        current_length = 0

            projected = current_length + len(block.text) + (2 if current_blocks else 0)
            if current_blocks and projected > self.chunk_size:
                chunks.append(self._build_chunk(doc_id, documents[0].source_path, current_blocks, order))
                order += 1
                overlap_blocks = current_blocks[-1:] if current_blocks else []
                current_blocks = overlap_blocks.copy()
                current_length = sum(len(item.text) for item in current_blocks)

            current_blocks.append(block)
            current_length += len(block.text) + (2 if len(current_blocks) > 1 else 0)

        if current_blocks:
            chunks.append(self._build_chunk(doc_id, documents[0].source_path, current_blocks, order))

        return chunks

    def _should_force_boundary(self, current_blocks: list[MarkdownBlock], next_block: MarkdownBlock) -> bool:
        if not current_blocks:
            return False

        if next_block.kind == "heading":
            return True

        if next_block.kind == "table" and current_blocks[-1].kind == "paragraph":
            if len(current_blocks[-1].text) <= 150:
                return False

        if next_block.kind == current_blocks[-1].kind:
            return False

        # 코드 블록은 따로 유지
        if next_block.kind == "code":
            return True

        # heading/code만 강하게 경계로 두고
        # list/table/paragraph 전환은 chunk_size 초과 시 자연스럽게 끊기게 둠
        return False

    def _blocks_from_markdown(self, markdown_text: str) -> list[MarkdownBlock]:
        text = _normalize_markdown_text(markdown_text)
        if not text:
            return []

        # "## Page N" 헤더가 없는 경우 (예: 고객사 메뉴얼 .md 파일) 일반 마크다운으로 파싱
        if not re.search(r"(?m)^## Page \d+", text):
            return self._parse_markdown_blocks(text, page_number=1)

        sections = re.split(r"(?m)^## Page (\d+)\n", text)
        if len(sections) <= 1:
            return []

        page_entries: list[list[tuple[str, int]]] = []
        for index in range(1, len(sections), 2):
            page_number = int(sections[index])
            body = sections[index + 1]
            if body.startswith("\n"):
                body = body[1:]
            body = body.split("\n---", 1)[0]
            page_lines: list[tuple[str, int]] = []
            for line in body.splitlines():
                stripped = line.rstrip()
                if stripped.startswith("- loader:") or stripped.startswith("- chars:"):
                    continue
                page_lines.append((stripped, page_number))
            page_entries.append(page_lines)

        self._apply_page_boundary_policies(page_entries)

        annotated_lines: list[tuple[str, int]] = []
        for page_lines in page_entries:
            annotated_lines.extend(page_lines)

        if not annotated_lines:
            return []
        return self._parse_annotated_markdown_blocks(annotated_lines)

    def _apply_page_boundary_policies(self, page_entries: list[list[tuple[str, int]]]) -> None:
        policies = self._page_boundary_policies()
        for index in range(len(page_entries) - 1):
            while True:
                applied = False
                next_index = self._find_next_nonempty_page(page_entries, index + 1)
                if next_index is None:
                    break
                for policy in policies:
                    trailing = policy["find_trailing"](page_entries[index])
                    if trailing is None:
                        continue
                    continuation = policy["extract_leading"](page_entries[next_index])
                    if continuation is None:
                        continue
                    policy["apply"](page_entries, index, next_index, trailing, continuation)
                    applied = True
                    break
                if not applied:
                    break

    def _page_boundary_policies(self) -> list[dict]:
        return [
            {
                "name": "code",
                "find_trailing": self._find_trailing_fenced_code_range,
                "extract_leading": self._extract_leading_code_continuation,
                "apply": self._apply_code_continuation,
            },
            {
                "name": "list",
                "find_trailing": self._find_trailing_list_range,
                "extract_leading": self._extract_leading_list_continuation,
                "apply": self._apply_boundary_join,
            },
            {
                "name": "paragraph",
                "find_trailing": self._find_trailing_paragraph_range,
                "extract_leading": self._extract_leading_paragraph_continuation,
                "apply": self._apply_boundary_join,
            },
        ]

    @staticmethod
    def _apply_code_continuation(
        page_entries: list[list[tuple[str, int]]],
        index: int,
        next_index: int,
        trailing: tuple[int, int, str],
        continuation: dict,
    ) -> None:
        _start_idx, end_idx, _language = trailing
        page_entries[index] = page_entries[index][:end_idx] + continuation["lines"] + page_entries[index][end_idx:]
        page_entries[next_index] = page_entries[next_index][continuation["consumed_count"]:]

    @staticmethod
    def _apply_boundary_join(
        page_entries: list[list[tuple[str, int]]],
        index: int,
        next_index: int,
        _trailing: tuple[int, int],
        continuation: dict,
    ) -> None:
        page_entries[index].extend(continuation["lines"])
        page_entries[next_index] = page_entries[next_index][continuation["consumed_count"]:]

    @staticmethod
    def _find_next_nonempty_page(page_entries: list[list[tuple[str, int]]], start_index: int) -> int | None:
        for index in range(start_index, len(page_entries)):
            if any(line.strip() for line, _page in page_entries[index]):
                return index
        return None

    def _find_trailing_fenced_code_range(self, lines: list[tuple[str, int]]) -> tuple[int, int, str] | None:
        if not lines:
            return None

        end_idx = None
        for index in range(len(lines) - 1, -1, -1):
            if lines[index][0].strip():
                end_idx = index
                break
        if end_idx is None:
            return None

        closing_line = lines[end_idx][0].strip()
        if not closing_line.startswith("```"):
            return None

        for start_idx in range(end_idx - 1, -1, -1):
            opening_line = lines[start_idx][0].strip()
            if not opening_line.startswith("```"):
                continue
            language = opening_line[3:].strip().lower()
            if language not in {"yaml", "yml", ""}:
                return None
            block_text = "\n".join(line for line, _page in lines[start_idx:end_idx + 1])
            lowered = block_text.casefold()
            if "apiversion:" not in lowered and "kind:" not in lowered:
                return None
            return start_idx, end_idx, language
        return None

    def _extract_leading_code_continuation(self, lines: list[tuple[str, int]]) -> dict | None:
        if not lines:
            return None

        consumed = 0
        while consumed < len(lines):
            stripped = lines[consumed][0].strip()
            if not stripped:
                consumed += 1
                continue
            if self._looks_like_yaml_continuation_line(lines[consumed][0]):
                break
            if self._looks_like_bridge_line(stripped) and self._next_meaningful_line_is_yaml_continuation(lines, consumed + 1):
                consumed += 1
                continue
            break

        if consumed >= len(lines):
            return None

        continuation_lines: list[tuple[str, int]] = []
        index = consumed
        while index < len(lines):
            raw_line, page_number = lines[index]
            stripped = raw_line.strip()
            if not stripped:
                if continuation_lines:
                    break
                index += 1
                consumed = index
                continue
            if not self._looks_like_yaml_continuation_line(raw_line):
                break
            continuation_lines.append((raw_line.rstrip(), page_number))
            index += 1

        if not continuation_lines:
            return None
        return {"lines": continuation_lines, "consumed_count": index}

    def _next_meaningful_line_is_yaml_continuation(self, lines: list[tuple[str, int]], start_index: int) -> bool:
        for raw_line, _page in lines[start_index:]:
            stripped = raw_line.strip()
            if not stripped:
                continue
            return self._looks_like_yaml_continuation_line(raw_line)
        return False

    def _looks_like_bridge_line(self, stripped: str) -> bool:
        if not stripped or len(stripped) > 40:
            return False
        if stripped.startswith(("```", "#")):
            return False
        if self._is_list_line(stripped):
            return False
        if self._looks_like_yaml_continuation_line(stripped):
            return False
        return ":" not in stripped

    def _find_trailing_list_range(self, lines: list[tuple[str, int]]) -> tuple[int, int] | None:
        meaningful = [(idx, line.strip()) for idx, (line, _page) in enumerate(lines) if line.strip()]
        if not meaningful:
            return None

        end_idx = meaningful[-1][0]
        start_idx = end_idx
        while start_idx >= 0 and self._is_list_line(lines[start_idx][0].strip()):
            start_idx -= 1
        start_idx += 1
        if start_idx > end_idx:
            return None
        if start_idx == end_idx and not self._is_list_line(lines[end_idx][0].strip()):
            return None
        if not all(self._is_list_line(lines[idx][0].strip()) for idx in range(start_idx, end_idx + 1) if lines[idx][0].strip()):
            return None
        return start_idx, end_idx

    def _extract_leading_list_continuation(self, lines: list[tuple[str, int]]) -> dict | None:
        consumed = 0
        while consumed < len(lines) and not lines[consumed][0].strip():
            consumed += 1
        continuation_lines: list[tuple[str, int]] = []
        index = consumed
        while index < len(lines):
            stripped = lines[index][0].strip()
            if not stripped or not self._is_list_line(stripped):
                break
            continuation_lines.append((lines[index][0].rstrip(), lines[index][1]))
            index += 1
        if not continuation_lines:
            return None
        return {"lines": continuation_lines, "consumed_count": index}

    def _find_trailing_paragraph_range(self, lines: list[tuple[str, int]]) -> tuple[int, int] | None:
        meaningful = [(idx, line.strip()) for idx, (line, _page) in enumerate(lines) if line.strip()]
        if not meaningful:
            return None
        end_idx, last_line = meaningful[-1]
        if self._is_boundary_line(last_line):
            return None
        return end_idx, end_idx

    def _extract_leading_paragraph_continuation(self, lines: list[tuple[str, int]]) -> dict | None:
        consumed = 0
        while consumed < len(lines):
            stripped = lines[consumed][0].strip()
            if not stripped:
                consumed += 1
                continue
            break
        if consumed >= len(lines):
            return None

        raw_line, page_number = lines[consumed]
        stripped = raw_line.strip()
        if self._is_boundary_line(stripped):
            return None
        return {"lines": [(raw_line.rstrip(), page_number)], "consumed_count": consumed + 1}

    def _is_boundary_line(self, stripped: str) -> bool:
        if not stripped:
            return True
        if stripped.startswith(("```", "#")):
            return True
        if self._is_list_line(stripped):
            return True
        if self._looks_like_yaml_continuation_line(stripped):
            return True
        if self._is_table_block([stripped]):
            return True
        return False

    @staticmethod
    def _looks_like_yaml_continuation_line(line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.casefold()
        if not stripped:
            return False
        if stripped.startswith("#"):
            return True
        if line.startswith(("  ", "\t")):
            return True
        if lowered.startswith((
            "-", "name:", "image:", "path:", "storage:", "accessmodes:", "resources:", "requests:",
            "claimname:", "mountpath:", "containers:", "volumes:", "selector:", "matchlabels:",
            "replicas:", "provisioner:", "parameters:", "type:", "allowvolumeexpansion:", "mountoptions:",
            "persistentvolumeclaim:", "volumemounts:", "volumemode:", "persistentvolumereclaimpolicy:",
            "host:", "to:", "port:", "targetport:", "tls:", "weight:", "reclaimpolicy:",
            "http:", "paths:", "backend:", "annotations:", "pathtype:", "number:", "rules:", "service:",
        )):
            return True
        return bool(re.match(r"^[A-Za-z0-9_.\"'/-]+\s*:\s*", stripped))

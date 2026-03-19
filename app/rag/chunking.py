from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.types import Chunk, Document
from app.rag.utils import normalize_text, stable_hash, tokenize


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return pieces


def _normalize_markdown_text(text: str) -> str:
    collapsed = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    collapsed = re.sub(r"[ \t]+\n", "\n", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


@dataclass(slots=True)
class MarkdownBlock:
    text: str
    page_start: int
    page_end: int
    kind: str


class TextChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document], markdown_text: str | None = None) -> list[Chunk]:
        del markdown_text
        chunks: list[Chunk] = []
        for document in documents:
            text = normalize_text(document.text)
            if not text:
                continue

            start = 0
            order = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk_id = stable_hash(f"{document.doc_id}:{order}:{chunk_text[:40]}")
                    page_number = document.page_number
                    metadata = {
                        **document.metadata,
                        "offset_start": start,
                        "offset_end": end,
                        "page_start": page_number,
                        "page_end": page_number,
                        "chunking_strategy": "page_window",
                    }
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=document.doc_id,
                            source_path=document.source_path,
                            text=chunk_text,
                            tokens=tokenize(chunk_text),
                            page_number=page_number,
                            metadata=metadata,
                        )
                    )

                if end >= len(text):
                    break
                start = max(0, end - self.overlap)
                order += 1
        return chunks


class StructuredMarkdownChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150, max_block_chars: int = 1400) -> None:
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

        for block in blocks:
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
                    )
                    chunks.append(self._build_chunk(doc_id, documents[0].source_path, [piece_block], order))
                    order += 1
                continue

            if current_blocks and (
                block.page_start != current_blocks[-1].page_end
                or self._should_force_boundary(current_blocks, block)
            ):
                chunks.append(self._build_chunk(doc_id, documents[0].source_path, current_blocks, order))
                order += 1
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

        current_kinds = {block.kind for block in current_blocks}
        structured_kinds = {"heading", "list", "table", "code"}
        if next_block.kind == "heading":
            return True
        if next_block.kind in structured_kinds and current_kinds - {next_block.kind}:
            return True
        if next_block.kind == "paragraph" and current_kinds & structured_kinds:
            return True
        return False

    def _blocks_from_markdown(self, markdown_text: str) -> list[MarkdownBlock]:
        text = _normalize_markdown_text(markdown_text)
        if not text:
            return []

        sections = re.split(r"(?m)^## Page (\d+)\n", text)
        if len(sections) <= 1:
            return []

        blocks: list[MarkdownBlock] = []
        for index in range(1, len(sections), 2):
            page_number = int(sections[index])
            body = sections[index + 1]
            if body.startswith("\n"):
                body = body[1:]
            body = body.split("\n---", 1)[0]
            lines = [line.rstrip() for line in body.splitlines()]
            content_lines = [line for line in lines if not line.startswith("- loader:") and not line.startswith("- chars:")]
            body_text = "\n".join(content_lines).strip()
            if not body_text:
                continue
            blocks.extend(self._parse_markdown_blocks(body_text, page_number))
        return blocks

    def _parse_markdown_blocks(self, body_text: str, page_number: int) -> list[MarkdownBlock]:
        blocks: list[MarkdownBlock] = []
        sections = self._split_markdown_sections(body_text)
        for section in sections:
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if not lines:
                continue
            if len(lines) == 1 and (lines[0].startswith("#") or (len(lines[0]) <= 40 and lines[0].endswith(":"))):
                normalized = normalize_text(lines[0].lstrip("#").strip())
                if normalized:
                    blocks.append(
                        MarkdownBlock(
                            text=normalized,
                            page_start=page_number,
                            page_end=page_number,
                            kind="heading",
                        )
                )
                continue

            if self._is_code_block(lines):
                normalized = "\n".join(line.rstrip() for line in lines).strip()
                blocks.append(
                    MarkdownBlock(
                        text=normalized,
                        page_start=page_number,
                        page_end=page_number,
                        kind="code",
                    )
                )
                continue

            if self._is_table_block(lines):
                normalized = "\n".join(line.rstrip() for line in lines).strip()
                blocks.append(
                    MarkdownBlock(
                        text=normalized,
                        page_start=page_number,
                        page_end=page_number,
                        kind="table",
                    )
                )
                continue

            all_list = all(self._is_list_line(line) for line in lines)
            if all_list:
                normalized = "\n".join(lines).strip()
                blocks.append(
                    MarkdownBlock(
                        text=normalized,
                        page_start=page_number,
                        page_end=page_number,
                        kind="list",
                    )
                )
                continue

            kind = "paragraph"
            normalized = normalize_text(" ".join(lines))
            if normalized:
                blocks.append(
                    MarkdownBlock(
                        text=normalized,
                        page_start=page_number,
                        page_end=page_number,
                        kind=kind,
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

            current_lines.append(line)

        if current_lines:
            sections.append("\n".join(current_lines).strip())
        return [section for section in sections if section]

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
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return True
        return False

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

    def _build_chunk(
        self,
        doc_id: str,
        source_path: str,
        blocks: list[MarkdownBlock],
        order: int,
    ) -> Chunk:
        chunk_text = "\n\n".join(block.text for block in blocks).strip()
        page_start = blocks[0].page_start
        page_end = blocks[-1].page_end
        chunk_id = stable_hash(f"{doc_id}:{order}:{page_start}:{page_end}:{chunk_text[:40]}")
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_path=source_path,
            text=chunk_text,
            tokens=tokenize(chunk_text),
            page_number=page_start if page_start == page_end else None,
            metadata={
                "page_start": page_start,
                "page_end": page_end,
                "block_types": ",".join(sorted({block.kind for block in blocks})),
                "block_count": len(blocks),
                "chunking_strategy": "structured_markdown",
            },
        )

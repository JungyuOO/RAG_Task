from __future__ import annotations

import re
from pathlib import Path

from apps.api.core.text import stable_hash
from apps.api.schemas.indexing import (
    BlockAttributes,
    BlockType,
    ParsedBlock,
    ParsedDocumentBundle,
    SourceDescriptor,
    SourceType,
)
from apps.api.rag.indexing.parsers.base import BaseParser


class GeneratedManualParser(BaseParser):
    parser_name = "generated_manual"
    _PAGE_HEADING_PATTERN = re.compile(r"^page\s+\d+$", flags=re.IGNORECASE)
    _CLIPBOARD_NOISE_PATTERN = re.compile(r"(?:Copy linkLink copied to clipboard!|Link copied to clipboard!)$", flags=re.IGNORECASE)

    def supports(self, source: SourceDescriptor) -> bool:
        return source.source_type == SourceType.GENERATED_MANUAL

    def parse(self, source: SourceDescriptor) -> ParsedDocumentBundle:
        markdown = self.read_text_if_exists(source.source_path)
        title = self._extract_markdown_title(markdown) or source.file_name or Path(source.source_path).stem
        bundle = self.build_empty_bundle(
            source,
            title=title,
            metadata={
                "parser_status": "implemented",
                "expected_specialization": "customer/generated manuals",
            },
        )

        blocks: list[ParsedBlock] = []
        if not markdown.strip():
            bundle.blocks = blocks
            return bundle

        section_path: list[str] = []
        lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        paragraph_buffer: list[str] = []
        list_buffer: list[str] = []
        code_buffer: list[str] = []
        code_language = ""
        in_code = False
        block_index = 0

        def flush_paragraph() -> None:
            nonlocal block_index, paragraph_buffer
            text = " ".join(self._clean_line(part) for part in paragraph_buffer if self._clean_line(part)).strip()
            paragraph_buffer = []
            if not text:
                return
            block_index += 1
            blocks.append(
                ParsedBlock(
                    block_id=stable_hash(f"{bundle.document.doc_id}:paragraph:{block_index}:{text[:80]}"),
                    block_type=BlockType.PARAGRAPH,
                    text=text,
                    section_title=section_path[-1] if section_path else title,
                    section_path=list(section_path),
                    metadata={"parser_name": self.parser_name},
                )
            )

        def flush_list() -> None:
            nonlocal block_index, list_buffer
            if not list_buffer:
                return
            text = "\n".join(self._clean_line(item) for item in list_buffer if self._clean_line(item)).strip()
            list_buffer = []
            if not text:
                return
            block_index += 1
            blocks.append(
                ParsedBlock(
                    block_id=stable_hash(f"{bundle.document.doc_id}:list:{block_index}:{text[:80]}"),
                    block_type=BlockType.LIST,
                    text=text,
                    section_title=section_path[-1] if section_path else title,
                    section_path=list(section_path),
                    attributes=BlockAttributes(list_style="bullet"),
                    metadata={"parser_name": self.parser_name},
                )
            )

        def flush_code() -> None:
            nonlocal block_index, code_buffer, code_language
            if not code_buffer:
                code_language = ""
                return
            text = "\n".join(line.rstrip() for line in code_buffer if line.strip()).strip()
            code_buffer = []
            language = code_language
            code_language = ""
            if not text or self._should_skip_tiny_code_block(text):
                return
            block_index += 1
            blocks.append(
                ParsedBlock(
                    block_id=stable_hash(f"{bundle.document.doc_id}:code:{block_index}:{text[:80]}"),
                    block_type=BlockType.CODE,
                    text=text,
                    section_title=section_path[-1] if section_path else title,
                    section_path=list(section_path),
                    attributes=BlockAttributes(language=language),
                    metadata={"parser_name": self.parser_name},
                )
            )

        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                flush_paragraph()
                flush_list()
                if in_code:
                    flush_code()
                    in_code = False
                else:
                    in_code = True
                    code_language = stripped[3:].strip()
                continue

            if in_code:
                code_buffer.append(raw_line)
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                flush_paragraph()
                flush_list()
                level = len(heading_match.group(1))
                heading_text = self._clean_line(heading_match.group(2))
                if not heading_text or self._PAGE_HEADING_PATTERN.match(heading_text):
                    continue
                section_path = section_path[: max(level - 1, 0)]
                section_path.append(heading_text)
                block_index += 1
                blocks.append(
                    ParsedBlock(
                        block_id=stable_hash(f"{bundle.document.doc_id}:heading:{block_index}:{heading_text}"),
                        block_type=BlockType.HEADING,
                        text=heading_text,
                        section_title=heading_text,
                        section_path=list(section_path),
                        attributes=BlockAttributes(heading_level=level),
                        metadata={"parser_name": self.parser_name},
                    )
                )
                continue

            list_match = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", stripped)
            if list_match:
                flush_paragraph()
                cleaned = self._clean_line(list_match.group(1))
                if cleaned:
                    list_buffer.append(cleaned)
                continue

            if not stripped:
                flush_paragraph()
                flush_list()
                continue

            cleaned = self._clean_line(stripped)
            if cleaned:
                paragraph_buffer.append(cleaned)

        flush_paragraph()
        flush_list()
        if in_code:
            flush_code()

        bundle.blocks = blocks
        return bundle

    @staticmethod
    def _extract_markdown_title(markdown: str) -> str:
        if not markdown:
            return ""
        match = re.search(r"^\s*#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @classmethod
    def _clean_line(cls, text: str) -> str:
        cleaned = cls._CLIPBOARD_NOISE_PATTERN.sub("", str(text or "")).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _should_skip_tiny_code_block(text: str) -> bool:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return True
        tokens = normalized.split()
        if len(tokens) > 3:
            return False
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", normalized))



from __future__ import annotations

import re

from apps.api.schemas.indexing import BlockType, NormalizedBlock, ParsedDocumentBundle


class BlockNormalizer:
    """Converts parser output into normalized blocks for chunking/enrichment."""

    _PAGE_HEADING_PATTERN = re.compile(r"^page\s+\d+$", flags=re.IGNORECASE)
    _CLIPBOARD_NOISE_PATTERN = re.compile(r"(?:Copy linkLink copied to clipboard!|Link copied to clipboard!)$", flags=re.IGNORECASE)

    def normalize_bundle(self, bundle: ParsedDocumentBundle) -> list[NormalizedBlock]:
        normalized: list[NormalizedBlock] = []
        last_signature = ""
        for block in bundle.blocks:
            display_text = self._clean_display_text(str(block.text or ""))
            if not display_text:
                continue
            if block.block_type == BlockType.HEADING and self._PAGE_HEADING_PATTERN.match(display_text):
                continue
            normalized_text = self._normalize_text(display_text, block.block_type)
            if not normalized_text:
                continue
            if block.block_type != BlockType.CODE and len(normalized_text.split()) <= 1 and len(normalized_text) < 8:
                continue
            signature = f"{block.block_type.value}:{display_text.casefold()}:{' > '.join(block.section_path)}"
            if signature == last_signature:
                continue
            last_signature = signature
            normalized.append(
                NormalizedBlock(
                    block_id=block.block_id,
                    doc_id=bundle.document.doc_id,
                    block_type=block.block_type,
                    normalized_text=normalized_text,
                    display_text=display_text,
                    html_anchor=block.html_anchor,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    section_title=block.section_title,
                    section_path=[value.strip() for value in block.section_path if str(value).strip()],
                    resource_hints=list(block.resource_hints),
                    metadata={
                        **block.metadata,
                        "normalizer_version": "v1",
                    },
                )
            )
        return normalized

    @classmethod
    def _clean_display_text(cls, text: str) -> str:
        cleaned = cls._CLIPBOARD_NOISE_PATTERN.sub("", str(text or "").strip())
        return cleaned.strip()

    @staticmethod
    def _normalize_text(text: str, block_type: BlockType) -> str:
        if block_type == BlockType.CODE:
            lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
            normalized = "\n".join(line for line in lines if line.strip())
            return normalized.strip()
        return re.sub(r"\s+", " ", text).strip()



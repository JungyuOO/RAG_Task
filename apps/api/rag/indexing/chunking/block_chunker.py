from __future__ import annotations

from apps.api.core.text import stable_hash
from apps.api.schemas.indexing import BlockType, ChunkExpansionRef, ChunkRecord, NormalizedBlock, ParsedDocumentBundle


class BlockPreservingChunker:
    """Preserves section structure while packing blocks into larger semantic chunks."""

    ATTACHABLE_HEADING_TOKEN_LIMIT = 18
    TINY_CODE_TOKEN_LIMIT = 12
    MAX_MERGED_TOKENS = 440
    MAX_MERGED_BLOCKS = 10
    MAX_OVERLAP_TOKENS = 72
    MAX_OVERLAP_BLOCKS = 2
    PACKABLE_TYPES = {
        BlockType.PARAGRAPH,
        BlockType.LIST,
        BlockType.ADMONITION,
        BlockType.TABLE,
    }

    def chunk_document(
        self,
        bundle: ParsedDocumentBundle,
        normalized_blocks: list[NormalizedBlock],
    ) -> list[ChunkRecord]:
        groups = self._coalesce_blocks(normalized_blocks)
        chunks: list[ChunkRecord] = []
        chunk_ids = [
            stable_hash(
                f"{bundle.document.doc_id}:chunk:{index}:{':'.join(block.block_id for block in group)}"
            )
            for index, group in enumerate(groups)
        ]
        section_refs = [
            stable_hash(
                f"{bundle.document.doc_id}:section:{' > '.join((group[-1].section_path if group[-1].section_path else group[0].section_path))}:{group[-1].html_anchor or group[-1].section_title}"
            )
            for group in groups
        ]

        for index, group in enumerate(groups):
            primary_block = self._primary_block(group)
            section_block = group[-1]
            expansions: list[ChunkExpansionRef] = []
            if section_block.section_path or section_block.section_title:
                expansions.append(ChunkExpansionRef(relation="parent_section", target_id=section_refs[index]))
            if index > 0:
                expansions.append(ChunkExpansionRef(relation="previous_block", target_id=chunk_ids[index - 1]))
            if index + 1 < len(groups):
                expansions.append(ChunkExpansionRef(relation="next_block", target_id=chunk_ids[index + 1]))

            display_text = self._join_display_text(group)
            retrieval_text = self._join_retrieval_text(group)
            page_values = [block.page_start for block in group if block.page_start is not None]
            page_end_values = [block.page_end for block in group if block.page_end is not None]
            resource_hints = sorted({hint for block in group for hint in block.resource_hints if str(hint).strip()})

            metadata = {
                **primary_block.metadata,
                "chunker": "block_preserving_v4",
                "source_block_type": primary_block.block_type.value,
                "block_types": [block.block_type.value for block in group],
                "section_ref": section_refs[index],
                "merged_block_count": len(group),
                "chunk_merge_strategy": self._merge_strategy(group),
                "content_char_length": len(display_text),
                "overlap_prev_block_ids": self._overlap_prev_block_ids(groups, index),
                "overlap_prev_count": len(self._overlap_prev_block_ids(groups, index)),
            }

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_ids[index],
                    doc_id=bundle.document.doc_id,
                    source_path=bundle.document.source_path,
                    text=display_text,
                    retrieval_text=retrieval_text,
                    display_text=display_text,
                    chunk_order=index,
                    token_count=len(retrieval_text.split()),
                    page_start=min(page_values) if page_values else None,
                    page_end=max(page_end_values) if page_end_values else None,
                    section_title=section_block.section_title,
                    section_path=list(section_block.section_path),
                    html_anchor=section_block.html_anchor or primary_block.html_anchor,
                    block_ids=[block.block_id for block in group],
                    resource_hints=resource_hints,
                    expansions=expansions,
                    metadata=metadata,
                )
            )
        return chunks

    def _coalesce_blocks(self, blocks: list[NormalizedBlock]) -> list[list[NormalizedBlock]]:
        groups: list[list[NormalizedBlock]] = []
        index = 0

        while index < len(blocks):
            block = blocks[index]
            if block.block_type == BlockType.HEADING:
                group = [block]
                index += 1
                while index < len(blocks) and self._can_pack(group, blocks[index]):
                    group.append(blocks[index])
                    index += 1
                groups.append(group)
                continue

            if groups and self._can_pack(groups[-1], block):
                groups[-1].append(block)
                index += 1
                continue

            overlap_seed = self._build_overlap_seed(groups[-1], block) if groups else []
            groups.append([*overlap_seed, block] if overlap_seed else [block])
            index += 1

        return groups

    def _can_pack(self, group: list[NormalizedBlock], next_block: NormalizedBlock) -> bool:
        if len(group) >= self.MAX_MERGED_BLOCKS:
            return False
        if self._section_path(group[-1]) != self._section_path(next_block):
            return False
        if next_block.block_type == BlockType.HEADING:
            return False
        current_tokens = sum(len(block.normalized_text.split()) for block in group)
        next_tokens = len(next_block.normalized_text.split())
        if current_tokens + next_tokens > self.MAX_MERGED_TOKENS:
            return False

        if all(block.block_type == BlockType.HEADING for block in group):
            return next_block.block_type in self.PACKABLE_TYPES or self._is_tiny_code(next_block)

        primary = self._primary_block(group)
        if primary.block_type in self.PACKABLE_TYPES:
            return next_block.block_type in self.PACKABLE_TYPES or self._is_tiny_code(next_block)
        if primary.block_type == BlockType.CODE:
            return self._is_tiny_code(primary) and next_block.block_type in self.PACKABLE_TYPES
        return False

    def _is_tiny_code(self, block: NormalizedBlock) -> bool:
        return block.block_type == BlockType.CODE and len(block.normalized_text.split()) <= self.TINY_CODE_TOKEN_LIMIT

    def _build_overlap_seed(self, previous_group: list[NormalizedBlock], next_block: NormalizedBlock) -> list[NormalizedBlock]:
        if next_block.block_type == BlockType.HEADING:
            return []
        if self._section_path(previous_group[-1]) != self._section_path(next_block):
            return []

        overlap: list[NormalizedBlock] = []
        token_budget = 0
        for block in reversed(previous_group):
            if block.block_type == BlockType.HEADING:
                continue
            if block.block_type not in self.PACKABLE_TYPES and not self._is_tiny_code(block):
                continue
            block_tokens = len(block.normalized_text.split())
            if overlap and token_budget + block_tokens > self.MAX_OVERLAP_TOKENS:
                break
            if len(overlap) >= self.MAX_OVERLAP_BLOCKS:
                break
            overlap.append(block)
            token_budget += block_tokens

        overlap.reverse()
        candidate = [*overlap, next_block]
        if self._group_token_count(candidate) > self.MAX_MERGED_TOKENS:
            return []
        if len(candidate) > self.MAX_MERGED_BLOCKS:
            return []
        return overlap

    @staticmethod
    def _group_token_count(group: list[NormalizedBlock]) -> int:
        return sum(len(block.normalized_text.split()) for block in group)

    @staticmethod
    def _section_path(block: NormalizedBlock) -> tuple[str, ...]:
        return tuple(block.section_path)

    @staticmethod
    def _primary_block(group: list[NormalizedBlock]) -> NormalizedBlock:
        for block in group:
            if block.block_type != BlockType.HEADING:
                return block
        return group[0]

    @staticmethod
    def _join_display_text(group: list[NormalizedBlock]) -> str:
        parts = [block.display_text.strip() for block in group if block.display_text.strip()]
        return "\n\n".join(parts).strip()

    @staticmethod
    def _join_retrieval_text(group: list[NormalizedBlock]) -> str:
        if any(block.block_type == BlockType.CODE and len(block.normalized_text.split()) > 20 for block in group):
            return "\n\n".join(block.normalized_text.strip() for block in group if block.normalized_text.strip()).strip()
        return " ".join(block.normalized_text.strip() for block in group if block.normalized_text.strip()).strip()

    @staticmethod
    def _merge_strategy(group: list[NormalizedBlock]) -> str:
        if len(group) == 1:
            return "single_block"
        if group[0].block_type == BlockType.HEADING:
            return "section_packed"
        if any(block.block_type == BlockType.HEADING for block in group[1:]):
            return "mixed_with_heading"
        return "adjacent_packed_with_overlap" if len({block.block_id for block in group}) != len(group) else "adjacent_packed"

    @staticmethod
    def _overlap_prev_block_ids(groups: list[list[NormalizedBlock]], index: int) -> list[str]:
        if index <= 0:
            return []
        previous_ids = {block.block_id for block in groups[index - 1]}
        return [block.block_id for block in groups[index] if block.block_id in previous_ids]



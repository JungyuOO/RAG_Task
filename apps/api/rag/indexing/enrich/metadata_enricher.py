from __future__ import annotations

from apps.api.schemas.indexing import NormalizedBlock, ParsedDocumentBundle


class MetadataEnricher:
    """Applies document-level metadata to normalized blocks."""

    def enrich_blocks(
        self,
        bundle: ParsedDocumentBundle,
        normalized_blocks: list[NormalizedBlock],
    ) -> list[NormalizedBlock]:
        enriched: list[NormalizedBlock] = []
        for block in normalized_blocks:
            enriched.append(
                block.model_copy(
                    update={
                        "metadata": {
                            **block.metadata,
                            "document_title": bundle.document.title,
                            "normalized_document_title": bundle.document.title.casefold().strip(),
                            "source_type": bundle.document.source_type.value,
                            "document_group": bundle.document.document_group.value,
                            "locale": bundle.document.locale,
                            "section_depth": len(block.section_path),
                            "anchor_path": " > ".join(block.section_path),
                            "resource_hint_count": len(block.resource_hints),
                            "content_char_length": len(block.display_text),
                            "enricher_version": "v2",
                        }
                    }
                )
            )
        return enriched



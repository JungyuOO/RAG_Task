from __future__ import annotations

from pathlib import Path

from apps.api.schemas.indexing import ParsedDocumentBundle, SourceDescriptor, SourceType
from apps.api.rag.indexing.parsers.base import BaseParser


class StructuredJsonParser(BaseParser):
    parser_name = "structured_json"

    def supports(self, source: SourceDescriptor) -> bool:
        return source.source_type == SourceType.STRUCTURED_JSON

    def parse(self, source: SourceDescriptor) -> ParsedDocumentBundle:
        return self.build_empty_bundle(
            source,
            title=source.file_name or Path(source.source_path).stem,
            metadata={
                "parser_status": "skeleton",
                "expected_specialization": "structured key/value source",
            },
        )



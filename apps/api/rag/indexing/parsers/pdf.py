from __future__ import annotations

from pathlib import Path

from apps.api.api.schemas.indexing import ParsedDocumentBundle, SourceDescriptor, SourceType
from apps.api.rag.indexing.parsers.base import BaseParser


class PdfParser(BaseParser):
    parser_name = "pdf"

    def supports(self, source: SourceDescriptor) -> bool:
        return source.source_type == SourceType.PDF

    def parse(self, source: SourceDescriptor) -> ParsedDocumentBundle:
        return self.build_empty_bundle(
            source,
            title=source.file_name or Path(source.source_path).stem,
            trust_score=0.8,
            metadata={
                "parser_status": "skeleton",
                "expected_specialization": "pdf fallback extraction",
            },
        )


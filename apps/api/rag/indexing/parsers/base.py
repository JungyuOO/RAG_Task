from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from apps.api.core.text import stable_hash
from apps.api.schemas.indexing import (
    ParsedDocument,
    ParsedDocumentBundle,
    ParsedDocumentMetadata,
    SourceDescriptor,
)


class BaseParser(ABC):
    """Common parser contract for new ingestion modules.

    These parsers are intentionally not wired into the legacy runtime yet.
    The current phase creates a stable contract and minimal concrete classes
    so the ingestion rewrite can proceed incrementally.
    """

    parser_name = "base"

    @abstractmethod
    def supports(self, source: SourceDescriptor) -> bool:
        """Return True when the parser can handle the given source."""

    @abstractmethod
    def parse(self, source: SourceDescriptor) -> ParsedDocumentBundle:
        """Parse a source descriptor into the normalized parser bundle."""

    def build_document(
        self,
        source: SourceDescriptor,
        *,
        title: str = "",
        trust_score: float = 1.0,
        metadata: dict | None = None,
    ) -> ParsedDocument:
        combined_metadata = {
            **source.metadata,
            **(metadata or {}),
            "parser_name": self.parser_name,
        }
        resolved_title = title.strip() or source.file_name or Path(source.source_path).stem
        return ParsedDocument(
            doc_id=stable_hash(f"{source.source_type}:{source.source_path}"),
            source_path=source.source_path,
            title=resolved_title,
            source_type=source.source_type,
            locale=source.locale,
            document_group=source.document_group,
            metadata=ParsedDocumentMetadata(
                source_url=source.source_url,
                locale=source.locale,
                source_type=source.source_type,
                document_group=source.document_group,
                trust_score=trust_score,
                metadata=combined_metadata,
            ),
        )

    def build_empty_bundle(
        self,
        source: SourceDescriptor,
        *,
        title: str = "",
        trust_score: float = 1.0,
        metadata: dict | None = None,
    ) -> ParsedDocumentBundle:
        return ParsedDocumentBundle(
            document=self.build_document(
                source,
                title=title,
                trust_score=trust_score,
                metadata=metadata,
            ),
            blocks=[],
        )

    @staticmethod
    def read_text_if_exists(path: str, *, encoding: str = "utf-8") -> str:
        candidate = Path(path)
        if not candidate.exists() or not candidate.is_file():
            return ""
        try:
            return candidate.read_text(encoding=encoding, errors="ignore")
        except OSError:
            return ""




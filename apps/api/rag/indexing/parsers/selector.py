from __future__ import annotations

from collections.abc import Iterable

from apps.api.schemas.indexing import SourceDescriptor
from apps.api.rag.indexing.parsers.base import BaseParser
from apps.api.rag.indexing.parsers.generated_manual import GeneratedManualParser
from apps.api.rag.indexing.parsers.html_page import HtmlPageParser
from apps.api.rag.indexing.parsers.html_single import HtmlSingleParser
from apps.api.rag.indexing.parsers.pdf import PdfParser
from apps.api.rag.indexing.parsers.structured import StructuredJsonParser


class ParserSelector:
    """Selects the first parser that claims support for a source descriptor."""

    def __init__(self, parsers: Iterable[BaseParser]) -> None:
        self._parsers = list(parsers)

    def select(self, source: SourceDescriptor) -> BaseParser:
        for parser in self._parsers:
            if parser.supports(source):
                return parser
        raise LookupError(f"No parser registered for source_type={source.source_type!s}")

    def available_parser_names(self) -> list[str]:
        return [parser.parser_name for parser in self._parsers]


def build_default_parser_selector() -> ParserSelector:
    return ParserSelector(
        [
            HtmlSingleParser(),
            HtmlPageParser(),
            PdfParser(),
            GeneratedManualParser(),
            StructuredJsonParser(),
        ]
    )



"""Parser selection and parser implementations for indexing."""

from apps.api.rag.indexing.parsers.base import BaseParser
from apps.api.rag.indexing.parsers.generated_manual import GeneratedManualParser
from apps.api.rag.indexing.parsers.html_page import HtmlPageParser
from apps.api.rag.indexing.parsers.html_single import HtmlSingleParser
from apps.api.rag.indexing.parsers.pdf import PdfParser
from apps.api.rag.indexing.parsers.selector import ParserSelector, build_default_parser_selector
from apps.api.rag.indexing.parsers.structured import StructuredJsonParser

__all__ = [
    "BaseParser",
    "GeneratedManualParser",
    "HtmlPageParser",
    "HtmlSingleParser",
    "ParserSelector",
    "PdfParser",
    "StructuredJsonParser",
    "build_default_parser_selector",
]

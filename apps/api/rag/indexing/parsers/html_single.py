from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

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


@dataclass(slots=True)
class _HtmlSection:
    level: int
    title: str
    anchor: str
    path: list[str]
    blocks: list[tuple[str, str]] = field(default_factory=list)


class _HtmlSingleSectionParser(HTMLParser):
    IGNORE_TAGS = {"script", "style", "nav", "header", "footer"}
    TEXT_TAGS = {"p", "li", "pre", "code"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[_HtmlSection] = []
        self.page_title = ""
        self._ignored_depth = 0
        self._content_depth = 0
        self._capture_tag = ""
        self._capture_attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self._path_by_level: dict[int, str] = {}
        self._current_section: _HtmlSection | None = None
        self._inside_title = False
        self._title_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        if tag == "title":
            self._inside_title = True
            self._title_buffer = []
            return
        if tag in self.IGNORE_TAGS:
            self._ignored_depth += 1
            return
        if tag in {"main", "article"}:
            self._content_depth += 1
        if not self._in_content:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} | self.TEXT_TAGS:
            self._capture_tag = tag
            self._capture_attrs = attr_map
            self._buffer = []
        elif tag == "br" and self._capture_tag:
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._inside_title = False
            self.page_title = _clean_text("".join(self._title_buffer))
            self._title_buffer = []
            return
        if tag in self.IGNORE_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag in {"main", "article"}:
            self._content_depth = max(0, self._content_depth - 1)
        if not self._capture_tag or tag != self._capture_tag:
            return
        capture_tag = self._capture_tag
        attrs = self._capture_attrs
        text = _clean_text("".join(self._buffer))
        self._capture_tag = ""
        self._capture_attrs = {}
        self._buffer = []
        if not text:
            return
        if capture_tag.startswith("h"):
            self._start_section(level=int(capture_tag[1]), title=text, anchor=attrs.get("id", ""))
            return
        if self._current_section is None:
            # If content starts before the first heading, create an implicit root section.
            self._start_section(level=1, title=self.page_title or "Document", anchor="")
        assert self._current_section is not None
        self._current_section.blocks.append((capture_tag, text))

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self._title_buffer.append(data)
        if self._capture_tag and self._in_content:
            self._buffer.append(data)

    @property
    def _in_content(self) -> bool:
        return self._content_depth > 0 and self._ignored_depth == 0

    def _start_section(self, *, level: int, title: str, anchor: str) -> None:
        self._path_by_level[level] = title
        for key in list(self._path_by_level):
            if key > level:
                self._path_by_level.pop(key, None)
        path = [self._path_by_level[key] for key in sorted(self._path_by_level)]
        self._current_section = _HtmlSection(level=level, title=title, anchor=anchor, path=path)
        self.sections.append(self._current_section)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _detect_resource_hints(*values: str) -> list[str]:
    haystack = " ".join(value.casefold() for value in values if value)
    candidates = {
        "pod": (" pod", "pods", "pod "),
        "deployment": ("deployment", "deployments"),
        "service": ("service", "services"),
        "route": ("route", "routes"),
        "event": ("event", "events"),
        "namespace": ("namespace", "namespaces"),
        "pvc": ("persistentvolumeclaim", " pvc", "pvcs"),
    }
    hints: list[str] = []
    for hint, markers in candidates.items():
        if any(marker in haystack for marker in markers):
            hints.append(hint)
    return hints


class HtmlSingleParser(BaseParser):
    parser_name = "html_single"

    def supports(self, source: SourceDescriptor) -> bool:
        return source.source_type == SourceType.HTML_SINGLE

    def parse(self, source: SourceDescriptor) -> ParsedDocumentBundle:
        raw_html = self.read_text_if_exists(source.source_path)
        parser = _HtmlSingleSectionParser()
        parser.feed(raw_html)
        parser.close()

        title = parser.page_title or (parser.sections[0].title if parser.sections else "") or source.file_name
        bundle = self.build_empty_bundle(
            source,
            title=title,
            metadata={
                "parser_status": "implemented",
                "expected_specialization": "docs.redhat.com html-single",
                "section_count": len(parser.sections),
            },
        )

        blocks: list[ParsedBlock] = []
        for section_index, section in enumerate(parser.sections, start=1):
            section_path = list(section.path)
            heading_anchor = section.anchor or f"section-{section_index}"
            blocks.append(
                ParsedBlock(
                    block_id=stable_hash(f"{bundle.document.doc_id}:heading:{section_index}:{section.title}"),
                    block_type=BlockType.HEADING,
                    text=section.title,
                    html_anchor=heading_anchor,
                    section_title=section.title,
                    section_path=section_path,
                    resource_hints=_detect_resource_hints(section.title, " ".join(section_path)),
                    attributes=BlockAttributes(heading_level=section.level),
                    metadata={"parser_name": self.parser_name},
                )
            )
            for block_index, (tag, text) in enumerate(section.blocks, start=1):
                block_type = self._map_block_type(tag)
                hints = _detect_resource_hints(section.title, " ".join(section_path), text)
                blocks.append(
                    ParsedBlock(
                        block_id=stable_hash(
                            f"{bundle.document.doc_id}:{heading_anchor}:{block_index}:{block_type.value}:{text[:80]}"
                        ),
                        block_type=block_type,
                        text=text,
                        html_anchor=heading_anchor,
                        section_title=section.title,
                        section_path=section_path,
                        resource_hints=hints,
                        attributes=BlockAttributes(
                            language="text" if block_type == BlockType.CODE else "",
                            list_style="bullet" if block_type == BlockType.LIST else "",
                        ),
                        metadata={"parser_name": self.parser_name, "source_tag": tag},
                    )
                )

        bundle.blocks = blocks
        return bundle

    @staticmethod
    def _map_block_type(tag: str) -> BlockType:
        return {
            "p": BlockType.PARAGRAPH,
            "li": BlockType.LIST,
            "pre": BlockType.CODE,
            "code": BlockType.CODE,
        }.get(tag, BlockType.PARAGRAPH)




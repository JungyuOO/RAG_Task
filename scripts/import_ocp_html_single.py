"""Import a docs.redhat.com html-single source into markdown + extracted artifacts.

Usage:
    python scripts/import_ocp_html_single.py --input-html path/to/advanced_networking.html --input-meta path/to/advanced_networking.meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.rag.utils import extracted_html_path, extracted_metadata_path


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


@dataclass(slots=True)
class HtmlSection:
    level: int
    title: str
    anchor: str
    path: list[str]
    blocks: list[tuple[str, str]]


class HtmlSingleSectionParser(HTMLParser):
    IGNORE_TAGS = {"script", "style", "nav", "header", "footer"}
    TEXT_TAGS = {"p", "li", "pre", "code"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[HtmlSection] = []
        self._ignored_depth = 0
        self._content_depth = 0
        self._capture_tag = ""
        self._capture_attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self._path_by_level: dict[int, str] = {}
        self._current_section: HtmlSection | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
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
        if tag in self.IGNORE_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if tag in {"main", "article"}:
            self._content_depth = max(0, self._content_depth - 1)
        if not self._capture_tag or tag != self._capture_tag:
            return
        text = _clean_text("".join(self._buffer))
        attrs = self._capture_attrs
        capture_tag = self._capture_tag
        self._capture_tag = ""
        self._capture_attrs = {}
        self._buffer = []
        if not text:
            return
        if capture_tag.startswith("h"):
            self._start_section(int(capture_tag[1]), text, attrs.get("id", ""))
        elif self._current_section is not None:
            self._current_section.blocks.append((capture_tag, text))

    def handle_data(self, data: str) -> None:
        if self._capture_tag and self._in_content:
            self._buffer.append(data)

    @property
    def _in_content(self) -> bool:
        return self._content_depth > 0 and self._ignored_depth == 0

    def _start_section(self, level: int, title: str, anchor: str) -> None:
        self._path_by_level[level] = title
        for key in list(self._path_by_level):
            if key > level:
                self._path_by_level.pop(key, None)
        path = [self._path_by_level[key] for key in sorted(self._path_by_level)]
        self._current_section = HtmlSection(level=level, title=title, anchor=anchor, path=path, blocks=[])
        self.sections.append(self._current_section)


def extract_sections_from_html(html_text: str) -> list[HtmlSection]:
    parser = HtmlSingleSectionParser()
    parser.feed(html_text)
    return [section for section in parser.sections if section.title]


def build_virtual_paged_markdown(sections: list[HtmlSection]) -> str:
    pages: list[str] = []
    for index, section in enumerate(sections, start=1):
        heading = "#" * min(max(section.level, 1), 6)
        body = [f"## Page {index}", f"{heading} {section.title}", ""]
        for tag, text in section.blocks:
            if tag == "li":
                body.append(f"- {text}")
            elif tag in {"pre", "code"}:
                body.append("```text")
                body.append(text)
                body.append("```")
            else:
                body.append(text)
            body.append("")
        pages.append("\n".join(line for line in body if line is not None).strip())
    return "\n\n".join(page for page in pages if page).strip() + "\n"


def build_extracted_metadata(meta: dict, sections: list[HtmlSection]) -> dict:
    viewer_prefix = f"/docs/ocp/{meta['version']}/{meta['locale']}/{meta['book_slug']}/index.html"
    return {
        "source_url": meta["source_url"],
        "viewer_path": viewer_prefix,
        "book_slug": meta["book_slug"],
        "book_title": meta["book_title"],
        "source_lane": meta["source_lane"],
        "source_type": meta["source_type"],
        "source_collection": meta["source_collection"],
        "product": meta["product"],
        "version": meta["version"],
        "version_tag": meta["version"],
        "locale": meta["locale"],
        "source_language": meta["locale"],
        "display_language": meta["locale"],
        "translation_status": meta["translation_status"],
        "translation_stage": meta["translation_status"],
        "translation_source_url": meta["source_url"],
        "trust_score": 1.0,
        "verifiability": "anchor_backed",
        "pages": [
            {
                "page_number": index,
                "html_anchor": section.anchor or f"section-{index}",
                "section_title": section.title,
                "section_path": " > ".join(section.path),
                "blocks": [
                    {
                        "block_id": f"{section.anchor or f'section-{index}'}-block-{block_index}",
                        "html_anchor": section.anchor or f"section-{index}",
                        "block_type": "code" if tag in {"pre", "code"} else ("list" if tag == "li" else "paragraph"),
                        "section_title": section.title,
                        "section_path": " > ".join(section.path),
                    }
                    for block_index, (tag, _text) in enumerate(section.blocks or [("paragraph", section.title)], start=1)
                ],
            }
            for index, section in enumerate(sections, start=1)
        ],
    }


def import_ocp_html_single(
    raw_html_path: Path,
    raw_meta_path: Path,
    output_root: Path,
    extract_root: Path,
) -> dict:
    meta_payload = json.loads(raw_meta_path.read_text(encoding="utf-8-sig"))
    source_meta = {
        "book_slug": str(meta_payload.get("book_slug") or raw_html_path.stem),
        "book_title": str(meta_payload.get("book_title") or raw_html_path.stem.replace("_", " ")),
        "source_url": str(meta_payload.get("resolved_source_url") or meta_payload.get("source_url") or ""),
        "version": str(meta_payload.get("ocp_version") or "4.20"),
        "locale": str(meta_payload.get("resolved_language") or meta_payload.get("docs_language") or "en"),
        "source_lane": "official_en" if str(meta_payload.get("resolved_language") or meta_payload.get("docs_language") or "en") == "en" else "official_ko",
        "source_type": "official_doc",
        "source_collection": "core",
        "product": "openshift",
        "translation_status": "published_native" if str(meta_payload.get("resolved_language") or "en") == "en" else "approved_ko",
    }
    sections = extract_sections_from_html(raw_html_path.read_text(encoding="utf-8-sig"))
    if not sections:
        raise ValueError(f"No sections extracted from {raw_html_path}")

    source_dir = output_root / f"ocp-html-single-{source_meta['version']}-{source_meta['locale']}"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / f"{source_meta['book_slug']}.md"
    source_path.write_text(build_virtual_paged_markdown(sections), encoding="utf-8")

    html_artifact_path = extracted_html_path(extract_root, source_path)
    html_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    html_artifact_path.write_text(raw_html_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    metadata_artifact_path = extracted_metadata_path(extract_root, source_path)
    metadata_artifact_path.write_text(json.dumps(build_extracted_metadata(source_meta, sections), ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "source_path": str(source_path),
        "html_artifact_path": str(html_artifact_path),
        "metadata_artifact_path": str(metadata_artifact_path),
        "section_count": len(sections),
        "book_slug": source_meta["book_slug"],
        "version": source_meta["version"],
        "locale": source_meta["locale"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a docs.redhat.com html-single page into the local corpus.")
    parser.add_argument("--input-html", required=True)
    parser.add_argument("--input-meta", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--extract-root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    output_root = Path(args.output_root) if args.output_root else settings.rag_source_dir
    extract_root = Path(args.extract_root) if args.extract_root else settings.rag_extract_dir
    result = import_ocp_html_single(
        Path(args.input_html),
        Path(args.input_meta),
        output_root,
        extract_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

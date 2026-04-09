from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.types import Chunk, Document
from app.rag.utils import normalize_text, stable_hash, tokenize


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return pieces


def _normalize_markdown_text(text: str) -> str:
    collapsed = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    collapsed = re.sub(r"[ \t]+\n", "\n", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _infer_structure_flags_from_text(text: str, page_number: int | None) -> dict[str, bool]:
    normalized = str(text or "").replace("\r\n", "\n")
    lowered = normalized.casefold()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    short_lines = [line for line in lines if len(line) <= 90]
    numbered_heading_lines = [
        line for line in lines
        if re.match(r"^\d+(?:\.\d+){1,4}\.?\s+", line)
    ]
    is_toc = bool(
        any(marker in lowered for marker in ("table of contents", "contents", "목차"))
        or (
            len(numbered_heading_lines) >= 2
            and len(short_lines) >= max(2, len(lines) // 2)
        )
    )
    is_overview = bool(
        not is_toc
        and any(marker in lowered for marker in ("overview", "introduction", "about", "개요", "소개"))
    )
    is_intro = bool(
        not is_toc
        and (page_number or 0) <= 3
        and (
            is_overview
            or (len(lines) <= 4 and len(short_lines) >= max(1, len(lines) - 1))
        )
    )
    is_procedure = bool(
        any(marker in lowered for marker in ("procedure", "procedures", "steps", "step", "절차", "단계", "순서"))
        or any(re.match(r"^(?:\d+\.\s+|[-*]\s+)", line) for line in lines[:6])
    )
    return {
        "is_toc": is_toc,
        "is_intro": is_intro,
        "is_overview": is_overview,
        "is_procedure": is_procedure,
    }


@dataclass(slots=True)
class MarkdownBlock:
    text: str
    page_start: int
    page_end: int
    kind: str
    heading_level: int | None = None
    heading_path: tuple[str, ...] = ()


__all__ = [
    "MarkdownBlock",
    "_normalize_markdown_text",
    "_split_long_text",
    "_infer_structure_flags_from_text",
]

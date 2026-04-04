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


@dataclass(slots=True)
class MarkdownBlock:
    text: str
    page_start: int
    page_end: int
    kind: str
    heading_level: int | None = None
    heading_path: tuple[str, ...] = ()


class TextChunker:
    """페이지 단위 슬라이딩 윈도우로 텍스트를 고정 크기 청크로 분할한다."""

    def __init__(self, *, chunk_size: int, overlap: int) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document], markdown_text: str | None = None) -> list[Chunk]:  # noqa: ARG002
        chunks: list[Chunk] = []
        for document in documents:
            text = normalize_text(document.text)
            if not text:
                continue

            start = 0
            order = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk_id = stable_hash(f"{document.doc_id}:{order}:{chunk_text[:40]}")
                    page_number = document.page_number
                    metadata = {
                        **document.metadata,
                        "offset_start": start,
                        "offset_end": end,
                        "page_start": page_number,
                        "page_end": page_number,
                        "chunking_strategy": "page_window",
                    }
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=document.doc_id,
                            source_path=document.source_path,
                            text=chunk_text,
                            tokens=tokenize(chunk_text),
                            page_number=page_number,
                            metadata=metadata,
                        )
                    )

                if end >= len(text):
                    break
                start = max(0, end - self.overlap)
                order += 1
        return chunks


__all__ = [
    "MarkdownBlock",
    "TextChunker",
    "_normalize_markdown_text",
    "_split_long_text",
]

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Document:
    doc_id: str
    source_path: str
    page_number: int | None
    text: str
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    tokens: list[str]
    page_number: int | None
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ChatTurn:
    role: str
    content: str
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

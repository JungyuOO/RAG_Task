from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


LibraryDocumentGroup = Literal["official"]
LibraryOriginalKind = Literal["markdown", "pdf"]


class LibraryDocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = ""
    document_key: str
    title: str
    relative_path: str
    source_type: str
    group: LibraryDocumentGroup
    indexed: bool
    chunk_count: int = 0
    original_kind: LibraryOriginalKind
    original_key: str
    description: str = ""


class LibraryCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = ""
    documents: list[LibraryDocumentRecord] = Field(default_factory=list)
    message: str = ""


class LibraryChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    chunk_order: int | None = None
    page_number: int | None = None
    section_title: str = ""
    block_types: list[str] = Field(default_factory=list)
    preview_text: str = ""


class LibraryDocumentChunksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = ""
    document_key: str
    title: str
    chunk_count: int
    chunks: list[LibraryChunkRecord] = Field(default_factory=list)


class LibraryDocumentContentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = ""
    document_key: str
    title: str
    content: str


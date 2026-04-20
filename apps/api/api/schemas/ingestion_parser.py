from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    HTML_SINGLE = "html-single"
    HTML = "html"
    PDF = "pdf"
    GENERATED_MANUAL = "generated-manual"
    STRUCTURED_JSON = "structured-json"


class DocumentGroup(str, Enum):
    OFFICIAL_OCP = "official_ocp"
    CUSTOMER_GENERATED = "customer_generated"


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    ADMONITION = "admonition"
    METADATA = "metadata"


class SourceDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_path: str
    source_url: str = ""
    file_name: str = ""
    mime_type: str = ""
    locale: str = ""
    document_group: DocumentGroup = DocumentGroup.OFFICIAL_OCP
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_url: str = ""
    viewer_path: str = ""
    product: str = "openshift"
    locale: str = ""
    source_type: SourceType
    document_group: DocumentGroup
    trust_score: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source_path: str
    title: str = ""
    source_type: SourceType
    locale: str = ""
    document_group: DocumentGroup
    metadata: ParsedDocumentMetadata


class BlockAttributes(BaseModel):
    model_config = ConfigDict(extra="allow")

    heading_level: int | None = None
    language: str = ""
    table_headers: list[str] = Field(default_factory=list)
    admonition_kind: str = ""
    list_style: str = ""
    code_resource_kind: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class ParsedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    block_type: BlockType
    text: str
    html_anchor: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section_title: str = ""
    section_path: list[str] = Field(default_factory=list)
    resource_hints: list[str] = Field(default_factory=list)
    attributes: BlockAttributes = Field(default_factory=BlockAttributes)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocumentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: ParsedDocument
    blocks: list[ParsedBlock] = Field(default_factory=list)


class NormalizedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    doc_id: str
    block_type: BlockType
    normalized_text: str
    display_text: str
    html_anchor: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section_title: str = ""
    section_path: list[str] = Field(default_factory=list)
    resource_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkExpansionRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: str
    target_id: str


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    retrieval_text: str
    display_text: str
    chunk_order: int
    token_count: int = 0
    page_start: int | None = None
    page_end: int | None = None
    section_title: str = ""
    section_path: list[str] = Field(default_factory=list)
    html_anchor: str = ""
    block_ids: list[str] = Field(default_factory=list)
    resource_hints: list[str] = Field(default_factory=list)
    expansions: list[ChunkExpansionRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


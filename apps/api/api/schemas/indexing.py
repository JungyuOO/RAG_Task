from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from apps.api.api.schemas.batch_indexing import BatchIndexItem, BatchIndexRequest, BatchIndexResponse
from apps.api.api.schemas.batch_job_history import BatchJobListResponse
from apps.api.api.schemas.batch_jobs import BatchJobStatusResponse
from apps.api.api.schemas.index_admin import IndexResetResponse
from apps.api.api.schemas.ingestion_parser import (
    BlockAttributes,
    BlockType,
    ChunkExpansionRef,
    ChunkRecord,
    DocumentGroup,
    NormalizedBlock,
    ParsedBlock,
    ParsedDocument,
    ParsedDocumentBundle,
    ParsedDocumentMetadata,
    SourceDescriptor,
    SourceType,
)


class NewIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_type: SourceType
    source_path: str
    source_url: str = ""
    file_name: str = ""
    locale: str = ""
    document_group: DocumentGroup = DocumentGroup.OFFICIAL_OCP


class NewIndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    source_type: SourceType
    parsed_blocks: int
    normalized_blocks: int
    chunks: int
    indexed: bool
    parsed_artifact_path: str = ""
    normalized_artifact_path: str = ""
    chunk_artifact_path: str = ""


__all__ = [
    "BatchIndexItem",
    "BatchIndexRequest",
    "BatchIndexResponse",
    "BatchJobListResponse",
    "BatchJobStatusResponse",
    "BlockAttributes",
    "BlockType",
    "ChunkExpansionRef",
    "ChunkRecord",
    "DocumentGroup",
    "IndexResetResponse",
    "NewIndexRequest",
    "NewIndexResponse",
    "NormalizedBlock",
    "ParsedBlock",
    "ParsedDocument",
    "ParsedDocumentBundle",
    "ParsedDocumentMetadata",
    "SourceDescriptor",
    "SourceType",
]

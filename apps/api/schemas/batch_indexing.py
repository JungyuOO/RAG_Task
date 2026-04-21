from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from apps.api.schemas.ingestion_parser import DocumentGroup, SourceType


class BatchIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    root_path: str = ""
    explicit_source_paths: list[str] = Field(default_factory=list)
    source_type: SourceType | None = None
    document_group: DocumentGroup | None = None
    locale: str = ""
    max_files: int = 0
    include_subdirectories: bool = True


class BatchIndexItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    source_type: SourceType
    indexed: bool
    chunks: int
    error: str = ""


class BatchIndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discovered_files: int
    processed_files: int
    indexed_files: int
    failed_files: int
    progress_pct: int = 0
    current_file: str = ""
    items: list[BatchIndexItem] = Field(default_factory=list)




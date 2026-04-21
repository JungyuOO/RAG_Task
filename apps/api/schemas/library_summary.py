from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LibrarySourceBreakdownItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    count: int


class LibrarySummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = ""
    source_root: str
    extract_root: str
    corpus_files: int
    manifest_entries: int
    extracted_artifacts: int
    indexed_documents: int
    indexed_chunks: int
    batch_jobs: int
    latest_batch_status: str = ""
    source_breakdown: list[LibrarySourceBreakdownItem] = Field(default_factory=list)
    indexed_samples: list[str] = Field(default_factory=list)
    message: str = ""


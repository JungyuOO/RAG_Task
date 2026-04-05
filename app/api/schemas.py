from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    version_tag: str | None = None


class RetryChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str | None = None
    file_names: list[str] = Field(default_factory=list)


class RetrievalDebugRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    file_names: list[str] = Field(default_factory=list)


class BuildIndexResponse(BaseModel):
    task_id: str | None = None
    task_status: str | None = None
    indexed_files: int
    indexed_chunks: int
    skipped_files: int

class DeleteLibraryResponse(BuildIndexResponse):
    deleted_file: str
    deleted_markdown: bool


class LibraryDocument(BaseModel):
    file_name: str
    source_path: str
    extension: str
    indexed_pages: int
    indexed_chunks: int
    loaders: list[str]


class StartupIndexingStatus(BaseModel):
    status: str = "idle"  # "idle", "indexing", "done"
    total_files: int = 0
    completed_files: int = 0
    current_file: str = ""
    current_stage: str = ""
    current_chunk: int = 0
    total_chunks: int = 0
    progress_pct: int = 0


class LibraryStatusResponse(BaseModel):
    source_dir: str
    total_files: int
    indexed_documents: list[LibraryDocument]
    startup_indexing: StartupIndexingStatus = StartupIndexingStatus()
    reindexing: StartupIndexingStatus = StartupIndexingStatus()


class SessionHistoryItem(BaseModel):
    session_id: str
    title: str
    summary: str
    updated_at: str
    turn_count: int
    last_user_message: str = ""
    last_user_at: str = ""


class SessionHistoryResponse(BaseModel):
    sessions: list[SessionHistoryItem]


class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    payload: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    error: str = ""
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ChatTurnRequest:
    session_id: str
    message: str
    allowed_source_paths: set[str] | None = None
    append_user_turn: bool = True
    version_tag: str | None = None


@dataclass(slots=True)
class RetryChatRequestModel:
    session_id: str
    message: str
    owner_id: str | None = None
    allowed_source_paths: set[str] | None = None
    append_user_turn: bool = True

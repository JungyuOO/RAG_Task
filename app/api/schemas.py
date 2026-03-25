from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


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


class UploadLibraryResponse(BuildIndexResponse):
    uploaded_files: list[str]


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
    total: int = 0
    completed: int = 0
    current_file: str = ""


class LibraryStatusResponse(BaseModel):
    source_dir: str
    total_files: int
    indexed_documents: list[LibraryDocument]
    startup_indexing: StartupIndexingStatus = StartupIndexingStatus()


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

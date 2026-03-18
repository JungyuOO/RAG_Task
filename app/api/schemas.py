from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class RetryChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str | None = None
    file_names: list[str] = Field(default_factory=list)


class BuildIndexResponse(BaseModel):
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


class LibraryStatusResponse(BaseModel):
    source_dir: str
    total_files: int
    indexed_documents: list[LibraryDocument]


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

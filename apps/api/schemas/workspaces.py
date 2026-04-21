from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkspaceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    name: str
    slug: str
    industry: str = ""
    environment: str = ""
    created_at: datetime
    updated_at: datetime


class WorkspaceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[WorkspaceRecord] = Field(default_factory=list)


class WorkspaceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    slug: str = ""
    industry: str = ""
    environment: str = ""

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        next_value = str(value or "").strip()
        if not next_value:
            raise ValueError("name is required")
        return next_value


class WorkspaceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = ""
    slug: str = ""
    industry: str = ""
    environment: str = ""


class WorkspaceModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    chat_provider: str = "openai-compatible"
    chat_base_url: str = ""
    chat_model: str = ""
    chat_api_key_mode: str = "managed"
    embedding_provider: str = "tei"
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key_mode: str = "managed"
    updated_at: datetime


class WorkspaceModelProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chat_provider: str = "openai-compatible"
    chat_base_url: str = ""
    chat_model: str = ""
    chat_api_key_mode: str = "managed"
    embedding_provider: str = "tei"
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key_mode: str = "managed"

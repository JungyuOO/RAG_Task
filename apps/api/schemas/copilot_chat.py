from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CopilotChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str
    connection_id: str = ""
    namespace: str = ""
    history: list["CopilotChatHistoryTurn"] = Field(default_factory=list)


class CopilotChatHistoryTurn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    role: str
    text: str
    lane: str = ""
    source_paths: list[str] = Field(default_factory=list)
    resource_names: list[str] = Field(default_factory=list)
    namespace: str = ""


class CopilotChatSourceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: str
    label: str
    source_path: str = ""
    relative_source_path: str = ""
    page_number: int | None = None
    chunk_id: str = ""
    namespace: str = ""
    kind: str = ""
    score: float = 0.0
    provenance: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CopilotCitationMapItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_number: int
    source_index: int
    chunk_id: str = ""
    section_title: str = ""
    supporting_text: str = ""
    command_text: str = ""


class CopilotChatArtifactItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = ""
    namespace: str = ""
    resource: str = ""
    phase: str = ""
    type: str = ""
    host: str = ""
    to: str = ""
    node_name: str = ""
    cluster_ip: str = ""
    action_target: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CopilotChatArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    title: str = ""
    description: str = ""
    resource: str = ""
    namespace: str = ""
    resource_name: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    items: list[CopilotChatArtifactItem] = Field(default_factory=list)


class CopilotChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lane: str
    mode: str
    fallback_used: bool = False
    preview_ready: bool = False
    answer: str
    sources: list[CopilotChatSourceItem] = Field(default_factory=list)
    artifacts: list[CopilotChatArtifact] = Field(default_factory=list)
    citation_map: list[CopilotCitationMapItem] = Field(default_factory=list)


class CopilotChatStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    detail: str = ""
    status: str = "running"

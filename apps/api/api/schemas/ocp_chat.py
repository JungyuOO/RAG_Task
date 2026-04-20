from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from apps.api.api.schemas.copilot_chat import CopilotChatArtifact, CopilotChatHistoryTurn, CopilotChatSourceItem
from apps.api.api.schemas.ocp_live import OcpLiveResourceSummary


class OcpLiveChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection_id: str
    message: str
    namespace: str = ""
    history: list[CopilotChatHistoryTurn] = Field(default_factory=list)


class OcpLiveChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    cluster_url: str
    mode: str
    resource: str = ""
    namespace: str = ""
    answer: str
    items: list[OcpLiveResourceSummary] = Field(default_factory=list)
    sources: list[CopilotChatSourceItem] = Field(default_factory=list)
    artifacts: list[CopilotChatArtifact] = Field(default_factory=list)

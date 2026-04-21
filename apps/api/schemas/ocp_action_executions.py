from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from apps.api.schemas.ocp_actions import OcpActionPreviewResponse


class OcpActionExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OcpActionExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str
    request_id: str
    status: OcpActionExecutionStatus
    execution_mode: str = "simulated"
    simulated: bool = True
    preview: OcpActionPreviewResponse
    summary: str
    preflight_checks: list[str] = Field(default_factory=list)
    output_lines: list[str] = Field(default_factory=list)
    error: str = ""
    created_at: datetime
    updated_at: datetime


class OcpActionExecutionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[OcpActionExecutionRecord] = Field(default_factory=list)




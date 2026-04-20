from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from apps.api.api.schemas.ocp_actions import OcpActionPreviewResponse, OcpActionType


class OcpActionRequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class OcpActionRequestCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection_id: str
    actor_id: str = "ui"
    actor_roles: list[str] = Field(default_factory=list)
    action_type: OcpActionType
    namespace: str = ""
    resource_name: str = ""
    replicas: int = 1
    reason: str = ""
    break_glass: bool = False
    break_glass_reason: str = ""
    break_glass_ticket: str = ""
    manifest_yaml: str = ""
    resource_version: str | None = None


class OcpActionRequestDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    actor_id: str = "ui"
    actor_roles: list[str] = Field(default_factory=list)
    decision_note: str = ""


class OcpActionRequestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: OcpActionRequestStatus
    preview: OcpActionPreviewResponse
    requested_by: str = ""
    requested_roles: list[str] = Field(default_factory=list)
    required_approvals: int = 1
    approval_count: int = 0
    approver_ids: list[str] = Field(default_factory=list)
    approver_role_map: dict[str, list[str]] = Field(default_factory=dict)
    reason: str = ""
    manifest_yaml: str = ""
    resource_version: str | None = None
    decision_note: str = ""
    created_at: datetime
    updated_at: datetime


class OcpActionRequestListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[OcpActionRequestRecord] = Field(default_factory=list)



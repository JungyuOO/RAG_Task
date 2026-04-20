from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from apps.api.api.schemas.ocp_actions import OcpActionType


class OcpActionAuditEventType(str, Enum):
    REQUEST_CREATED = "request_created"
    REQUEST_APPROVED = "request_approved"
    REQUEST_REJECTED = "request_rejected"
    EXECUTION_SUCCEEDED = "execution_succeeded"
    EXECUTION_FAILED = "execution_failed"


class OcpActionAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: OcpActionAuditEventType
    actor_id: str = ""
    request_id: str = ""
    execution_id: str = ""
    action_type: OcpActionType
    namespace: str = ""
    resource_name: str = ""
    risk_level: str = ""
    decision_note: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class OcpActionAuditListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[OcpActionAuditRecord] = Field(default_factory=list)



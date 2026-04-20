from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OcpActionType(str, Enum):
    SCALE_DEPLOYMENT = "scale_deployment"
    ROLLOUT_RESTART = "rollout_restart"
    LOG_BUNDLE = "log_bundle"
    YAML_APPLY = "yaml_apply"


class OcpActionPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection_id: str
    actor_id: str = ""
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class OcpActionPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    action_type: OcpActionType
    namespace: str
    resource_name: str = ""
    allowed: bool
    risk_level: str
    summary: str
    preview_command: str = ""
    break_glass: bool = False
    break_glass_reason: str = ""
    break_glass_ticket: str = ""
    required_approvals: int = 1
    approval_strategy: str = "single_approval"
    requester_roles: list[str] = Field(default_factory=list)
    approver_roles: list[str] = Field(default_factory=list)
    executor_roles: list[str] = Field(default_factory=list)
    approval_rules: list[str] = Field(default_factory=list)
    policy_checks: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    validation_messages: list[str] = Field(default_factory=list)
    diff_unified: str = ""
    dry_run_status: Literal["ok", "rejected", "skipped"] = "skipped"
    dry_run_messages: list[str] = Field(default_factory=list)
    next_step: str = ""


from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class OcpAuthMode(str, Enum):
    TOKEN = "token"
    PASSWORD = "password"
    OAUTH_FUTURE = "oauth_future"


class OcpConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    cluster_url: str
    auth_mode: OcpAuthMode = OcpAuthMode.TOKEN
    verify_ssl: bool = True
    default_namespace: str = ""
    display_name: str = ""
    save_profile: bool = False
    token: SecretStr | None = None
    username: str = ""
    password: SecretStr | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cluster_url")
    @classmethod
    def _normalize_cluster_url(cls, value: str) -> str:
        normalized = str(value or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("cluster_url is required")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("cluster_url must start with http:// or https://")
        return normalized

    @field_validator("default_namespace")
    @classmethod
    def _normalize_namespace(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("username")
    @classmethod
    def _normalize_username(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_credentials(self) -> "OcpConnectionRequest":
        if self.auth_mode == OcpAuthMode.TOKEN:
            if self.token is None or not self.token.get_secret_value().strip():
                raise ValueError("token is required when auth_mode=token")
        elif self.auth_mode == OcpAuthMode.PASSWORD:
            if not self.username:
                raise ValueError("username is required when auth_mode=password")
            if self.password is None or not self.password.get_secret_value().strip():
                raise ValueError("password is required when auth_mode=password")
        return self


class OcpConnectionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    display_name: str = ""
    cluster_url: str
    auth_mode: OcpAuthMode
    verify_ssl: bool = True
    default_namespace: str = ""
    username_hint: str = ""
    secret_ref: str
    save_profile: bool = False
    status: str = "connected"
    last_verified_at: str = ""
    expires_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class OcpConnectionTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    connection_id: str
    cluster_url: str
    auth_mode: OcpAuthMode
    resolved_user: str = ""
    resolved_groups: list[str] = Field(default_factory=list)
    resolved_roles: list[str] = Field(default_factory=list)
    identity_source: str = ""
    permission_hints: dict[str, bool] = Field(default_factory=dict)
    rbac_evidence: list[str] = Field(default_factory=list)
    rbac_rules_incomplete: bool = False
    rbac_evaluation_error: str = ""
    secret_backend: str = ""
    secret_version: str = ""
    secret_created_at: str = ""
    secret_lease_renewable: bool = False
    secret_lease_ttl_seconds: int = 0
    secret_lease_expires_at: str = ""
    secret_rotation_supported: bool = False
    secret_auto_renew_applied: bool = False
    secret_auto_renew_threshold_seconds: int = 0
    secret_renew_message: str = ""
    resolved_namespace: str = ""
    expires_at: str = ""
    message: str = ""
    error: str = ""


class OcpConnectionTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection_id: str


class OcpConnectionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connected: bool
    connection: OcpConnectionProfile | None = None
    message: str = ""


class OcpDisconnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection_id: str


class OcpConnectionLeaseRefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection_id: str


class OcpLeaseSchedulerStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    running: bool
    interval_seconds: int
    last_run_at: str = ""
    last_success_at: str = ""
    last_failure_at: str = ""
    last_error: str = ""
    consecutive_failures: int = 0
    next_run_delay_seconds: int = 0
    alert_level: str = "none"
    profiles_checked: int = 0
    renewals_applied: int = 0
    recent_failures: list[str] = Field(default_factory=list)
    alert_delivery_enabled: bool = False
    alert_target: str = ""
    alert_dispatch_count: int = 0
    last_alert_at: str = ""
    last_alert_level: str = "none"
    last_alert_status: str = ""
    last_alert_error: str = ""


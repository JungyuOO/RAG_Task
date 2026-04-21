from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScmConnectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scm_connection_id: str
    workspace_id: str
    provider: str
    host_url: str = ""
    auth_type: str = "token"
    account_label: str = ""
    login_name: str = ""
    scopes: list[str] = Field(default_factory=list)
    secret_ref: str = ""
    status: str = "connected"
    created_at: datetime
    updated_at: datetime


class ScmConnectionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ScmConnectionRecord] = Field(default_factory=list)


class ScmConnectionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: str
    host_url: str = ""
    auth_type: str = "token"
    account_label: str = ""
    login_name: str = ""
    scopes: list[str] = Field(default_factory=list)
    secret_ref: str = ""

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        next_value = str(value or "").strip().lower()
        if next_value not in {"github", "gitlab"}:
            raise ValueError("provider must be github or gitlab")
        return next_value


class ScmRepositoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_id: str
    workspace_id: str
    scm_connection_id: str
    repo_full_name: str
    default_branch: str = "main"
    config_path: str = "config.yaml"
    delivery_mode: str = "gitops_commit"
    manifest_kind: str = "config_yaml"
    target_cluster_url: str = ""
    target_namespace: str = ""
    auto_deploy_enabled: bool = True
    sync_status: str = "configured"
    created_at: datetime
    updated_at: datetime


class ScmRepositoryListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ScmRepositoryRecord] = Field(default_factory=list)


class ScmRepositoryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scm_connection_id: str
    repo_full_name: str
    default_branch: str = "main"
    config_path: str = "config.yaml"
    delivery_mode: str = "gitops_commit"
    manifest_kind: str = "config_yaml"
    target_cluster_url: str = ""
    target_namespace: str = ""
    auto_deploy_enabled: bool = True

    @field_validator("repo_full_name")
    @classmethod
    def _validate_repo_full_name(cls, value: str) -> str:
        next_value = str(value or "").strip()
        if not next_value:
            raise ValueError("repo_full_name is required")
        return next_value


class ScmRepositoryUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    default_branch: str = ""
    config_path: str = ""
    delivery_mode: str = ""
    manifest_kind: str = ""
    target_cluster_url: str = ""
    target_namespace: str = ""
    auto_deploy_enabled: bool | None = None


class ScmDeploymentPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource_kind: str = "Deployment"
    resource_name: str
    target_namespace: str = ""
    replicas: int | None = None
    image_tag: str = ""
    config_key: str = "replicas"
    reason: str = ""


class ScmDeploymentPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_id: str
    workspace_id: str
    repo_full_name: str
    default_branch: str
    config_path: str
    delivery_mode: str
    manifest_kind: str
    target_cluster_url: str
    target_namespace: str
    auto_deploy_enabled: bool
    files_to_change: list[str] = Field(default_factory=list)
    suggested_updates: list[str] = Field(default_factory=list)
    trigger_kind: str = "gitops_sync"
    summary: str
    commit_title: str
    commit_body: str
    requires_pull_request: bool = True
    next_step: str

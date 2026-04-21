"""Grouped API schema exports."""

from apps.api.schemas.workspaces import (
    WorkspaceCreateRequest,
    WorkspaceListResponse,
    WorkspaceModelProfile,
    WorkspaceModelProfileUpdateRequest,
    WorkspaceRecord,
    WorkspaceUpdateRequest,
)
from apps.api.schemas.recommendations import (
    MetricSnapshotListResponse,
    MetricSnapshotRecord,
    RecommendationListResponse,
    RecommendationRecord,
    RecommendationRefreshRequest,
)
from apps.api.schemas.scm import (
    ScmConnectionCreateRequest,
    ScmConnectionListResponse,
    ScmConnectionRecord,
    ScmDeploymentPlanRequest,
    ScmDeploymentPlanResponse,
    ScmRepositoryCreateRequest,
    ScmRepositoryListResponse,
    ScmRepositoryRecord,
    ScmRepositoryUpdateRequest,
)

__all__ = [
    "WorkspaceCreateRequest",
    "WorkspaceListResponse",
    "WorkspaceModelProfile",
    "WorkspaceModelProfileUpdateRequest",
    "WorkspaceRecord",
    "WorkspaceUpdateRequest",
    "MetricSnapshotListResponse",
    "MetricSnapshotRecord",
    "RecommendationListResponse",
    "RecommendationRecord",
    "RecommendationRefreshRequest",
    "ScmConnectionCreateRequest",
    "ScmConnectionListResponse",
    "ScmConnectionRecord",
    "ScmDeploymentPlanRequest",
    "ScmDeploymentPlanResponse",
    "ScmRepositoryCreateRequest",
    "ScmRepositoryListResponse",
    "ScmRepositoryRecord",
    "ScmRepositoryUpdateRequest",
]

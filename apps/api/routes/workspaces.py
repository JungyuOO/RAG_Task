from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.runtime import (
    connected_ocp_service,
    connection_broker,
    metric_snapshot_repository,
    recommendation_log_repository,
    workspace_model_profile_repository,
    workspace_repository,
)
from apps.api.schemas.recommendations import (
    MetricSnapshotListResponse,
    RecommendationListResponse,
    RecommendationRefreshRequest,
)
from apps.api.schemas.workspaces import (
    WorkspaceCreateRequest,
    WorkspaceListResponse,
    WorkspaceModelProfile,
    WorkspaceModelProfileUpdateRequest,
    WorkspaceRecord,
    WorkspaceUpdateRequest,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("", response_model=WorkspaceListResponse)
async def list_workspaces() -> WorkspaceListResponse:
    return WorkspaceListResponse(items=workspace_repository.list_workspaces())


@router.post("", response_model=WorkspaceRecord)
async def create_workspace(request: WorkspaceCreateRequest) -> WorkspaceRecord:
    return workspace_repository.create_workspace(request)


@router.get("/{workspace_id}", response_model=WorkspaceRecord)
async def get_workspace(workspace_id: str) -> WorkspaceRecord:
    item = workspace_repository.get_workspace(workspace_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return item


@router.patch("/{workspace_id}", response_model=WorkspaceRecord)
async def update_workspace(workspace_id: str, request: WorkspaceUpdateRequest) -> WorkspaceRecord:
    try:
        return workspace_repository.update_workspace(workspace_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{workspace_id}/models/default", response_model=WorkspaceModelProfile)
async def get_default_workspace_model_profile(workspace_id: str) -> WorkspaceModelProfile:
    workspace = workspace_repository.get_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return workspace_model_profile_repository.get_profile(workspace_id)


@router.put("/{workspace_id}/models/default", response_model=WorkspaceModelProfile)
async def update_default_workspace_model_profile(
    workspace_id: str,
    request: WorkspaceModelProfileUpdateRequest,
) -> WorkspaceModelProfile:
    workspace = workspace_repository.get_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return workspace_model_profile_repository.put_profile(workspace_id, request)


@router.get("/{workspace_id}/metrics/snapshots", response_model=MetricSnapshotListResponse)
async def list_workspace_metric_snapshots(workspace_id: str, limit: int = 20) -> MetricSnapshotListResponse:
    workspace = workspace_repository.get_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return metric_snapshot_repository.list_recent(workspace_id=workspace_id, limit=limit)


@router.get("/{workspace_id}/recommendations", response_model=RecommendationListResponse)
async def list_workspace_recommendations(workspace_id: str, limit: int = 20) -> RecommendationListResponse:
    workspace = workspace_repository.get_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return recommendation_log_repository.list_recent(workspace_id=workspace_id, limit=limit)


@router.post("/{workspace_id}/recommendations/refresh", response_model=RecommendationListResponse)
async def refresh_workspace_recommendations(
    workspace_id: str,
    request: RecommendationRefreshRequest,
) -> RecommendationListResponse:
    workspace = workspace_repository.get_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    profile = None
    if request.connection_id:
        profile = connection_broker.get_profile(request.connection_id)
    else:
        profile = next(
            (
                item
                for item in connection_broker.profile_store.list_profiles()
                if (item.workspace_id or "") == workspace_id
            ),
            None,
        )
    if profile is None:
        raise HTTPException(status_code=404, detail="Workspace connection profile not found.")

    metrics = await connected_ocp_service.get_dashboard_metrics(
        profile.connection_id,
        connection_broker,
        window="1h",
        step="5m",
    )
    namespace = profile.default_namespace or ""
    for series in metrics.series:
        metric_snapshot_repository.create(
            workspace_id=workspace_id,
            connection_id=profile.connection_id,
            namespace=namespace,
            metric_key=series.metric_id,
            metric_value=series.current_value,
            unit=series.unit,
        )

    recommendations = []
    if namespace:
        deployments = await connected_ocp_service.list_resources(
            profile.connection_id,
            resource="deployments",
            namespace=namespace,
            broker=connection_broker,
        )
        for deployment in deployments.items:
            if deployment.replicas and deployment.ready_replicas < deployment.replicas:
                recommendations.append(
                    recommendation_log_repository.create(
                        workspace_id=workspace_id,
                        connection_id=profile.connection_id,
                        namespace=namespace,
                        recommendation_type="deployment_health",
                        risk_level="high",
                        summary=f"{deployment.name} has only {deployment.ready_replicas}/{deployment.replicas} ready replicas.",
                        rationale="Ready replica count is lower than desired replicas. Check rollout health or increase capacity.",
                        resource_kind=deployment.kind or "Deployment",
                        resource_name=deployment.name,
                    )
                )

    if not recommendations:
        recommendations.append(
            recommendation_log_repository.create(
                workspace_id=workspace_id,
                connection_id=profile.connection_id,
                namespace=namespace,
                recommendation_type="no_urgent_issue",
                risk_level="info",
                summary="No urgent scaling or rollout issues were detected in the current workspace connection.",
                rationale="Latest deployment and aggregate metric checks did not produce an immediate remediation signal.",
            )
        )

    return RecommendationListResponse(items=recommendations)

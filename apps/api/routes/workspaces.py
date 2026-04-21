from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.runtime import workspace_model_profile_repository, workspace_repository
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

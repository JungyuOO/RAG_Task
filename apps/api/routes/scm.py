from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.runtime import scm_connection_repository, scm_repository_repository, workspace_repository
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

router = APIRouter(prefix="/workspaces/{workspace_id}/scm", tags=["scm"])


@router.get("/connections", response_model=ScmConnectionListResponse)
async def list_scm_connections(workspace_id: str) -> ScmConnectionListResponse:
    if workspace_repository.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return scm_connection_repository.list_by_workspace(workspace_id)


@router.post("/connections", response_model=ScmConnectionRecord)
async def create_scm_connection(workspace_id: str, request: ScmConnectionCreateRequest) -> ScmConnectionRecord:
    if workspace_repository.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return scm_connection_repository.create(workspace_id, request)


@router.get("/repositories", response_model=ScmRepositoryListResponse)
async def list_scm_repositories(workspace_id: str) -> ScmRepositoryListResponse:
    if workspace_repository.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return scm_repository_repository.list_by_workspace(workspace_id)


@router.post("/repositories", response_model=ScmRepositoryRecord)
async def create_scm_repository(workspace_id: str, request: ScmRepositoryCreateRequest) -> ScmRepositoryRecord:
    if workspace_repository.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    connection = scm_connection_repository.get(request.scm_connection_id)
    if connection is None or connection.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="SCM connection not found.")
    return scm_repository_repository.create(workspace_id, request)


@router.patch("/repositories/{repository_id}", response_model=ScmRepositoryRecord)
async def update_scm_repository(
    workspace_id: str,
    repository_id: str,
    request: ScmRepositoryUpdateRequest,
) -> ScmRepositoryRecord:
    if workspace_repository.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    repository = scm_repository_repository.get(repository_id)
    if repository is None or repository.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Repository not found.")
    try:
        return scm_repository_repository.update(repository_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/repositories/{repository_id}/deployment-plan", response_model=ScmDeploymentPlanResponse)
async def build_scm_deployment_plan(
    workspace_id: str,
    repository_id: str,
    request: ScmDeploymentPlanRequest,
) -> ScmDeploymentPlanResponse:
    if workspace_repository.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    repository = scm_repository_repository.get(repository_id)
    if repository is None or repository.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Repository not found.")

    target_namespace = request.target_namespace or repository.target_namespace or "default"
    trigger_kind = "gitops_sync" if repository.delivery_mode == "gitops_commit" else "cicd_pipeline"
    files_to_change = [repository.config_path]
    suggested_updates: list[str] = []

    if repository.manifest_kind == "helm_values":
        if request.replicas is not None:
            suggested_updates.append(f"Set `replicaCount: {request.replicas}` in `{repository.config_path}`.")
        if request.image_tag:
            suggested_updates.append(f"Update `image.tag: {request.image_tag}` in `{repository.config_path}`.")
    elif repository.manifest_kind == "kustomize":
        if request.replicas is not None:
            suggested_updates.append(
                f"Adjust `{request.resource_kind}/{request.resource_name}` replica patch to `{request.replicas}` under `{repository.config_path}`."
            )
        if request.image_tag:
            suggested_updates.append(f"Update image tag to `{request.image_tag}` in the kustomize overlay under `{repository.config_path}`.")
    else:
        if request.replicas is not None:
            suggested_updates.append(f"Set `{request.config_key}: {request.replicas}` in `{repository.config_path}`.")
        if request.image_tag:
            suggested_updates.append(f"Set `imageTag: {request.image_tag}` in `{repository.config_path}`.")

    if not suggested_updates:
        suggested_updates.append(f"Update deployment values in `{repository.config_path}` for `{request.resource_kind}/{request.resource_name}`.")

    summary = (
        f"Modify `{repository.repo_full_name}` on `{repository.default_branch}` so {request.resource_kind} "
        f"`{request.resource_name}` in namespace `{target_namespace}` deploys through {trigger_kind.replace('_', ' ')}."
    )
    commit_title = f"deploy: update {request.resource_name} delivery config"
    commit_body = (
        f"Workspace `{workspace_id}` uses repo-driven delivery.\n"
        f"Update `{repository.config_path}` instead of applying live cluster YAML.\n"
        f"Reason: {request.reason or 'operational adjustment'}."
    )
    next_step = (
        "Open a commit or pull request against the connected repository so GitOps or CI/CD can roll the change into OpenShift."
        if repository.auto_deploy_enabled
        else "Prepare the repository change and wait for manual pipeline approval before deployment."
    )

    return ScmDeploymentPlanResponse(
        repository_id=repository.repository_id,
        workspace_id=repository.workspace_id,
        repo_full_name=repository.repo_full_name,
        default_branch=repository.default_branch,
        config_path=repository.config_path,
        delivery_mode=repository.delivery_mode,
        manifest_kind=repository.manifest_kind,
        target_cluster_url=repository.target_cluster_url,
        target_namespace=target_namespace,
        auto_deploy_enabled=repository.auto_deploy_enabled,
        files_to_change=files_to_change,
        suggested_updates=suggested_updates,
        trigger_kind=trigger_kind,
        summary=summary,
        commit_title=commit_title,
        commit_body=commit_body,
        requires_pull_request=True,
        next_step=next_step,
    )

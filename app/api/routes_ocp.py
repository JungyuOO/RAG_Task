from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.schemas import OcpNamespaceListResponse, OcpResourceListResponse, OcpResourceYamlResponse, OcpStatusResponse
from app.dependencies import AppContainer, get_container

router = APIRouter()


def _require_ocp_client(container: AppContainer):
    client = container.ocp_api_client
    if not client.enabled:
        raise HTTPException(status_code=503, detail="OCP API client is not configured.")
    return client


@router.get("/api/ocp/resources", response_model=OcpResourceListResponse)
async def list_ocp_resources(
    resource: str = Query(..., pattern="^(pods|deployments|services|routes|events)$"),
    namespace: str | None = Query(None),
    container: AppContainer = Depends(get_container),
) -> OcpResourceListResponse:
    client = _require_ocp_client(container)
    try:
        payload = await client.list_resources(resource, namespace=namespace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OcpResourceListResponse(**payload)


@router.get("/api/ocp/namespaces", response_model=OcpNamespaceListResponse)
async def list_ocp_namespaces(container: AppContainer = Depends(get_container)) -> OcpNamespaceListResponse:
    client = _require_ocp_client(container)
    try:
        payload = await client.list_namespaces()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OcpNamespaceListResponse(**payload)


@router.get("/api/ocp/status", response_model=OcpStatusResponse)
async def get_ocp_status(container: AppContainer = Depends(get_container)) -> OcpStatusResponse:
    client = container.ocp_api_client
    return OcpStatusResponse(
        enabled=client.enabled,
        base_url=client.base_url,
        default_namespace=client.default_namespace,
    )


@router.get("/api/ocp/resource-yaml", response_model=OcpResourceYamlResponse)
async def get_ocp_resource_yaml(
    resource: str = Query(..., pattern="^(pods|deployments|services|routes|events)$"),
    name: str = Query(..., min_length=1),
    namespace: str | None = Query(None),
    container: AppContainer = Depends(get_container),
) -> OcpResourceYamlResponse:
    client = _require_ocp_client(container)
    try:
        payload = await client.get_resource_yaml(resource, name=name, namespace=namespace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OcpResourceYamlResponse(**payload)

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query

from apps.api.api.schemas.ocp import (
    OcpDashboardMetricsResponse,
    OcpLiveNamespaceListResponse,
    OcpLiveResourceDetailResponse,
    OcpLiveResourceListResponse,
    OcpOverviewResponse,
)
from apps.api.runtime import connected_ocp_service, connection_broker

router = APIRouter(prefix="/ocp", tags=["ocp-live"])


def _raise_httpx_error(exc: httpx.HTTPStatusError) -> None:
    status_code = exc.response.status_code if exc.response is not None else 502
    detail = ""
    if exc.response is not None:
        try:
            payload = exc.response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = str(payload.get("message") or payload.get("detail") or payload.get("error") or "").strip()
        if not detail:
            detail = exc.response.text.strip()
    raise HTTPException(
        status_code=status_code,
        detail=detail or str(exc),
    ) from exc


@router.get("/overview/{connection_id}", response_model=OcpOverviewResponse)
async def get_ocp_overview(connection_id: str) -> OcpOverviewResponse:
    try:
        return await connected_ocp_service.get_overview(connection_id, connection_broker)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        _raise_httpx_error(exc)


@router.get("/metrics/{connection_id}", response_model=OcpDashboardMetricsResponse)
async def get_ocp_dashboard_metrics(
    connection_id: str,
    window: str = Query("1h", pattern="^(1h|6h|24h)$"),
    step: str = Query("5m", pattern="^(1m|5m|15m)$"),
) -> OcpDashboardMetricsResponse:
    try:
        return await connected_ocp_service.get_dashboard_metrics(
            connection_id,
            connection_broker,
            window=window,
            step=step,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        _raise_httpx_error(exc)


@router.get("/namespaces/{connection_id}", response_model=OcpLiveNamespaceListResponse)
async def list_ocp_namespaces(connection_id: str) -> OcpLiveNamespaceListResponse:
    try:
        return await connected_ocp_service.list_namespaces(connection_id, connection_broker)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        _raise_httpx_error(exc)


@router.get("/resources/{connection_id}", response_model=OcpLiveResourceListResponse)
async def list_ocp_resources(
    connection_id: str,
    resource: str = Query(..., pattern="^(pods|deployments|services|routes|events)$"),
    namespace: str = Query(""),
) -> OcpLiveResourceListResponse:
    try:
        return await connected_ocp_service.list_resources(
            connection_id,
            resource=resource,
            namespace=namespace,
            broker=connection_broker,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        _raise_httpx_error(exc)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/resource-detail/{connection_id}", response_model=OcpLiveResourceDetailResponse)
async def get_ocp_resource_detail(
    connection_id: str,
    resource: str = Query(..., pattern="^(pods|deployments|services|routes|events)$"),
    namespace: str = Query(""),
    name: str = Query(..., min_length=1),
) -> OcpLiveResourceDetailResponse:
    try:
        return await connected_ocp_service.get_resource_detail(
            connection_id,
            resource=resource,
            namespace=namespace,
            name=name,
            broker=connection_broker,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        _raise_httpx_error(exc)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

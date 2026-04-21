from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from apps.api.schemas.actions import (
    OcpActionAuditListResponse,
    OcpActionExecuteRequest,
    OcpActionExecutionListResponse,
    OcpActionExecutionRecord,
    OcpActionPreviewRequest,
    OcpActionPreviewResponse,
    OcpActionRequestCreateRequest,
    OcpActionRequestDecisionRequest,
    OcpActionRequestListResponse,
    OcpActionRequestRecord,
)
from apps.api.runtime import (
    action_audit_service,
    action_execution_service,
    action_preview_service,
    action_request_service,
    connection_broker,
)

router = APIRouter(prefix="/actions", tags=["actions"])


@router.post("/preview", response_model=OcpActionPreviewResponse)
async def preview_ocp_action(request: OcpActionPreviewRequest) -> OcpActionPreviewResponse:
    try:
        return action_preview_service.build_preview(request, connection_broker)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/requests", response_model=OcpActionRequestRecord)
async def create_action_request(request: OcpActionRequestCreateRequest) -> OcpActionRequestRecord:
    try:
        return action_request_service.create(request, connection_broker)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/requests", response_model=OcpActionRequestListResponse)
async def list_action_requests(limit: int = Query(20, ge=1, le=100)) -> OcpActionRequestListResponse:
    return action_request_service.list_recent(limit=limit)


@router.post("/requests/{request_id}/approve", response_model=OcpActionRequestRecord)
async def approve_action_request(request_id: str, request: OcpActionRequestDecisionRequest) -> OcpActionRequestRecord:
    try:
        return action_request_service.approve(request_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/requests/{request_id}/reject", response_model=OcpActionRequestRecord)
async def reject_action_request(request_id: str, request: OcpActionRequestDecisionRequest) -> OcpActionRequestRecord:
    try:
        return action_request_service.reject(request_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/requests/{request_id}/execute", response_model=OcpActionExecutionRecord)
async def execute_action_request(request_id: str, request: OcpActionExecuteRequest) -> OcpActionExecutionRecord:
    try:
        return action_execution_service.execute(request_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/executions", response_model=OcpActionExecutionListResponse)
async def list_action_executions(limit: int = Query(20, ge=1, le=100)) -> OcpActionExecutionListResponse:
    return action_execution_service.list_recent(limit=limit)


@router.get("/audit", response_model=OcpActionAuditListResponse)
async def list_action_audit(limit: int = Query(20, ge=1, le=100)) -> OcpActionAuditListResponse:
    return action_audit_service.list_recent(limit=limit)


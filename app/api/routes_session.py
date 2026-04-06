from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.routes_shared import logger, resolve_owner_id, scoped_session_id, unscoped_session_id
from app.api.schemas import SessionHistoryResponse
from app.dependencies import AppContainer, get_container

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions(request: Request, container: AppContainer = Depends(get_container)) -> SessionHistoryResponse:
    owner_id = resolve_owner_id(request)
    logger.info("[SessionList] owner_id=%s", owner_id)
    sessions = container.session_repository.list_sessions(owner_id=owner_id)
    sanitized = [{**item, "session_id": unscoped_session_id(owner_id, str(item.get("session_id", "")))} for item in sessions]
    return SessionHistoryResponse(sessions=sanitized)


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request, container: AppContainer = Depends(get_container)) -> dict:
    owner_id = resolve_owner_id(request)
    logger.info("[SessionLoad] owner_id=%s session_id=%s", owner_id, session_id)
    exported = container.session_repository.export_session(scoped_session_id(owner_id, session_id), owner_id)
    if not exported:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {**exported, "session_id": session_id}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request, container: AppContainer = Depends(get_container)) -> dict:
    owner_id = resolve_owner_id(request)
    logger.info("[SessionDelete] owner_id=%s session_id=%s", owner_id, session_id)
    deleted = container.session_repository.delete_session(scoped_session_id(owner_id, session_id), owner_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True, "session_id": session_id}

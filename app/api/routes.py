from __future__ import annotations

from fastapi import APIRouter

from app.api.routes_chat import router as chat_router
from app.api.routes_library import router as library_router
from app.api.routes_ocp import router as ocp_router
from app.api.routes_session import router as session_router

router = APIRouter()
router.include_router(library_router)
router.include_router(session_router)
router.include_router(chat_router)
router.include_router(ocp_router)

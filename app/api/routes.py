from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter()


@router.get("/")
async def index() -> FileResponse:
    return FileResponse("app/web/index.html")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.api.routes.actions import router as actions_router
from apps.api.routes.auth import router as ocp_auth_router
from apps.api.routes.chat import router as chat_router
from apps.api.routes.docs_preview import router as docs_preview_router
from apps.api.routes.indexing import router as index_router
from apps.api.routes.library import router as library_router
from apps.api.routes.ocp import router as ocp_live_router
from apps.api.routes.workspaces import router as workspaces_router
from apps.api.runtime import chat_llm_client, connection_lease_scheduler


@asynccontextmanager
async def _lifespan(_: FastAPI):
    await connection_lease_scheduler.start()
    try:
        yield
    finally:
        await chat_llm_client.close()
        await connection_lease_scheduler.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Task API (Next Layout)", lifespan=_lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(actions_router, prefix="/api/v1")
    app.include_router(ocp_auth_router, prefix="/api/v1/auth")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(docs_preview_router, prefix="/api/v1")
    app.include_router(index_router, prefix="/api/v1")
    app.include_router(library_router, prefix="/api/v1")
    app.include_router(ocp_live_router, prefix="/api/v1")
    app.include_router(workspaces_router, prefix="/api/v1")
    return app


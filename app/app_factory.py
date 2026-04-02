from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, get_settings
from app.dependencies import build_container

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Track startup and reindex progress for the app lifecycle."""
    import asyncio

    startup_state: dict = {
        "status": "idle",
        "total_files": 0,
        "completed_files": 0,
        "current_file": "",
        "current_stage": "",
        "current_chunk": 0,
        "total_chunks": 0,
        "progress_pct": 0,
    }
    reindex_state: dict = {
        "status": "idle",
        "total_files": 0,
        "completed_files": 0,
        "current_file": "",
        "current_stage": "",
        "current_chunk": 0,
        "total_chunks": 0,
        "progress_pct": 0,
    }
    app.state.startup_indexing = startup_state
    app.state.reindexing = reindex_state
    container = app.state.container
    task = asyncio.create_task(
        asyncio.to_thread(container.indexing_service.sync_unindexed_documents, startup_state)
    )
    try:
        yield
    finally:
        task.cancel()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.state.container = build_container(resolved_settings)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="app/web"), name="static")
    app.mount("/resources", StaticFiles(directory="app/resources"), name="resources")
    return app

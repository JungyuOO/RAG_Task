from __future__ import annotations

import asyncio
import logging
import time
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

logger = logging.getLogger("rag.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Track startup and reindex progress for the app lifecycle."""
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

    def _startup_prepare_index() -> None:
        t_total = time.perf_counter()
        logger.info("[Startup] index preparation started")
        startup_state.update(status="indexing", current_stage="indexing", progress_pct=0)
        logger.info("[Startup] auto indexing started")
        container.indexing_service.sync_unindexed_documents(startup_state)
        startup_state.update(status="indexing", current_stage="warming_cache", progress_pct=95)
        logger.info("[Startup] auto indexing completed, warming cache")
        t_warm = time.perf_counter()
        warmed_items = container.pipeline.index_repository.warm_cache()
        logger.info(
            "[Timing][Startup] warm_cache=%.3fs items=%d",
            time.perf_counter() - t_warm,
            warmed_items,
        )
        startup_state.update(
            status="done",
            current_file="",
            current_stage="done",
            current_chunk=0,
            total_chunks=warmed_items,
            progress_pct=100,
        )
        logger.info("[Timing][Startup] total=%.3fs", time.perf_counter() - t_total)

    try:
        if container.settings.startup_auto_index_enabled:
            startup_state.update(status="indexing", current_stage="startup", progress_pct=0)
            await asyncio.to_thread(_startup_prepare_index)
        else:
            startup_state.update(status="done", current_stage="skipped", progress_pct=100)
        yield
    finally:
        pass


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.state.container = build_container(resolved_settings)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="app/web"), name="static")
    app.mount("/resources", StaticFiles(directory="app/resources"), name="resources")
    return app

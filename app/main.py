from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.dependencies import build_container

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("rag.startup")


def _sync_index_on_startup(container, startup_state: dict) -> None:
    """앱 시작 시 아직 인덱싱되지 않은 PDF를 백그라운드에서 채운다."""
    from pathlib import Path

    library = container.pipeline.list_library_documents()
    unindexed = [doc for doc in library["indexed_documents"] if doc["indexed_chunks"] == 0]
    if not unindexed:
        logger.info("모든 문서가 이미 인덱싱된 상태입니다.")
        startup_state.update(status="done", progress_pct=100)
        return

    total = len(unindexed)
    startup_state.update(
        status="indexing",
        total_files=total,
        completed_files=0,
        current_file="",
        current_stage="",
        current_chunk=0,
        total_chunks=0,
        progress_pct=0,
    )
    logger.info("%d개 미인덱싱 문서를 발견해 자동 인덱싱을 시작합니다.", total)

    for index, doc in enumerate(unindexed):
        source_path = Path(doc["source_path"])
        startup_state.update(
            current_file=doc["file_name"],
            completed_files=index,
            current_stage="prepare",
            current_chunk=0,
            total_chunks=0,
            progress_pct=int((index / max(total, 1)) * 100),
        )
        if not source_path.exists():
            logger.warning("파일이 존재하지 않아 건너뜁니다. %s", source_path)
            continue

        def progress_callback(stage, current, chunk_total, meta=None):
            file_progress = current / max(chunk_total, 1)
            overall_progress = ((index + file_progress) / max(total, 1)) * 100
            startup_state.update(
                current_file=doc["file_name"],
                current_stage=stage,
                current_chunk=current,
                total_chunks=chunk_total,
                progress_pct=min(100, int(overall_progress)),
            )

        try:
            result = container.pipeline.index_single_file(source_path, progress_callback)
            logger.info(
                "자동 인덱싱 완료: %s (청크 %d개, 페이지 %d개)",
                doc["file_name"],
                result.get("indexed_chunks", 0),
                result.get("indexed_pages", 0),
            )
            startup_state.update(
                completed_files=index + 1,
                current_stage="done",
                current_chunk=result.get("indexed_chunks", 0),
                total_chunks=result.get("indexed_chunks", 0),
                progress_pct=min(100, int(((index + 1) / max(total, 1)) * 100)),
            )
        except Exception:
            logger.exception("자동 인덱싱 실패: %s", doc["file_name"])

    startup_state.update(
        status="done",
        completed_files=total,
        current_file="",
        current_stage="done",
        current_chunk=0,
        total_chunks=0,
        progress_pct=100,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 수명주기 동안 startup/reindex 진행 상태를 유지한다."""
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
    task = asyncio.create_task(asyncio.to_thread(_sync_index_on_startup, container, startup_state))
    try:
        yield
    finally:
        task.cancel()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.container = build_container(settings)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="app/web"), name="static")
    app.mount("/resources", StaticFiles(directory="app/resources"), name="resources")
    return app


app = create_app()

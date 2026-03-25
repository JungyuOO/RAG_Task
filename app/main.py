from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("rag.startup")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.dependencies import build_container


def _sync_index_on_startup(container, startup_state: dict) -> None:
    """디스크에 PDF가 있지만 DB 인덱스가 없는 파일을 자동으로 인덱싱한다."""
    from pathlib import Path

    library = container.pipeline.list_library_documents()
    unindexed = [
        doc for doc in library["indexed_documents"]
        if doc["indexed_chunks"] == 0
    ]
    if not unindexed:
        logger.info("모든 문서가 인덱싱 완료 상태입니다.")
        startup_state["status"] = "done"
        return

    total = len(unindexed)
    startup_state.update(status="indexing", total=total, completed=0, current_file="")
    logger.info("%d개 미인덱싱 문서 발견 — 자동 인덱싱을 시작합니다.", total)

    for i, doc in enumerate(unindexed):
        source_path = Path(doc["source_path"])
        startup_state.update(current_file=doc["file_name"], completed=i)
        if not source_path.exists():
            logger.warning("파일이 존재하지 않아 건너뜁니다: %s", source_path)
            continue
        try:
            result = container.pipeline.index_single_file(source_path)
            logger.info(
                "자동 인덱싱 완료: %s (청크 %d개, 페이지 %d개)",
                doc["file_name"],
                result.get("indexed_chunks", 0),
                result.get("indexed_pages", 0),
            )
        except Exception:
            logger.exception("자동 인덱싱 실패: %s", doc["file_name"])

    startup_state.update(status="done", completed=total, current_file="")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 미인덱싱 문서를 백그라운드에서 자동 인덱싱한다."""
    import asyncio
    startup_state: dict = {"status": "idle", "total": 0, "completed": 0, "current_file": ""}
    app.state.startup_indexing = startup_state
    container = app.state.container
    task = asyncio.create_task(asyncio.to_thread(_sync_index_on_startup, container, startup_state))
    yield
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

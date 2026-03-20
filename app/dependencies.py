from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.config import Settings, get_settings
from app.rag.pipeline import RagPipeline
from app.repositories.task_repository import TaskRepository
from app.services.task_service import TaskService


@dataclass(slots=True)
class AppContainer:
    """애플리케이션 의존성 컨테이너 — 서버 시작 시 한 번 생성되어 전 라우트에서 공유된다.

    RagPipeline(검색·LLM·세션 관리)과 TaskService(비동기 태스크)를
    하나의 진입점으로 묶어 라우트가 Depends(get_container)로 접근한다.
    """

    settings: Settings
    pipeline: RagPipeline
    task_service: TaskService


def build_container(settings: Settings | None = None) -> AppContainer:
    """Settings를 주입받아 파이프라인과 태스크 서비스를 초기화하고 컨테이너를 반환한다."""
    resolved_settings = settings or get_settings()
    pipeline = RagPipeline(resolved_settings)
    task_repository = TaskRepository(resolved_settings.rag_index_dir / "task_store.sqlite3")
    task_service = TaskService(task_repository)
    return AppContainer(
        settings=resolved_settings,
        pipeline=pipeline,
        task_service=task_service,
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container

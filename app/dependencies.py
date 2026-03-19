from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.config import Settings, get_settings
from app.rag.pipeline import RagPipeline
from app.repositories.task_repository import TaskRepository
from app.services.task_service import TaskService


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    pipeline: RagPipeline
    task_service: TaskService


def build_container(settings: Settings | None = None) -> AppContainer:
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

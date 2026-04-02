from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.config import Settings, get_settings
from app.rag.chat_service import ChatService
from app.rag.pipeline import RagPipeline
from app.rag.indexing import IndexingService
from app.session.repository import SessionRepository
from app.storage import TaskRepository


@dataclass(slots=True)
class AppContainer:
    """Shared runtime container exposed to FastAPI routes."""

    settings: Settings
    pipeline: RagPipeline
    chat_service: ChatService
    session_repository: SessionRepository
    indexing_service: IndexingService
    task_repository: TaskRepository


def build_container(settings: Settings | None = None) -> AppContainer:
    """Build the runtime container from settings."""
    resolved_settings = settings or get_settings()
    pipeline = RagPipeline(resolved_settings)
    session_repository = SessionRepository(pipeline.session_store)
    task_repository = TaskRepository(resolved_settings.db_dsn)
    return AppContainer(
        settings=resolved_settings,
        pipeline=pipeline,
        chat_service=ChatService(pipeline, session_repository),
        session_repository=session_repository,
        indexing_service=pipeline.indexing_service,
        task_repository=task_repository,
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from app.config import Settings, get_settings
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


class ChatService:
    """Public entrypoint for chat and retry flows."""

    def __init__(self, pipeline: Any, session_repository: Any) -> None:
        self.pipeline = pipeline
        self.session_repository = session_repository

    def stream(self, request: Any):
        return self.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=request.message,
            allowed_source_paths=request.allowed_source_paths,
            append_user_turn=request.append_user_turn,
            version_tag=getattr(request, "version_tag", None),
        )

    def retry(self, request: Any):
        requested_message = (request.message or "").strip()
        pending_message = self.session_repository.pending_user_message(
            request.session_id,
            owner_id=request.owner_id,
        )
        user_message = pending_message or requested_message
        if not user_message:
            raise LookupError("No pending user message found for retry.")

        append_user_turn = request.append_user_turn
        if pending_message and (not requested_message or pending_message == requested_message):
            append_user_turn = False
        return self.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=user_message,
            allowed_source_paths=request.allowed_source_paths,
            append_user_turn=append_user_turn,
        )

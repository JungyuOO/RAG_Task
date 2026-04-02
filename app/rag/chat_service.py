"""Top-level chat service entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChatTurnRequest:
    session_id: str
    message: str
    allowed_source_paths: set[str] | None = None
    append_user_turn: bool = True


@dataclass(slots=True)
class RetryChatRequestModel:
    session_id: str
    message: str
    owner_id: str | None = None
    allowed_source_paths: set[str] | None = None
    append_user_turn: bool = True


class ChatService:
    """Public entrypoint for chat and retry flows."""

    def __init__(self, pipeline: Any, session_repository: Any) -> None:
        self.pipeline = pipeline
        self.session_repository = session_repository

    def stream(self, request: ChatTurnRequest):
        return self.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=request.message,
            allowed_source_paths=request.allowed_source_paths,
            append_user_turn=request.append_user_turn,
        )

    def retry(self, request: RetryChatRequestModel):
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

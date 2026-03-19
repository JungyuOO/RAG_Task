from __future__ import annotations

from app.rag.memory import SessionStore
from app.rag.types import ChatTurn


class SessionRepository:
    def __init__(self, backend: SessionStore) -> None:
        self.backend = backend

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        self.backend.add_turn(session_id, role, content, metadata=metadata)

    def delete_session(self, session_id: str) -> bool:
        return self.backend.delete_session(session_id)

    def recent_turns(self, session_id: str) -> list[ChatTurn]:
        return self.backend.recent_turns(session_id)

    def summary(self, session_id: str) -> str:
        return self.backend.summary(session_id)

    def structured_summary(self, session_id: str) -> dict:
        return self.backend.structured_summary(session_id)

    def topic_state(self, session_id: str) -> dict:
        return self.backend.topic_state(session_id)

    def memory_snapshot(self, session_id: str) -> dict:
        return self.backend.memory_snapshot(session_id)

    def rewrite_query(self, session_id: str, user_message: str) -> str:
        return self.backend.rewrite_query(session_id, user_message)

    def export_session(self, session_id: str) -> dict:
        return self.backend.export_session(session_id)

    def pending_user_message(self, session_id: str) -> str | None:
        return self.backend.pending_user_message(session_id)

    def list_sessions(self, limit: int = 50) -> list[dict]:
        return self.backend.list_sessions(limit=limit)

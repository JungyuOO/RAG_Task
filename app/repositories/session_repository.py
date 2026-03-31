from __future__ import annotations

from app.rag.memory import SessionStore
from app.rag.types import ChatTurn


class SessionRepository:
    """세션 데이터 접근 계층 — SessionStore와 파이프라인 사이의 추상 경계.

    현재는 SessionStore에 직접 위임하지만, 향후 캐싱·로깅·트랜잭션 관리 등
    교차 관심사를 이 계층에서 일괄 적용할 수 있도록 분리한다.
    """

    def __init__(self, backend: SessionStore) -> None:
        self.backend = backend

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> int:
        return self.backend.add_turn(session_id, role, content, metadata=metadata)

    def delete_session(self, session_id: str, owner_id: str | None = None) -> bool:
        return self.backend.delete_session(session_id, owner_id=owner_id)

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

    def create_topic(self, session_id: str, seed_label: str, seed_turn_id: int | None = None) -> dict:
        return self.backend.create_topic(session_id, seed_label, seed_turn_id=seed_turn_id)

    def list_topics(self, session_id: str) -> list[dict]:
        return self.backend.list_topics(session_id)

    def get_topic(self, topic_id: str) -> dict | None:
        return self.backend.get_topic(topic_id)

    def get_last_active_topic(self, session_id: str) -> dict | None:
        return self.backend.get_last_active_topic(session_id)

    def link_turn_to_topic(
        self,
        turn_id: int,
        session_id: str,
        topic_id: str,
        role: str,
        link_type: str,
        confidence: float,
    ) -> None:
        self.backend.link_turn_to_topic(turn_id, session_id, topic_id, role, link_type, confidence)

    def recent_topic_turns(self, session_id: str, topic_id: str, limit: int | None = None) -> list[ChatTurn]:
        return self.backend.recent_topic_turns(session_id, topic_id, limit=limit)

    def topic_memory_snapshot(self, session_id: str, topic_id: str) -> dict:
        return self.backend.topic_memory_snapshot(session_id, topic_id)

    def refresh_topic_memory(self, session_id: str, topic_id: str) -> None:
        self.backend.refresh_topic_memory(session_id, topic_id)

    def build_rewrite_context(self, session_id: str, user_message: str) -> dict | None:
        return self.backend.build_rewrite_context(session_id, user_message)

    def export_session(self, session_id: str, owner_id: str | None = None) -> dict:
        return self.backend.export_session(session_id, owner_id=owner_id)

    def pending_user_message(self, session_id: str, owner_id: str | None = None) -> str | None:
        return self.backend.pending_user_message(session_id, owner_id=owner_id)

    def list_sessions(
        self,
        limit: int = 50,
        session_prefix: str | None = None,
        owner_id: str | None = None,
    ) -> list[dict]:
        return self.backend.list_sessions(limit=limit, session_prefix=session_prefix, owner_id=owner_id)

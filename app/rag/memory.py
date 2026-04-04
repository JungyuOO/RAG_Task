from __future__ import annotations

from contextlib import contextmanager
import json

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.rag.memory_schema import initialize_session_store, SessionStoreTopicsMixin
from app.rag.types import ChatTurn
from app.rag.utils import normalize_text
from app.session.state import (
    DEFAULT_SUMMARY,
    DEFAULT_TOPIC_STATE,
    build_rewrite_context_payload,
    build_summary_bundle,
    extract_entities,
    merge_defaults,
)
from app.session.store_sql import persist_summary

_SESSION_SELECT = """
    SELECT s.session_id, s.summary, s.summary_json, s.updated_at,
           (SELECT content FROM session_turns WHERE session_id = s.session_id AND role = 'user' ORDER BY turn_id ASC LIMIT 1) AS first_user_message,
           (SELECT content FROM session_turns WHERE session_id = s.session_id AND role = 'user' ORDER BY turn_id DESC LIMIT 1) AS last_user_message,
           (SELECT created_at FROM session_turns WHERE session_id = s.session_id AND role = 'user' ORDER BY turn_id DESC LIMIT 1) AS last_user_at,
           (SELECT COUNT(*) FROM session_turns WHERE session_id = s.session_id) AS turn_count
    FROM sessions s
"""


def _row_to_turn(row) -> ChatTurn:
    return ChatTurn(role=row["role"], content=row["content"], metadata=json.loads(row["metadata"] or "{}"))


class SessionStoreSummaryMixin:
    def build_rewrite_context(self, session_id: str, user_message: str) -> dict | None:  # noqa: ARG002
        recent = self.recent_turns(session_id)
        summary = self.structured_summary(session_id)
        topic_state = self.topic_state(session_id)
        return build_rewrite_context_payload(recent, summary, topic_state)

    def recent_turns(self, session_id: str, limit: int | None = None) -> list[ChatTurn]:
        effective_limit = limit if limit is not None else self.memory_window_turns
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT role, content, metadata FROM (
                        SELECT role, content, metadata, turn_id
                        FROM session_turns
                        WHERE session_id = %s
                        ORDER BY turn_id DESC
                        LIMIT %s
                    ) sub ORDER BY turn_id ASC
                    """,
                    (session_id, effective_limit),
                )
                rows = cursor.fetchall()
        return [_row_to_turn(row) for row in rows]

    def _refresh_summary(self, session_id: str) -> None:
        turns = self.recent_turns(session_id, limit=10)
        summary_json, topic_state, summary = build_summary_bundle(turns)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                persist_summary(cursor, session_id, summary, summary_json, topic_state)

    def _extract_entities(self, text: str) -> list[str]:
        return extract_entities(text)


class SessionStore(SessionStoreTopicsMixin, SessionStoreSummaryMixin):
    """LLM 히스토리에 의존하지 않는 애플리케이션 수준 세션 메모리."""

    def __init__(self, dsn: str, memory_window_turns: int) -> None:
        self.dsn = dsn
        self.memory_window_turns = memory_window_turns
        self._pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn)
        self._initialize()

    @contextmanager
    def _connection(self):
        connection = self._pool.getconn()
        connection.autocommit = False
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self._pool.putconn(connection)

    def close(self) -> None:
        self._pool.closeall()

    @staticmethod
    def _owner_from_session_id(session_id: str) -> str:
        if "::" in session_id:
            return session_id.split("::", 1)[0]
        return "legacy"

    def _initialize(self) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                initialize_session_store(cursor)

    def _ensure_session_row(self, cursor, session_id: str) -> None:
        owner_id = self._owner_from_session_id(session_id)
        cursor.execute(
            """
            INSERT INTO sessions (session_id, owner_id, summary, summary_json, topic_state_json)
            VALUES (%s, %s, '', '{}', '{}')
            ON CONFLICT (session_id) DO NOTHING
            """,
            (session_id, owner_id),
        )

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> int:
        normalized_content = normalize_text(content)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                self._ensure_session_row(cursor, session_id)
                cursor.execute(
                    """
                    INSERT INTO session_turns (owner_id, session_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING turn_id
                    """,
                    (self._owner_from_session_id(session_id), session_id, role, normalized_content, metadata_json),
                )
                turn_id = int(cursor.fetchone()[0])
        self._refresh_summary(session_id)
        return turn_id

    def delete_session(self, session_id: str, owner_id: str | None = None) -> bool:
        resolved_owner_id = owner_id or self._owner_from_session_id(session_id)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM turn_topic_links WHERE session_id = %s AND owner_id = %s",
                    (session_id, resolved_owner_id),
                )
                cursor.execute(
                    "DELETE FROM session_topics WHERE session_id = %s AND owner_id = %s",
                    (session_id, resolved_owner_id),
                )
                cursor.execute(
                    "DELETE FROM session_turns WHERE session_id = %s AND owner_id = %s",
                    (session_id, resolved_owner_id),
                )
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = %s AND owner_id = %s",
                    (session_id, resolved_owner_id),
                )
                deleted = cursor.rowcount > 0
        return deleted

    def summary(self, session_id: str) -> str:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT summary FROM sessions WHERE session_id = %s", (session_id,))
                row = cursor.fetchone()
        return str(row[0]) if row else ""

    def structured_summary(self, session_id: str) -> dict:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT summary_json FROM sessions WHERE session_id = %s", (session_id,))
                row = cursor.fetchone()
        if not row:
            return DEFAULT_SUMMARY.copy()
        try:
            return merge_defaults(DEFAULT_SUMMARY, json.loads(row[0] or "{}"))
        except json.JSONDecodeError:
            return DEFAULT_SUMMARY.copy()

    def topic_state(self, session_id: str) -> dict:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT topic_state_json FROM sessions WHERE session_id = %s", (session_id,))
                row = cursor.fetchone()
        if not row:
            return DEFAULT_TOPIC_STATE.copy()
        try:
            return merge_defaults(DEFAULT_TOPIC_STATE, json.loads(row[0] or "{}"))
        except json.JSONDecodeError:
            return DEFAULT_TOPIC_STATE.copy()

    def memory_snapshot(self, session_id: str) -> dict:
        return {
            "recent_turns": [turn.to_dict() for turn in self.recent_turns(session_id)],
            "session_summary": self.structured_summary(session_id),
            "topic_state": self.topic_state(session_id),
        }

    def export_session(self, session_id: str, owner_id: str | None = None) -> dict:
        resolved_owner_id = owner_id or self._owner_from_session_id(session_id)
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT session_id FROM sessions WHERE session_id = %s AND owner_id = %s",
                    (session_id, resolved_owner_id),
                )
                exists = cursor.fetchone()
        if not exists:
            return {}
        turns = self.all_turns(session_id)
        return {
            "session_id": session_id,
            "summary": self.summary(session_id),
            "memory": self.memory_snapshot(session_id),
            "topics": self.list_topics(session_id),
            "turns": [turn.to_dict() for turn in turns],
        }

    def last_turn(self, session_id: str) -> ChatTurn | None:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT role, content, metadata
                    FROM session_turns
                    WHERE session_id = %s
                    ORDER BY turn_id DESC
                    LIMIT 1
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return _row_to_turn(row)

    def pending_user_message(self, session_id: str, owner_id: str | None = None) -> str | None:  # noqa: ARG002
        last_turn = self.last_turn(session_id)
        if last_turn and last_turn.role == "user":
            return last_turn.content
        return None

    def all_turns(self, session_id: str) -> list[ChatTurn]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT role, content, metadata
                    FROM session_turns
                    WHERE session_id = %s
                    ORDER BY turn_id ASC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()
        return [_row_to_turn(row) for row in rows]

    def list_sessions(self, limit: int = 50, session_prefix: str | None = None, owner_id: str | None = None) -> list[dict]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                if owner_id:
                    cursor.execute(
                        _SESSION_SELECT + "WHERE s.owner_id = %s ORDER BY s.updated_at DESC LIMIT %s",
                        (owner_id, limit),
                    )
                elif session_prefix:
                    cursor.execute(
                        _SESSION_SELECT + "WHERE s.session_id LIKE %s ORDER BY s.updated_at DESC LIMIT %s",
                        (f"{session_prefix}%", limit),
                    )
                else:
                    cursor.execute(
                        _SESSION_SELECT + "ORDER BY s.updated_at DESC LIMIT %s",
                        (limit,),
                    )
                rows = cursor.fetchall()

        sessions: list[dict] = []
        for row in rows:
            summary_json = merge_defaults(DEFAULT_SUMMARY, json.loads(row["summary_json"] or "{}"))
            title_source = row["last_user_message"] or row["first_user_message"] or summary_json["topic"] or "New chat"
            title = normalize_text(str(title_source))[:42] or "New chat"
            summary_text = normalize_text(str(row["summary"] or ""))[:96]
            sessions.append(
                {
                    "session_id": row["session_id"],
                    "title": title,
                    "summary": summary_text,
                    "updated_at": str(row["updated_at"]),
                    "turn_count": int(row["turn_count"] or 0),
                    "last_user_message": normalize_text(str(row["last_user_message"] or ""))[:80],
                    "last_user_at": str(row["last_user_at"] or ""),
                }
            )
        return sessions

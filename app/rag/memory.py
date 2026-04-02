from __future__ import annotations

from contextlib import contextmanager
import json
from uuid import uuid4

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.session.state import (
    DEFAULT_SUMMARY,
    DEFAULT_TOPIC_STATE,
    DEFAULT_TOPIC_THREAD_SUMMARY,
    build_rewrite_context_payload,
    build_summary_bundle,
    build_topic_thread_summary,
    deserialize_topic_row,
    extract_entities,
    merge_defaults,
)
from app.session.store_sql import persist_summary
from app.rag.types import ChatTurn
from app.rag.utils import normalize_text


class SessionStore:
    """LLM 히스토리에 의존하지 않는 애플리케이션 수준 세션 메모리.

    PostgreSQL에 대화 턴, 구조화된 요약, 토픽 상태를 저장하고,
    매 턴마다 요약과 토픽을 자동 갱신한다. 질의 재작성(query rewrite)은
    대화 맥락과 엔티티를 사용하여 모호한 후속 질문을 보강한다.
    """

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
        """커넥션 풀을 닫는다."""
        self._pool.closeall()

    @staticmethod
    def _owner_from_session_id(session_id: str) -> str:
        if "::" in session_id:
            return session_id.split("::", 1)[0]
        return "legacy"

    def _initialize(self) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL DEFAULT 'legacy',
                        summary TEXT NOT NULL DEFAULT '',
                        summary_json TEXT NOT NULL DEFAULT '{}',
                        topic_state_json TEXT NOT NULL DEFAULT '{}',
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_turns (
                        turn_id SERIAL PRIMARY KEY,
                        owner_id TEXT NOT NULL DEFAULT 'legacy',
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_turns_session_id
                    ON session_turns(session_id, turn_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_topics (
                        topic_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL DEFAULT 'legacy',
                        session_id TEXT NOT NULL,
                        topic_label TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'active',
                        summary_json TEXT NOT NULL DEFAULT '{}',
                        source_state_json TEXT NOT NULL DEFAULT '{}',
                        entity_state_json TEXT NOT NULL DEFAULT '{}',
                        open_questions_json TEXT NOT NULL DEFAULT '[]',
                        resolved_facts_json TEXT NOT NULL DEFAULT '[]',
                        last_user_focus TEXT NOT NULL DEFAULT '',
                        last_retrieval_mode TEXT NOT NULL DEFAULT '',
                        turn_count INTEGER NOT NULL DEFAULT 0,
                        last_active_turn_id INTEGER,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_topics_session_id
                    ON session_topics(session_id, updated_at DESC)
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS turn_topic_links (
                        turn_id INTEGER NOT NULL,
                        owner_id TEXT NOT NULL DEFAULT 'legacy',
                        session_id TEXT NOT NULL,
                        topic_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        link_type TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 0.0,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (turn_id, topic_id),
                        FOREIGN KEY (topic_id) REFERENCES session_topics(topic_id),
                        FOREIGN KEY (turn_id) REFERENCES session_turns(turn_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_turn_topic_links_session_id
                    ON turn_topic_links(session_id, topic_id, turn_id)
                    """
                )

                # 마이그레이션: 기존 테이블에 누락된 컬럼 추가
                cursor.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'sessions'
                    """
                )
                session_columns = {row[0] for row in cursor.fetchall()}
                if "summary_json" not in session_columns:
                    cursor.execute(
                        "ALTER TABLE sessions ADD COLUMN summary_json TEXT NOT NULL DEFAULT '{}'"
                    )
                if "topic_state_json" not in session_columns:
                    cursor.execute(
                        "ALTER TABLE sessions ADD COLUMN topic_state_json TEXT NOT NULL DEFAULT '{}'"
                    )
                if "owner_id" not in session_columns:
                    cursor.execute(
                        "ALTER TABLE sessions ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'"
                    )
                cursor.execute(
                    """
                    UPDATE sessions
                    SET owner_id = CASE
                        WHEN position('::' in session_id) > 0 THEN split_part(session_id, '::', 1)
                        ELSE 'legacy'
                    END
                    WHERE owner_id = 'legacy'
                    """
                )

                cursor.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'session_turns'
                    """
                )
                turn_columns = {row[0] for row in cursor.fetchall()}
                if "metadata" not in turn_columns:
                    cursor.execute(
                        "ALTER TABLE session_turns ADD COLUMN metadata TEXT NOT NULL DEFAULT ''"
                    )
                if "owner_id" not in turn_columns:
                    cursor.execute(
                        "ALTER TABLE session_turns ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'"
                    )
                cursor.execute(
                    """
                    UPDATE session_turns
                    SET owner_id = CASE
                        WHEN position('::' in session_id) > 0 THEN split_part(session_id, '::', 1)
                        ELSE 'legacy'
                    END
                    WHERE owner_id = 'legacy'
                    """
                )
                cursor.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'session_topics'
                    """
                )
                topic_columns = {row[0] for row in cursor.fetchall()}
                topic_defaults = {
                    "summary_json": "TEXT NOT NULL DEFAULT '{}'",
                    "source_state_json": "TEXT NOT NULL DEFAULT '{}'",
                    "entity_state_json": "TEXT NOT NULL DEFAULT '{}'",
                    "open_questions_json": "TEXT NOT NULL DEFAULT '[]'",
                    "resolved_facts_json": "TEXT NOT NULL DEFAULT '[]'",
                    "last_user_focus": "TEXT NOT NULL DEFAULT ''",
                    "last_retrieval_mode": "TEXT NOT NULL DEFAULT ''",
                    "turn_count": "INTEGER NOT NULL DEFAULT 0",
                    "last_active_turn_id": "INTEGER",
                }
                for column_name, column_def in topic_defaults.items():
                    if column_name not in topic_columns:
                        cursor.execute(
                            f"ALTER TABLE session_topics ADD COLUMN {column_name} {column_def}"
                        )
                if "owner_id" not in topic_columns:
                    cursor.execute(
                        "ALTER TABLE session_topics ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'"
                    )
                cursor.execute(
                    """
                    UPDATE session_topics
                    SET owner_id = CASE
                        WHEN position('::' in session_id) > 0 THEN split_part(session_id, '::', 1)
                        ELSE 'legacy'
                    END
                    WHERE owner_id = 'legacy'
                    """
                )
                cursor.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'turn_topic_links'
                    """
                )
                turn_topic_columns = {row[0] for row in cursor.fetchall()}
                if "owner_id" not in turn_topic_columns:
                    cursor.execute(
                        "ALTER TABLE turn_topic_links ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'"
                    )
                cursor.execute(
                    """
                    UPDATE turn_topic_links
                    SET owner_id = CASE
                        WHEN position('::' in session_id) > 0 THEN split_part(session_id, '::', 1)
                        ELSE 'legacy'
                    END
                    WHERE owner_id = 'legacy'
                    """
                )

    def _ensure_session_row(self, cursor, session_id: str) -> None:
        owner_id = self._owner_from_session_id(session_id)
        cursor.execute(
            """
            INSERT INTO sessions (session_id, owner_id, summary, summary_json, topic_state_json, updated_at)
            VALUES (%s, %s, '', '{}', '{}', CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET owner_id = EXCLUDED.owner_id, updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, owner_id),
        )

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> int:
        owner_id = self._owner_from_session_id(session_id)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                self._ensure_session_row(cursor, session_id)
                cursor.execute(
                    """
                    INSERT INTO session_turns (owner_id, session_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING turn_id
                    """,
                    (owner_id, session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
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
                turn_count = cursor.rowcount
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = %s AND owner_id = %s",
                    (session_id, resolved_owner_id),
                )
                session_count = cursor.rowcount
        return bool(turn_count or session_count)

    def recent_turns(self, session_id: str) -> list[ChatTurn]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT role, content, metadata
                    FROM session_turns
                    WHERE session_id = %s
                    ORDER BY turn_id DESC
                    LIMIT %s
                    """,
                    (session_id, self.memory_window_turns),
                )
                rows = cursor.fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in reversed(rows)
        ]

    def summary(self, session_id: str) -> str:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT summary FROM sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
        return row["summary"] if row else ""

    def structured_summary(self, session_id: str) -> dict:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT summary_json FROM sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
        if not row:
            return DEFAULT_SUMMARY.copy()
        return merge_defaults(DEFAULT_SUMMARY, json.loads(row["summary_json"] or "{}"))

    def topic_state(self, session_id: str) -> dict:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT topic_state_json FROM sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
        if not row:
            return DEFAULT_TOPIC_STATE.copy()
        return merge_defaults(DEFAULT_TOPIC_STATE, json.loads(row["topic_state_json"] or "{}"))

    def memory_snapshot(self, session_id: str) -> dict:
        return {
            "recent_turns": [turn.to_dict() for turn in self.recent_turns(session_id)],
            "session_summary": self.structured_summary(session_id),
            "topic_state": self.topic_state(session_id),
        }

    def create_topic(self, session_id: str, seed_label: str, seed_turn_id: int | None = None) -> dict:
        topic_id = f"topic_{uuid4().hex}"
        owner_id = self._owner_from_session_id(session_id)
        topic_label = normalize_text(seed_label)[:120] or "Untitled topic"
        topic_summary = DEFAULT_TOPIC_THREAD_SUMMARY.copy()
        topic_summary["topic_label"] = topic_label
        with self._connection() as connection:
            with connection.cursor() as cursor:
                self._ensure_session_row(cursor, session_id)
                cursor.execute(
                    """
                    INSERT INTO session_topics (
                        topic_id, owner_id, session_id, topic_label, status, summary_json,
                        source_state_json, entity_state_json, open_questions_json,
                        resolved_facts_json, last_user_focus, last_retrieval_mode,
                        turn_count, last_active_turn_id, updated_at
                    )
                    VALUES (%s, %s, %s, %s, 'active', %s, '{}', '{}', '[]', '[]', '', '', 0, %s, CURRENT_TIMESTAMP)
                    """,
                    (
                        topic_id,
                        owner_id,
                        session_id,
                        topic_label,
                        json.dumps(topic_summary, ensure_ascii=False),
                        seed_turn_id,
                    ),
                )
                self._update_session_topic_meta(
                    cursor,
                    session_id,
                    last_active_topic_id=topic_id,
                    known_topic_id=topic_id,
                )
        return self.get_topic(topic_id) or {
            "topic_id": topic_id,
            "session_id": session_id,
            "topic_label": topic_label,
            "status": "active",
        }

    def list_topics(self, session_id: str) -> list[dict]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT topic_id, session_id, topic_label, status, summary_json,
                           source_state_json, entity_state_json, open_questions_json,
                           resolved_facts_json, last_user_focus, last_retrieval_mode,
                           turn_count, last_active_turn_id, created_at, updated_at
                    FROM session_topics
                    WHERE session_id = %s
                    ORDER BY updated_at DESC, created_at DESC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()
        return [deserialize_topic_row(row) for row in rows]

    def get_topic(self, topic_id: str) -> dict | None:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT topic_id, session_id, topic_label, status, summary_json,
                           source_state_json, entity_state_json, open_questions_json,
                           resolved_facts_json, last_user_focus, last_retrieval_mode,
                           turn_count, last_active_turn_id, created_at, updated_at
                    FROM session_topics
                    WHERE topic_id = %s
                    """,
                    (topic_id,),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return deserialize_topic_row(row)

    def get_last_active_topic(self, session_id: str) -> dict | None:
        topic_state = self.topic_state(session_id)
        last_active_topic_id = str(topic_state.get("last_active_topic_id") or "")
        if last_active_topic_id:
            topic = self.get_topic(last_active_topic_id)
            if topic is not None:
                return topic
        topics = self.list_topics(session_id)
        return topics[0] if topics else None

    def link_turn_to_topic(
        self,
        turn_id: int,
        session_id: str,
        topic_id: str,
        role: str,
        link_type: str,
        confidence: float,
    ) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                self._ensure_session_row(cursor, session_id)
                cursor.execute(
                    """
                    INSERT INTO turn_topic_links (
                        turn_id, owner_id, session_id, topic_id, role, link_type, confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (turn_id, topic_id) DO UPDATE
                    SET owner_id = EXCLUDED.owner_id,
                        role = EXCLUDED.role,
                        link_type = EXCLUDED.link_type,
                        confidence = EXCLUDED.confidence
                    """,
                    (turn_id, self._owner_from_session_id(session_id), session_id, topic_id, role, link_type, confidence),
                )
                cursor.execute(
                    """
                    UPDATE session_topics
                    SET last_active_turn_id = %s,
                        turn_count = (
                            SELECT COUNT(*)
                            FROM turn_topic_links
                            WHERE topic_id = %s
                        ),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE topic_id = %s
                    """,
                    (turn_id, topic_id, topic_id),
                )
                self._update_session_topic_meta(
                    cursor,
                    session_id,
                    last_active_topic_id=topic_id,
                    known_topic_id=topic_id,
                )
        self.refresh_topic_memory(session_id, topic_id)

    def recent_topic_turns(self, session_id: str, topic_id: str, limit: int | None = None) -> list[ChatTurn]:
        query_limit = limit if limit is not None else self.memory_window_turns
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT role, content, metadata FROM (
                        SELECT st.role, st.content, st.metadata, st.turn_id
                        FROM turn_topic_links ttl
                        JOIN session_turns st ON st.turn_id = ttl.turn_id
                        WHERE ttl.session_id = %s AND ttl.topic_id = %s
                        ORDER BY st.turn_id DESC
                        LIMIT %s
                    ) sub
                    ORDER BY turn_id ASC
                    """,
                    (session_id, topic_id, query_limit),
                )
                rows = cursor.fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def topic_memory_snapshot(self, session_id: str, topic_id: str) -> dict:
        topic = self.get_topic(topic_id)
        if topic is None or topic.get("session_id") != session_id:
            return DEFAULT_TOPIC_THREAD_SUMMARY.copy()
        snapshot = DEFAULT_TOPIC_THREAD_SUMMARY.copy()
        snapshot.update(topic.get("summary", {}))
        snapshot["sources"] = topic.get("sources", [])
        snapshot["entities"] = topic.get("entities", [])
        snapshot["open_questions"] = topic.get("open_questions", [])
        snapshot["resolved_facts"] = topic.get("resolved_facts", [])
        snapshot["last_user_focus"] = topic.get("last_user_focus", "")
        snapshot["last_retrieval_mode"] = topic.get("last_retrieval_mode", "")
        snapshot["turn_count"] = int(topic.get("turn_count", 0))
        return snapshot

    def refresh_topic_memory(self, session_id: str, topic_id: str) -> None:
        turns = self.recent_topic_turns(session_id, topic_id, limit=10)
        topic = self.get_topic(topic_id)
        if topic is None or topic.get("session_id") != session_id:
            return
        summary_json = build_topic_thread_summary(
            topic_label=str(topic.get("topic_label") or ""),
            turns=turns,
        )
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE session_topics
                    SET summary_json = %s,
                        source_state_json = %s,
                        entity_state_json = %s,
                        open_questions_json = %s,
                        resolved_facts_json = %s,
                        last_user_focus = %s,
                        last_retrieval_mode = %s,
                        turn_count = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE topic_id = %s AND session_id = %s
                    """,
                    (
                        json.dumps(summary_json, ensure_ascii=False),
                        json.dumps({"sources": summary_json["sources"]}, ensure_ascii=False),
                        json.dumps({"entities": summary_json["entities"]}, ensure_ascii=False),
                        json.dumps(summary_json["open_questions"], ensure_ascii=False),
                        json.dumps(summary_json["resolved_facts"], ensure_ascii=False),
                        summary_json["last_user_focus"],
                        summary_json["last_retrieval_mode"],
                        summary_json["turn_count"],
                        topic_id,
                        session_id,
                    ),
                )

    def build_rewrite_context(self, session_id: str, user_message: str) -> dict | None:
        """LLM 질의 재작성에 필요한 대화 맥락을 구성한다.

        대화 이력이 없으면 None을 반환하여 재작성이 불필요함을 알린다.
        반환된 dict는 LLM 프롬프트 구성에 사용되며, 하드코딩된 규칙 대신
        LLM이 맥락을 판단하여 질의를 재작성한다.
        """
        recent = self.recent_turns(session_id)
        summary = self.structured_summary(session_id)
        topic_state = self.topic_state(session_id)
        return build_rewrite_context_payload(recent, summary, topic_state)

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
        return ChatTurn(
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def pending_user_message(self, session_id: str, owner_id: str | None = None) -> str | None:
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
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def list_sessions(
        self,
        limit: int = 50,
        session_prefix: str | None = None,
        owner_id: str | None = None,
    ) -> list[dict]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                if owner_id:
                    cursor.execute(
                        """
                        SELECT s.session_id, s.summary, s.summary_json, s.updated_at,
                               (
                                   SELECT content
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id ASC
                                   LIMIT 1
                               ) AS first_user_message,
                               (
                                   SELECT content
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id DESC
                                   LIMIT 1
                               ) AS last_user_message,
                               (
                                   SELECT created_at
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id DESC
                                   LIMIT 1
                               ) AS last_user_at,
                               (
                                   SELECT COUNT(*)
                                   FROM session_turns
                                   WHERE session_id = s.session_id
                               ) AS turn_count
                        FROM sessions s
                        WHERE s.owner_id = %s
                        ORDER BY s.updated_at DESC
                        LIMIT %s
                        """,
                        (owner_id, limit),
                    )
                elif session_prefix:
                    cursor.execute(
                        """
                        SELECT s.session_id, s.summary, s.summary_json, s.updated_at,
                               (
                                   SELECT content
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id ASC
                                   LIMIT 1
                               ) AS first_user_message,
                               (
                                   SELECT content
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id DESC
                                   LIMIT 1
                               ) AS last_user_message,
                               (
                                   SELECT created_at
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id DESC
                                   LIMIT 1
                               ) AS last_user_at,
                               (
                                   SELECT COUNT(*)
                                   FROM session_turns
                                   WHERE session_id = s.session_id
                               ) AS turn_count
                        FROM sessions s
                        WHERE s.session_id LIKE %s
                        ORDER BY s.updated_at DESC
                        LIMIT %s
                        """,
                        (f"{session_prefix}%", limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT s.session_id, s.summary, s.summary_json, s.updated_at,
                               (
                                   SELECT content
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id ASC
                                   LIMIT 1
                               ) AS first_user_message,
                               (
                                   SELECT content
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id DESC
                                   LIMIT 1
                               ) AS last_user_message,
                               (
                                   SELECT created_at
                                   FROM session_turns
                                   WHERE session_id = s.session_id AND role = 'user'
                                   ORDER BY turn_id DESC
                                   LIMIT 1
                               ) AS last_user_at,
                               (
                                   SELECT COUNT(*)
                                   FROM session_turns
                                   WHERE session_id = s.session_id
                               ) AS turn_count
                        FROM sessions s
                        ORDER BY s.updated_at DESC
                        LIMIT %s
                        """,
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

    def _recent_turns_for_refresh(self, session_id: str, limit: int = 10) -> list[ChatTurn]:
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
                    (session_id, limit),
                )
                rows = cursor.fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def _refresh_summary(self, session_id: str) -> None:
        turns = self._recent_turns_for_refresh(session_id, limit=10)
        summary_json, topic_state, summary = build_summary_bundle(turns)

        with self._connection() as connection:
            with connection.cursor() as cursor:
                persist_summary(cursor, session_id, summary, summary_json, topic_state)

    def _extract_entities(self, text: str) -> list[str]:
        return extract_entities(text)

    def _update_session_topic_meta(
        self,
        cursor,
        session_id: str,
        last_active_topic_id: str,
        known_topic_id: str,
    ) -> None:
        cursor.execute(
            "SELECT topic_state_json FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cursor.fetchone()
        current_topic_state = json.loads(row[0] or "{}") if row else {}
        known_topic_ids = [
            str(topic_id)
            for topic_id in current_topic_state.get("known_topic_ids", [])
            if topic_id
        ]
        if known_topic_id and known_topic_id not in known_topic_ids:
            known_topic_ids.append(known_topic_id)
        current_topic_state["last_active_topic_id"] = last_active_topic_id
        current_topic_state["known_topic_ids"] = known_topic_ids
        cursor.execute(
            """
            UPDATE sessions
            SET topic_state_json = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s
            """,
            (json.dumps(current_topic_state, ensure_ascii=False), session_id),
        )

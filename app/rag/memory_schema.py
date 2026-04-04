from __future__ import annotations

import json
from uuid import uuid4

import psycopg2.extras

from app.session.state import DEFAULT_TOPIC_THREAD_SUMMARY, build_topic_thread_summary, deserialize_topic_row
from app.rag.types import ChatTurn
from app.rag.utils import normalize_text

_TOPIC_COLUMNS = """
    topic_id, session_id, topic_label, status, summary_json,
    source_state_json, entity_state_json, open_questions_json,
    resolved_facts_json, last_user_focus, last_retrieval_mode,
    turn_count, last_active_turn_id, created_at, updated_at
"""

_OWNER_ID_UPDATE_SQL = """
    UPDATE {table}
    SET owner_id = CASE
        WHEN position('::' in session_id) > 0 THEN split_part(session_id, '::', 1)
        ELSE 'legacy'
    END
    WHERE owner_id = 'legacy'
"""


def _get_columns(cursor, table_name: str) -> set[str]:
    cursor.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
        (table_name,),
    )
    return {row[0] for row in cursor.fetchall()}


def _ensure_owner_id_column(cursor, table_name: str) -> None:
    columns = _get_columns(cursor, table_name)
    if "owner_id" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'")
    cursor.execute(_OWNER_ID_UPDATE_SQL.format(table=table_name))


def initialize_session_store(cursor) -> None:
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

    session_columns = _get_columns(cursor, "sessions")
    if "summary_json" not in session_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN summary_json TEXT NOT NULL DEFAULT '{}'")
    if "topic_state_json" not in session_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN topic_state_json TEXT NOT NULL DEFAULT '{}'")
    _ensure_owner_id_column(cursor, "sessions")

    turn_columns = _get_columns(cursor, "session_turns")
    if "metadata" not in turn_columns:
        cursor.execute("ALTER TABLE session_turns ADD COLUMN metadata TEXT NOT NULL DEFAULT ''")
    _ensure_owner_id_column(cursor, "session_turns")

    topic_columns = _get_columns(cursor, "session_topics")
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
            cursor.execute(f"ALTER TABLE session_topics ADD COLUMN {column_name} {column_def}")
    _ensure_owner_id_column(cursor, "session_topics")

    _ensure_owner_id_column(cursor, "turn_topic_links")


# ---------------------------------------------------------------------------
# SessionStoreTopicsMixin — merged from memory_topics.py
# ---------------------------------------------------------------------------

class SessionStoreTopicsMixin:
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
                self._update_session_topic_meta(cursor, session_id, last_active_topic_id=topic_id, known_topic_id=topic_id)
        return self.get_topic(topic_id) or {"topic_id": topic_id, "session_id": session_id, "topic_label": topic_label, "status": "active"}

    def list_topics(self, session_id: str) -> list[dict]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT {_TOPIC_COLUMNS}
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
                    f"""
                    SELECT {_TOPIC_COLUMNS}
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

    def link_turn_to_topic(self, turn_id: int, session_id: str, topic_id: str, role: str, link_type: str, confidence: float) -> None:
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
                self._update_session_topic_meta(cursor, session_id, last_active_topic_id=topic_id, known_topic_id=topic_id)
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
            ChatTurn(role=row["role"], content=row["content"], metadata=json.loads(row["metadata"] or "{}"))
            for row in rows
        ]

    def topic_memory_snapshot(self, session_id: str, topic_id: str) -> dict:
        topic = self.get_topic(topic_id)
        if topic is None or topic.get("session_id") != session_id:
            return DEFAULT_TOPIC_THREAD_SUMMARY.copy()
        snapshot = DEFAULT_TOPIC_THREAD_SUMMARY.copy()
        snapshot.update(topic.get("summary", {}))
        for field, default in [
            ("sources", []), ("entities", []), ("open_questions", []), ("resolved_facts", []),
            ("last_user_focus", ""), ("last_retrieval_mode", ""),
        ]:
            snapshot[field] = topic.get(field, default)
        snapshot["turn_count"] = int(topic.get("turn_count", 0))
        return snapshot

    def refresh_topic_memory(self, session_id: str, topic_id: str) -> None:
        turns = self.recent_topic_turns(session_id, topic_id, limit=10)
        topic = self.get_topic(topic_id)
        if topic is None or topic.get("session_id") != session_id:
            return
        summary_json = build_topic_thread_summary(topic_label=str(topic.get("topic_label") or ""), turns=turns)
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

    def _update_session_topic_meta(self, cursor, session_id: str, last_active_topic_id: str, known_topic_id: str) -> None:
        cursor.execute("SELECT topic_state_json FROM sessions WHERE session_id = %s", (session_id,))
        row = cursor.fetchone()
        current_topic_state = json.loads(row[0] or "{}") if row else {}
        known_topic_ids = [str(topic_id) for topic_id in current_topic_state.get("known_topic_ids", []) if topic_id]
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

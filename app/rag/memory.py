from __future__ import annotations

from contextlib import contextmanager
import json
from uuid import uuid4

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.rag.types import ChatTurn
from app.rag.utils import normalize_text


DEFAULT_SUMMARY = {
    "topic": "",
    "user_goal": "",
    "recent_documents": [],
    "recent_pages": [],
    "unresolved_questions": [],
    "last_user_message": "",
}

DEFAULT_TOPIC_STATE = {
    "active_topic": "",
    "active_entities": [],
    "selected_sources": [],
    "selected_pages": [],
    "last_retrieval_mode": "",
    "last_answer_citations": [],
    "last_user_focus": "",
    "recent_user_topics": [],
    "last_explicit_resource": "",
    "last_explicit_resources": [],
    "last_intent": "",
    "last_response_shape": "",
    "last_answer_route": "",
    "last_format_constraints": [],
    "last_code_resource_kind": "",
    "last_grounded_chunk_ids": [],
    "last_grounded_section_paths": [],
    "last_example_source_pages": [],
    "last_example_anchor": {},
    "procedure_state": {},
}

DEFAULT_TOPIC_THREAD_SUMMARY = {
    "topic_label": "",
    "summary": "",
    "entities": [],
    "sources": [],
    "important_pages": [],
    "open_questions": [],
    "resolved_facts": [],
    "last_user_focus": "",
    "last_retrieval_mode": "",
    "last_explicit_resource": "",
    "last_explicit_resources": [],
    "last_intent": "",
    "last_response_shape": "",
    "last_answer_route": "",
    "last_format_constraints": [],
    "last_code_resource_kind": "",
    "last_grounded_chunk_ids": [],
    "last_grounded_section_paths": [],
    "last_example_source_pages": [],
    "last_example_anchor": {},
    "turn_count": 0,
}

_GENERIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "may", "might", "shall", "should",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "my", "your", "his", "its", "our", "their",
    "what", "when", "where", "which", "who", "whom", "how", "why",
    "that", "this", "these", "those", "there", "here",
    "and", "or", "but", "if", "then", "so", "because", "as", "than",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "not", "no", "yes", "all", "some", "any", "each", "every",
    "more", "most", "very", "too", "also", "just", "only",
    "again", "above", "below", "up", "down", "out", "off",
    "explain", "show", "tell", "give", "get", "make", "let",
}


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
        return self._merge_defaults(DEFAULT_SUMMARY, json.loads(row["summary_json"] or "{}"))

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
        return self._merge_defaults(DEFAULT_TOPIC_STATE, json.loads(row["topic_state_json"] or "{}"))

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
        return [self._deserialize_topic_row(row) for row in rows]

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
        return self._deserialize_topic_row(row)

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
        summary_json = self._build_topic_thread_summary(
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
        if not recent:
            return None

        summary = self.structured_summary(session_id)
        topic_state = self.topic_state(session_id)

        # 최근 대화 턴을 간결하게 정리
        conversation_history: list[dict] = []
        last_assistant_turn: ChatTurn | None = None
        for turn in recent[-4:]:
            entry: dict = {"role": turn.role, "content": turn.content[:200]}
            if turn.role == "assistant" and turn.metadata:
                sources = [
                    s.get("file_name", "")
                    for s in turn.metadata.get("source_grounding", [])[:2]
                    if s.get("file_name")
                ]
                if sources:
                    entry["sources"] = sources
                last_assistant_turn = turn
            conversation_history.append(entry)

        # 마지막 assistant 응답의 포맷/형태 정보 추출 (LLM 재작성에 활용)
        last_response_shape = ""
        last_response_intent = ""
        if last_assistant_turn and last_assistant_turn.metadata:
            qi = last_assistant_turn.metadata.get("query_interpretation") or {}
            last_response_shape = str(qi.get("response_shape") or "")
            last_response_intent = str(qi.get("intent") or "")

        return {
            "conversation_history": conversation_history,
            "active_topic": str(topic_state.get("active_topic") or summary.get("topic") or ""),
            "active_entities": [str(e) for e in topic_state.get("active_entities", []) if e][:6],
            "selected_sources": [str(s) for s in topic_state.get("selected_sources", []) if s][:3],
            "selected_pages": topic_state.get("selected_pages", [])[:5],
            "last_retrieval_mode": str(topic_state.get("last_retrieval_mode") or ""),
            "last_response_shape": last_response_shape,
            "last_response_intent": last_response_intent,
            "last_explicit_resources": [str(v) for v in topic_state.get("last_explicit_resources", []) if v][:4],
            "last_code_resource_kind": str(topic_state.get("last_code_resource_kind") or ""),
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
            summary_json = self._merge_defaults(DEFAULT_SUMMARY, json.loads(row["summary_json"] or "{}"))
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
        summary_json = self._build_structured_summary(turns)
        topic_state = self._build_topic_state(turns)
        summary = self._stringify_summary(summary_json, topic_state)

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (session_id, summary, summary_json, topic_state_json, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE
                    SET summary = EXCLUDED.summary,
                        summary_json = EXCLUDED.summary_json,
                        topic_state_json = EXCLUDED.topic_state_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        session_id,
                        summary,
                        json.dumps(summary_json, ensure_ascii=False),
                        json.dumps(topic_state, ensure_ascii=False),
                    ),
                )

    def _build_structured_summary(self, turns: list[ChatTurn]) -> dict:
        recent_turns = turns[-10:]
        user_turns = [turn for turn in recent_turns if turn.role == "user"]
        assistant_turns = [turn for turn in recent_turns if turn.role == "assistant"]

        recent_documents: list[str] = []
        recent_pages: list[int] = []
        unresolved_questions: list[str] = []
        for turn in assistant_turns[-3:]:
            metadata = turn.metadata or {}
            for source in metadata.get("source_grounding", []):
                file_name = normalize_text(str(source.get("file_name") or ""))
                if file_name and file_name not in recent_documents:
                    recent_documents.append(file_name)
            for page in metadata.get("grounded_pages", []):
                page_number = int(page.get("page_number") or 0)
                if page_number and page_number not in recent_pages:
                    recent_pages.append(page_number)
            if metadata.get("mode") == "general" and turn.content:
                unresolved_questions.append(normalize_text(turn.content[:120]))

        last_user_message = normalize_text(user_turns[-1].content) if user_turns else ""
        user_goal = normalize_text(user_turns[-1].content) if user_turns else ""
        topic = recent_documents[0] if recent_documents else self._extract_topic_from_turns(user_turns)

        return {
            "topic": topic,
            "user_goal": user_goal,
            "recent_documents": recent_documents[:3],
            "recent_pages": recent_pages[:5],
            "unresolved_questions": unresolved_questions[:2],
            "last_user_message": last_user_message,
        }

    def _build_topic_state(self, turns: list[ChatTurn]) -> dict:
        recent_turns = turns[-8:]
        selected_sources: list[str] = []
        selected_pages: list[int] = []
        last_answer_citations: list[dict] = []
        last_retrieval_mode = ""
        active_entities: list[str] = []
        recent_user_topics: list[str] = []
        last_user_focus = ""
        procedure_state: dict = {}
        last_explicit_resource = ""
        last_explicit_resources: list[str] = []
        last_intent = ""
        last_response_shape = ""
        last_answer_route = ""
        last_format_constraints: list[str] = []
        last_code_resource_kind = ""
        last_grounded_chunk_ids: list[str] = []
        last_grounded_section_paths: list[str] = []
        last_example_source_pages: list[int] = []
        last_example_anchor: dict = {}

        for turn in recent_turns:
            metadata = turn.metadata or {}
            if turn.role == "assistant":
                last_retrieval_mode = str(metadata.get("mode") or last_retrieval_mode)
                query_interpretation = metadata.get("query_interpretation") or {}
                resources = [
                    normalize_text(str(value)).lower()
                    for value in query_interpretation.get("resources", []) or []
                    if value
                ]
                if resources:
                    last_explicit_resources = []
                    for resource in resources:
                        if resource and resource not in last_explicit_resources:
                            last_explicit_resources.append(resource)
                    last_explicit_resource = last_explicit_resources[0]
                last_intent = str(query_interpretation.get("intent") or last_intent)
                last_response_shape = str(query_interpretation.get("response_shape") or last_response_shape)
                last_answer_route = str(metadata.get("answer_route") or last_answer_route)
                formats = [
                    normalize_text(str(value)).lower()
                    for value in query_interpretation.get("format_constraints", []) or []
                    if value
                ]
                if formats:
                    last_format_constraints = []
                    for fmt in formats:
                        if fmt and fmt not in last_format_constraints:
                            last_format_constraints.append(fmt)
                for item in metadata.get("source_grounding", []):
                    file_name = normalize_text(str(item.get("file_name") or ""))
                    if file_name and file_name not in selected_sources:
                        selected_sources.append(file_name)
                for item in metadata.get("preview_pages", []):
                    page_number = int(item.get("page_number") or 0)
                    if page_number and page_number not in selected_pages:
                        selected_pages.append(page_number)
                    if page_number and page_number not in last_example_source_pages:
                        last_example_source_pages.append(page_number)
                if metadata.get("answer_citations"):
                    last_answer_citations = metadata["answer_citations"][:4]
                if isinstance(metadata.get("procedure_state"), dict) and metadata.get("procedure_state", {}).get("steps"):
                    procedure_state = metadata["procedure_state"]
                if isinstance(metadata.get("last_example_anchor"), dict) and metadata.get("last_example_anchor"):
                    last_example_anchor = metadata["last_example_anchor"]
                for item in metadata.get("items", []) or []:
                    chunk_id = normalize_text(str(item.get("chunk_id") or ""))
                    section_path = normalize_text(str(item.get("section_path") or ""))
                    page_number = int(item.get("page_number") or item.get("page_start") or 0)
                    if chunk_id and chunk_id not in last_grounded_chunk_ids:
                        last_grounded_chunk_ids.append(chunk_id)
                    if section_path and section_path not in last_grounded_section_paths:
                        last_grounded_section_paths.append(section_path)
                    if page_number and page_number not in last_example_source_pages:
                        last_example_source_pages.append(page_number)
                    if not last_code_resource_kind and str(item.get("code_subtype") or "") == "k8s_manifest":
                        code_signals = [
                            normalize_text(str(value)).lower()
                            for value in item.get("code_signals", []) or []
                            if value
                        ]
                        for signal in code_signals:
                            if signal in {
                                "configmap", "secret", "pod", "deployment", "service",
                                "persistentvolume", "persistentvolumeclaim", "route", "ingress", "storageclass",
                            }:
                                last_code_resource_kind = signal
                                break
                if not last_code_resource_kind and len(last_explicit_resources) == 1 and last_answer_route == "extractive_code":
                    last_code_resource_kind = last_explicit_resources[0]
            else:
                active_entities.extend(self._extract_entities(turn.content))
                focus = self._extract_focus_phrase(turn.content)
                if focus:
                    last_user_focus = focus
                    if focus not in recent_user_topics:
                        recent_user_topics.append(focus)

        active_topic = selected_sources[0] if selected_sources else self._extract_topic_from_turns(
            [turn for turn in recent_turns if turn.role == "user"]
        )
        deduped_entities: list[str] = []
        for entity in active_entities:
            if entity and entity not in deduped_entities:
                deduped_entities.append(entity)

        return {
            "active_topic": active_topic,
            "active_entities": deduped_entities[:6],
            "selected_sources": selected_sources[:3],
            "selected_pages": selected_pages[:5],
            "last_retrieval_mode": last_retrieval_mode,
            "last_answer_citations": last_answer_citations,
            "last_user_focus": last_user_focus,
            "recent_user_topics": recent_user_topics[-4:],
            "last_explicit_resource": last_explicit_resource,
            "last_explicit_resources": last_explicit_resources[:4],
            "last_intent": last_intent,
            "last_response_shape": last_response_shape,
            "last_answer_route": last_answer_route,
            "last_format_constraints": last_format_constraints[:4],
            "last_code_resource_kind": last_code_resource_kind,
            "last_grounded_chunk_ids": last_grounded_chunk_ids[:6],
            "last_grounded_section_paths": last_grounded_section_paths[:4],
            "last_example_source_pages": last_example_source_pages[:6],
            "last_example_anchor": last_example_anchor,
            "procedure_state": procedure_state,
        }

    def _stringify_summary(self, summary_json: dict, topic_state: dict) -> str:
        parts: list[str] = []
        if summary_json.get("topic"):
            parts.append(f"Topic: {summary_json['topic']}")
        if summary_json.get("user_goal"):
            parts.append(f"Goal: {normalize_text(str(summary_json['user_goal']))[:160]}")
        if summary_json.get("recent_documents"):
            parts.append("Docs: " + ", ".join(summary_json["recent_documents"][:3]))
        if summary_json.get("recent_pages"):
            parts.append("Pages: " + ", ".join(str(page) for page in summary_json["recent_pages"][:5]))
        if topic_state.get("active_entities"):
            parts.append("Entities: " + ", ".join(topic_state["active_entities"][:4]))
        if summary_json.get("unresolved_questions"):
            parts.append("Open: " + " | ".join(summary_json["unresolved_questions"][:2]))
        return normalize_text(" ; ".join(parts)[:700])

    def _build_topic_thread_summary(self, topic_label: str, turns: list[ChatTurn]) -> dict:
        structured_summary = self._build_structured_summary(turns)
        topic_state = self._build_topic_state(turns)
        resolved_facts: list[str] = []
        for turn in turns:
            if turn.role != "assistant":
                continue
            normalized = normalize_text(turn.content)
            if normalized and normalized not in resolved_facts:
                resolved_facts.append(normalized[:180])
        summary_text = self._stringify_summary(structured_summary, topic_state)
        return {
            "topic_label": topic_label or structured_summary.get("topic") or "",
            "summary": summary_text,
            "entities": topic_state.get("active_entities", [])[:6],
            "sources": structured_summary.get("recent_documents", [])[:3],
            "important_pages": structured_summary.get("recent_pages", [])[:5],
            "open_questions": structured_summary.get("unresolved_questions", [])[:3],
            "resolved_facts": resolved_facts[:5],
            "last_user_focus": topic_state.get("last_user_focus", ""),
            "last_retrieval_mode": topic_state.get("last_retrieval_mode", ""),
            "last_explicit_resource": topic_state.get("last_explicit_resource", ""),
            "last_explicit_resources": topic_state.get("last_explicit_resources", [])[:4],
            "last_intent": topic_state.get("last_intent", ""),
            "last_response_shape": topic_state.get("last_response_shape", ""),
            "last_answer_route": topic_state.get("last_answer_route", ""),
            "last_format_constraints": topic_state.get("last_format_constraints", [])[:4],
            "last_code_resource_kind": topic_state.get("last_code_resource_kind", ""),
            "last_grounded_chunk_ids": topic_state.get("last_grounded_chunk_ids", [])[:6],
            "last_grounded_section_paths": topic_state.get("last_grounded_section_paths", [])[:4],
            "last_example_source_pages": topic_state.get("last_example_source_pages", [])[:6],
            "last_example_anchor": topic_state.get("last_example_anchor", {}),
            "turn_count": len(turns),
        }

    def _extract_topic_from_turns(self, user_turns: list[ChatTurn]) -> str:
        for turn in reversed(user_turns):
            entities = self._extract_entities(turn.content)
            if entities:
                return entities[0]
            words = normalize_text(turn.content).split(" ")
            if words:
                return " ".join(words[:5])
        return ""

    def _extract_entities(self, text: str) -> list[str]:
        """텍스트에서 의미 있는 엔티티를 범용적으로 추출한다.

        하드코딩된 도메인 키워드 없이, 토큰의 형태적 특성만으로 판별한다:
        - 대문자로 시작하는 단어 (고유명사, 약어: PV, StorageClass)
        - 대소문자 혼합 단어 (camelCase: hostPath, configMap)
        - 한글이 포함된 토큰 (한국어 명사)
        - 숫자가 포함된 토큰 (버전, 식별자)
        - .pdf로 끝나는 파일명
        일반적인 영어 불용어(관사, 전치사, 대명사 등)는 제외한다.
        """
        normalized = normalize_text(text)
        if not normalized:
            return []
        tokens = [token.strip(".,:;()[]{}!?") for token in normalized.split(" ")]
        entities: list[str] = []

        for token in tokens:
            if len(token) < 2:
                continue
            lowered = token.lower()
            if lowered in _GENERIC_STOPWORDS:
                continue
            # 파일명
            if lowered.endswith(".pdf"):
                entities.append(token)
                continue
            # 숫자 포함 (버전, 식별자)
            if any(char.isdigit() for char in token):
                entities.append(token)
                continue
            # 대문자 시작 (고유명사, 약어: PV, Pod, StorageClass)
            if token[0].isupper():
                entities.append(token)
                continue
            # 대소문자 혼합 (camelCase: hostPath, configMap)
            if any(char.isalpha() for char in token) and any(char.isupper() for char in token[1:]):
                entities.append(token)
                continue
            # 한글 포함 토큰 (한국어 명사)
            if any("\uac00" <= char <= "\ud7a3" for char in token):
                entities.append(token)
                continue
        return entities[:8]

    def _extract_focus_phrase(self, text: str) -> str:
        """사용자 메시지에서 핵심 구문을 추출한다."""
        normalized = normalize_text(text)
        if not normalized:
            return ""
        entities = self._extract_entities(normalized)
        if entities:
            return normalize_text(" ".join(entities[:3]))
        words = [
            word
            for word in normalized.split(" ")
            if len(word) >= 2 and word.lower() not in _GENERIC_STOPWORDS
        ]
        return normalize_text(" ".join(words[:4]))

    def _merge_defaults(self, default: dict, loaded: dict) -> dict:
        merged = default.copy()
        merged.update(loaded or {})
        return merged

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

    def _deserialize_topic_row(self, row: dict) -> dict:
        summary = self._merge_defaults(
            DEFAULT_TOPIC_THREAD_SUMMARY,
            json.loads(row["summary_json"] or "{}"),
        )
        source_state = json.loads(row["source_state_json"] or "{}")
        entity_state = json.loads(row["entity_state_json"] or "{}")
        open_questions = json.loads(row["open_questions_json"] or "[]")
        resolved_facts = json.loads(row["resolved_facts_json"] or "[]")
        if source_state.get("sources"):
            summary["sources"] = [str(source) for source in source_state["sources"] if source][:3]
        if entity_state.get("entities"):
            summary["entities"] = [str(entity) for entity in entity_state["entities"] if entity][:6]
        summary["open_questions"] = [str(item) for item in open_questions if item][:3]
        summary["resolved_facts"] = [str(item) for item in resolved_facts if item][:5]
        summary["last_user_focus"] = str(row["last_user_focus"] or summary.get("last_user_focus") or "")
        summary["last_retrieval_mode"] = str(row["last_retrieval_mode"] or summary.get("last_retrieval_mode") or "")
        summary["turn_count"] = int(row["turn_count"] or summary.get("turn_count") or 0)
        return {
            "topic_id": row["topic_id"],
            "session_id": row["session_id"],
            "topic_label": row["topic_label"],
            "status": row["status"],
            "summary": summary,
            "sources": summary.get("sources", []),
            "entities": summary.get("entities", []),
            "open_questions": summary.get("open_questions", []),
            "resolved_facts": summary.get("resolved_facts", []),
            "last_user_focus": summary.get("last_user_focus", ""),
            "last_retrieval_mode": summary.get("last_retrieval_mode", ""),
            "turn_count": summary.get("turn_count", 0),
            "last_active_turn_id": row["last_active_turn_id"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

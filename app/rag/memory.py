from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.rag.types import ChatTurn
from app.rag.utils import normalize_text


class SessionStore:
    def __init__(self, db_path: Path, memory_window_turns: int = 6) -> None:
        self.db_path = db_path
        self.memory_window_turns = memory_window_turns
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS session_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_session_turns_session_id
                ON session_turns(session_id, turn_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(session_turns)").fetchall()
            }
            if "metadata" not in columns:
                connection.execute(
                    "ALTER TABLE session_turns ADD COLUMN metadata TEXT NOT NULL DEFAULT ''"
                )

    def add_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, summary, updated_at)
                VALUES (?, '', CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )
            connection.execute(
                "INSERT INTO session_turns (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
            )
        self._refresh_summary(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            turn_result = connection.execute(
                "DELETE FROM session_turns WHERE session_id = ?",
                (session_id,),
            )
            session_result = connection.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        return bool(turn_result.rowcount or session_result.rowcount)

    def recent_turns(self, session_id: str) -> list[ChatTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, metadata
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (session_id, self.memory_window_turns),
            ).fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in reversed(rows)
        ]

    def summary(self, session_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def rewrite_query(self, session_id: str, user_message: str) -> str:
        recent = self.recent_turns(session_id)
        if not recent:
            return user_message

        lowered = user_message.lower()
        ambiguous_markers = (
            "that",
            "it",
            "again",
            "previous",
            "그거",
            "그 부분",
            "이전",
            "다시",
        )
        if any(marker in lowered for marker in ambiguous_markers):
            prior_user_messages = [turn.content for turn in reversed(recent) if turn.role == "user"]
            if prior_user_messages:
                return f"{prior_user_messages[0]} / follow-up: {user_message}"
        return user_message

    def export_session(self, session_id: str) -> dict:
        turns = self.all_turns(session_id)
        return {
            "session_id": session_id,
            "summary": self.summary(session_id),
            "turns": [turn.to_dict() for turn in turns],
        }

    def last_turn(self, session_id: str) -> ChatTurn | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT role, content, metadata
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return ChatTurn(
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def pending_user_message(self, session_id: str) -> str | None:
        last_turn = self.last_turn(session_id)
        if last_turn and last_turn.role == "user":
            return last_turn.content
        return None

    def all_turns(self, session_id: str) -> list[ChatTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, metadata
                FROM session_turns
                WHERE session_id = ?
                ORDER BY turn_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.session_id, s.summary, s.updated_at,
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
                ORDER BY datetime(s.updated_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        sessions: list[dict] = []
        for row in rows:
            title_source = row["last_user_message"] or row["first_user_message"] or row["summary"] or "새 채팅"
            title = normalize_text(str(title_source))[:42] or "새 채팅"
            summary = normalize_text(str(row["summary"] or ""))[:96]
            sessions.append(
                {
                    "session_id": row["session_id"],
                    "title": title,
                    "summary": summary,
                    "updated_at": row["updated_at"],
                    "turn_count": int(row["turn_count"] or 0),
                    "last_user_message": normalize_text(str(row["last_user_message"] or ""))[:80],
                    "last_user_at": str(row["last_user_at"] or ""),
                }
            )
        return sessions

    def _refresh_summary(self, session_id: str) -> None:
        recent = self.all_turns(session_id)[-10:]
        user_points = [turn.content for turn in recent if turn.role == "user"][-3:]
        assistant_points = [turn.content for turn in recent if turn.role == "assistant"][-2:]
        summary = normalize_text(" | ".join(user_points + assistant_points)[:700])

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, summary, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE
                SET summary = excluded.summary, updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, summary),
            )

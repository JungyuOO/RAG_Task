from __future__ import annotations

from contextlib import closing
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path


class TaskRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        with closing(self._connect()) as connection:
            with connection:
                yield connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_task(self, task_id: str, task_type: str, payload: dict | None = None) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO tasks (task_id, task_type, status, payload_json, result_json, error, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, '{}', '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (task_id, task_type, json.dumps(payload or {}, ensure_ascii=False)),
            )

    def update_task(
        self,
        task_id: str,
        *,
        status: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET status = ?,
                    result_json = ?,
                    error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (
                    status,
                    json.dumps(result or {}, ensure_ascii=False),
                    error or "",
                    task_id,
                ),
            )

    def get_task(self, task_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT task_id, task_type, status, payload_json, result_json, error, created_at, updated_at
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "result": json.loads(row["result_json"] or "{}"),
            "error": row["error"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

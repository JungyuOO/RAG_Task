from __future__ import annotations

from contextlib import contextmanager
import json

import psycopg2
import psycopg2.extras


class TaskRepository:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._initialize()

    def _connect(self):
        connection = psycopg2.connect(self.dsn)
        connection.autocommit = False
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        result_json TEXT NOT NULL DEFAULT '{}',
                        error TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def create_task(self, task_id: str, task_type: str, payload: dict | None = None) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tasks (task_id, task_type, status, payload_json, result_json, error, created_at, updated_at)
                    VALUES (%s, %s, 'pending', %s, '{}', '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
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
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = %s,
                        result_json = %s,
                        error = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = %s
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
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT task_id, task_type, status, payload_json, result_json, error, created_at, updated_at
                    FROM tasks
                    WHERE task_id = %s
                    """,
                    (task_id,),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "result": json.loads(row["result_json"] or "{}"),
            "error": row["error"] or "",
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

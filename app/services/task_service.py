from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from app.repositories.task_repository import TaskRepository


class TaskService:
    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository

    def run_inline(self, task_type: str, payload: dict, runner: Callable[[], dict]) -> dict:
        task_id = str(uuid4())
        self.repository.create_task(task_id, task_type, payload=payload)
        self.repository.update_task(task_id, status="running")
        try:
            result = runner()
        except Exception as exc:
            self.repository.update_task(task_id, status="failed", error=str(exc))
            raise
        self.repository.update_task(task_id, status="completed", result=result)
        return {
            "task_id": task_id,
            "task_status": "completed",
            **result,
        }

    def get_task(self, task_id: str) -> dict | None:
        return self.repository.get_task(task_id)

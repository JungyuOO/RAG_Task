from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from apps.api.persistence.json_persistence import load_json_file, save_json_file
from apps.api.api.schemas.actions import (
    OcpActionExecutionListResponse,
    OcpActionExecutionRecord,
    OcpActionExecutionStatus,
    OcpActionPreviewResponse,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass(slots=True)
class _ExecutionModel:
    order: int
    execution_id: str
    request_id: str
    status: OcpActionExecutionStatus
    execution_mode: str
    simulated: bool
    preview: OcpActionPreviewResponse
    summary: str
    preflight_checks: list[str]
    output_lines: list[str]
    error: str
    created_at: datetime
    updated_at: datetime


class InMemoryActionExecutionRepository:
    def __init__(self, *, storage_path: Path | None = None) -> None:
        self._lock = Lock()
        self._counter = 0
        self._items: dict[str, _ExecutionModel] = {}
        self.storage_path = storage_path
        self._load()

    def create(
        self,
        *,
        request_id: str,
        status: OcpActionExecutionStatus,
        execution_mode: str,
        simulated: bool,
        preview: OcpActionPreviewResponse,
        summary: str,
        preflight_checks: list[str],
        output_lines: list[str],
        error: str = "",
    ) -> OcpActionExecutionRecord:
        with self._lock:
            self._counter += 1
            now = _utc_now()
            model = _ExecutionModel(
                order=self._counter,
                execution_id=f"exec-{uuid4().hex}",
                request_id=request_id,
                status=status,
                execution_mode=execution_mode,
                simulated=simulated,
                preview=preview,
                summary=summary,
                preflight_checks=list(preflight_checks),
                output_lines=list(output_lines),
                error=error,
                created_at=now,
                updated_at=now,
            )
            self._items[model.execution_id] = model
            self._save()
            return self._to_record(model)

    def list_recent(self, limit: int = 20) -> OcpActionExecutionListResponse:
        with self._lock:
            models = sorted(
                self._items.values(),
                key=lambda item: (item.updated_at, item.created_at, item.order),
                reverse=True,
            )
            return OcpActionExecutionListResponse(items=[self._to_record(model) for model in models[: max(limit, 1)]])

    def clear(self) -> None:
        with self._lock:
            self._counter = 0
            self._items.clear()
            self._save()

    @staticmethod
    def _to_record(model: _ExecutionModel) -> OcpActionExecutionRecord:
        return OcpActionExecutionRecord(
            execution_id=model.execution_id,
            request_id=model.request_id,
            status=model.status,
            execution_mode=model.execution_mode,
            simulated=model.simulated,
            preview=model.preview,
            summary=model.summary,
            preflight_checks=model.preflight_checks,
            output_lines=model.output_lines,
            error=model.error,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _load(self) -> None:
        if self.storage_path is None:
            return
        payload = load_json_file(self.storage_path, default={"counter": 0, "items": []})
        self._counter = int(payload.get("counter") or 0)
        self._items = {}
        for item in payload.get("items", []):
            model = _ExecutionModel(
                order=int(item["order"]),
                execution_id=item["execution_id"],
                request_id=item["request_id"],
                status=OcpActionExecutionStatus(item["status"]),
                execution_mode=item.get("execution_mode", "simulated"),
                simulated=bool(item.get("simulated", True)),
                preview=OcpActionPreviewResponse.model_validate(item["preview"]),
                summary=item.get("summary", ""),
                preflight_checks=list(item.get("preflight_checks") or []),
                output_lines=list(item.get("output_lines") or []),
                error=item.get("error", ""),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )
            self._items[model.execution_id] = model

    def _save(self) -> None:
        if self.storage_path is None:
            return
        save_json_file(
            self.storage_path,
            {
                "counter": self._counter,
                "items": [
                    {
                        "order": model.order,
                        "execution_id": model.execution_id,
                        "request_id": model.request_id,
                        "status": model.status.value,
                        "execution_mode": model.execution_mode,
                        "simulated": model.simulated,
                        "preview": model.preview.model_dump(mode="json"),
                        "summary": model.summary,
                        "preflight_checks": model.preflight_checks,
                        "output_lines": model.output_lines,
                        "error": model.error,
                        "created_at": model.created_at.isoformat(),
                        "updated_at": model.updated_at.isoformat(),
                    }
                    for model in self._items.values()
                ],
            },
        )


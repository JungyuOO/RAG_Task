from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from apps.api.persistence.json_persistence import load_json_file, save_json_file
from apps.api.api.schemas.indexing import BatchIndexRequest, BatchIndexResponse, BatchJobStatusResponse


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass(slots=True)
class _BatchJobRecord:
    order: int
    job_id: str
    status: str
    request: BatchIndexRequest
    result: BatchIndexResponse | None
    error: str
    progress_pct: int
    current_file: str
    step: str
    message: str
    recent_logs: list[str]
    created_at: datetime
    updated_at: datetime


class InMemoryBatchJobRepository:
    """Process-local job store for new batch reindex orchestration."""

    def __init__(self, *, storage_path: Path | None = None) -> None:
        self._lock = Lock()
        self._jobs: dict[str, _BatchJobRecord] = {}
        self._counter = 0
        self.storage_path = storage_path
        self._load()

    def create(self, request: BatchIndexRequest) -> BatchJobStatusResponse:
        with self._lock:
            now = _utc_now()
            self._counter += 1
            record = _BatchJobRecord(
                order=self._counter,
                job_id=f"batch-{uuid4().hex}",
                status="pending",
                request=request,
                result=None,
                error="",
                progress_pct=0,
                current_file="",
                step="pending",
                message="job queued",
                recent_logs=[],
                created_at=now,
                updated_at=now,
            )
            self._jobs[record.job_id] = record
            self._save()
            return self._to_response(record)

    def mark_running(self, job_id: str) -> BatchJobStatusResponse:
        with self._lock:
            record = self._require(job_id)
            record.status = "running"
            record.step = "RUNNING"
            record.message = "job started"
            record.updated_at = _utc_now()
            self._save()
            return self._to_response(record)

    def mark_progress(self, job_id: str, *, progress_pct: int, current_file: str) -> BatchJobStatusResponse:
        with self._lock:
            record = self._require(job_id)
            record.status = "running"
            record.progress_pct = max(0, min(100, int(progress_pct)))
            record.current_file = current_file
            if not record.step or record.step == "pending":
                record.step = "RUNNING"
            record.updated_at = _utc_now()
            self._save()
            return self._to_response(record)

    def mark_step(self, job_id: str, *, step: str, message: str = "", current_file: str | None = None) -> BatchJobStatusResponse:
        with self._lock:
            record = self._require(job_id)
            record.status = "running" if step not in {"COMPLETED", "FAILED", "CANCELLED"} else step.lower()
            record.step = step
            record.message = message
            if current_file is not None:
                record.current_file = current_file
            record.updated_at = _utc_now()
            self._save()
            return self._to_response(record)

    def append_log(self, job_id: str, message: str) -> BatchJobStatusResponse:
        with self._lock:
            record = self._require(job_id)
            text = str(message or "").strip()
            if text:
                record.recent_logs = [*record.recent_logs, text][-40:]
            record.updated_at = _utc_now()
            self._save()
            return self._to_response(record)

    def mark_completed(self, job_id: str, result: BatchIndexResponse) -> BatchJobStatusResponse:
        with self._lock:
            record = self._require(job_id)
            record.status = "completed"
            record.result = result
            record.error = ""
            record.progress_pct = 100
            record.current_file = ""
            record.step = "COMPLETED"
            record.message = "batch completed"
            record.updated_at = _utc_now()
            self._save()
            return self._to_response(record)

    def mark_failed(self, job_id: str, error: str) -> BatchJobStatusResponse:
        with self._lock:
            record = self._require(job_id)
            record.status = "failed"
            record.error = error
            record.current_file = ""
            record.step = "FAILED"
            record.message = error
            record.updated_at = _utc_now()
            self._save()
            return self._to_response(record)

    def mark_cancelled(self, job_id: str, result: BatchIndexResponse | None = None) -> BatchJobStatusResponse:
        with self._lock:
            record = self._require(job_id)
            if record.status in {"completed", "failed", "cancelled"}:
                if result is not None:
                    record.result = result
                    self._save()
                return self._to_response(record)
            record.status = "cancelled"
            if result is not None:
                record.result = result
            record.current_file = ""
            record.step = "CANCELLED"
            record.message = "job cancelled"
            record.updated_at = _utc_now()
            self._save()
            return self._to_response(record)

    def get(self, job_id: str) -> BatchJobStatusResponse | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return self._to_response(record) if record is not None else None

    def list_recent(self, limit: int = 20) -> list[BatchJobStatusResponse]:
        with self._lock:
            records = sorted(
                self._jobs.values(),
                key=lambda item: (item.updated_at, item.created_at, item.order),
                reverse=True,
            )
            return [self._to_response(record) for record in records[: max(int(limit), 1)]]

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._counter = 0
            self._save()

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            return bool(record and record.status == "cancelled")

    def _require(self, job_id: str) -> _BatchJobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise LookupError(f"Unknown batch job: {job_id}")
        return record

    def _load(self) -> None:
        if self.storage_path is None:
            return
        payload = load_json_file(self.storage_path, default={"counter": 0, "jobs": []})
        self._counter = int(payload.get("counter") or 0)
        self._jobs = {}
        for item in payload.get("jobs", []):
            record = _BatchJobRecord(
                order=int(item["order"]),
                job_id=item["job_id"],
                status=item["status"],
                request=BatchIndexRequest.model_validate(item["request"]),
                result=BatchIndexResponse.model_validate(item["result"]) if item.get("result") else None,
                error=item.get("error", ""),
                progress_pct=int(item.get("progress_pct") or 0),
                current_file=item.get("current_file", ""),
                step=item.get("step", item.get("status", "pending")),
                message=item.get("message", ""),
                recent_logs=list(item.get("recent_logs") or []),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )
            self._jobs[record.job_id] = record

    def _save(self) -> None:
        if self.storage_path is None:
            return
        save_json_file(
            self.storage_path,
            {
                "counter": self._counter,
                "jobs": [
                    {
                        "order": record.order,
                        "job_id": record.job_id,
                        "status": record.status,
                        "request": record.request.model_dump(mode="json"),
                        "result": record.result.model_dump(mode="json") if record.result else None,
                        "error": record.error,
                        "progress_pct": record.progress_pct,
                        "current_file": record.current_file,
                        "step": record.step,
                        "message": record.message,
                        "recent_logs": record.recent_logs,
                        "created_at": record.created_at.isoformat(),
                        "updated_at": record.updated_at.isoformat(),
                    }
                    for record in self._jobs.values()
                ],
            },
        )

    @staticmethod
    def _to_response(record: _BatchJobRecord) -> BatchJobStatusResponse:
        return BatchJobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            request=record.request,
            result=record.result,
            error=record.error,
            progress_pct=record.progress_pct,
            current_file=record.current_file,
            step=record.step,
            message=record.message,
            recent_logs=record.recent_logs,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


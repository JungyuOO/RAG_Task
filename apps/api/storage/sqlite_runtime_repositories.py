from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from apps.api.storage.json_persistence import load_json_file
from apps.api.schemas.actions import (
    OcpActionAuditEventType,
    OcpActionAuditListResponse,
    OcpActionAuditRecord,
    OcpActionExecutionListResponse,
    OcpActionExecutionRecord,
    OcpActionExecutionStatus,
    OcpActionRequestListResponse,
    OcpActionRequestRecord,
    OcpActionRequestStatus,
    OcpActionPreviewResponse,
    OcpActionType,
)
from apps.api.schemas.auth import OcpConnectionProfile
from apps.api.schemas.indexing import BatchIndexRequest, BatchIndexResponse, BatchJobStatusResponse
from apps.api.schemas.workspaces import (
    WorkspaceCreateRequest,
    WorkspaceModelProfile,
    WorkspaceModelProfileUpdateRequest,
    WorkspaceRecord,
    WorkspaceUpdateRequest,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _json_loads(payload: str) -> dict:
    return json.loads(payload) if payload else {}


class _SQLiteRepositoryBase:
    def __init__(self, *, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._initialize()

    def _open_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connect(self):
        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA foreign_keys=ON;")
            self._create_schema(connection)
            connection.commit()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        raise NotImplementedError


class SQLiteConnectionProfileStore(_SQLiteRepositoryBase):
    def __init__(self, *, db_path: Path, legacy_json_path: Path | None = None) -> None:
        self.legacy_json_path = legacy_json_path
        super().__init__(db_path=db_path)
        self._migrate_legacy_json_if_needed()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS connection_profiles (
                connection_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def put(self, profile: OcpConnectionProfile) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO connection_profiles (connection_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(connection_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    profile.connection_id,
                    _json_dumps(profile.model_dump(mode="json")),
                    _utc_now().isoformat(),
                ),
            )
            connection.commit()

    def get(self, connection_id: str) -> OcpConnectionProfile | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM connection_profiles WHERE connection_id = ?",
                (connection_id,),
            ).fetchone()
        if row is None:
            return None
        return OcpConnectionProfile.model_validate(_json_loads(row["payload_json"]))

    def list_profiles(self) -> list[OcpConnectionProfile]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM connection_profiles ORDER BY updated_at DESC, connection_id ASC"
            ).fetchall()
        return [OcpConnectionProfile.model_validate(_json_loads(row["payload_json"])) for row in rows]

    def delete(self, connection_id: str) -> OcpConnectionProfile | None:
        existing = self.get(connection_id)
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM connection_profiles WHERE connection_id = ?", (connection_id,))
            connection.commit()
        return existing

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM connection_profiles")
            connection.commit()

    def _migrate_legacy_json_if_needed(self) -> None:
        if self.legacy_json_path is None:
            return
        with self._lock, self._connect() as connection:
            existing_count = int(
                connection.execute("SELECT COUNT(*) AS count FROM connection_profiles").fetchone()["count"]
            )
        if existing_count > 0:
            return
        payload = load_json_file(self.legacy_json_path, default={})
        if not isinstance(payload, dict) or not payload:
            return
        for item in payload.values():
            profile = OcpConnectionProfile.model_validate(item)
            self.put(profile)


class SQLiteWorkspaceRepository(_SQLiteRepositoryBase):
    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def create_workspace(self, request: WorkspaceCreateRequest) -> WorkspaceRecord:
        now = _utc_now()
        normalized_slug = self._normalize_slug(request.slug or request.name)
        record = WorkspaceRecord(
            workspace_id=f"workspace-{uuid4().hex}",
            name=request.name,
            slug=normalized_slug,
            industry=request.industry,
            environment=request.environment,
            created_at=now,
            updated_at=now,
        )
        self._upsert(record)
        return record

    def list_workspaces(self) -> list[WorkspaceRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM workspaces ORDER BY updated_at DESC, created_at DESC, seq DESC"
            ).fetchall()
        return [WorkspaceRecord.model_validate(_json_loads(row["payload_json"])) for row in rows]

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkspaceRecord.model_validate(_json_loads(row["payload_json"]))

    def update_workspace(self, workspace_id: str, request: WorkspaceUpdateRequest) -> WorkspaceRecord:
        existing = self.get_workspace(workspace_id)
        if existing is None:
            raise LookupError("Workspace not found.")
        updated = existing.model_copy(
            update={
                "name": request.name or existing.name,
                "slug": self._normalize_slug(request.slug) if request.slug else existing.slug,
                "industry": request.industry if request.industry else existing.industry,
                "environment": request.environment if request.environment else existing.environment,
                "updated_at": _utc_now(),
            }
        )
        self._upsert(updated)
        return updated

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM workspaces")
            connection.commit()

    def _upsert(self, record: WorkspaceRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspaces (workspace_id, slug, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    slug=excluded.slug,
                    payload_json=excluded.payload_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    record.workspace_id,
                    record.slug,
                    _json_dumps(record.model_dump(mode="json")),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
            connection.commit()

    @staticmethod
    def _normalize_slug(value: str) -> str:
        next_value = str(value or "").strip().lower()
        slug = "".join(character if character.isalnum() else "-" for character in next_value).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug or f"workspace-{uuid4().hex[:8]}"


class SQLiteWorkspaceModelProfileRepository(_SQLiteRepositoryBase):
    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_model_profiles (
                workspace_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def get_profile(self, workspace_id: str) -> WorkspaceModelProfile:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM workspace_model_profiles WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            return WorkspaceModelProfile(
                workspace_id=workspace_id,
                updated_at=_utc_now(),
            )
        return WorkspaceModelProfile.model_validate(_json_loads(row["payload_json"]))

    def put_profile(self, workspace_id: str, request: WorkspaceModelProfileUpdateRequest) -> WorkspaceModelProfile:
        record = WorkspaceModelProfile(
            workspace_id=workspace_id,
            chat_provider=request.chat_provider,
            chat_base_url=request.chat_base_url,
            chat_model=request.chat_model,
            chat_api_key_mode=request.chat_api_key_mode,
            embedding_provider=request.embedding_provider,
            embedding_base_url=request.embedding_base_url,
            embedding_model=request.embedding_model,
            embedding_api_key_mode=request.embedding_api_key_mode,
            updated_at=_utc_now(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspace_model_profiles (workspace_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    workspace_id,
                    _json_dumps(record.model_dump(mode="json")),
                    record.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return record

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM workspace_model_profiles")
            connection.commit()


class SQLiteBatchJobRepository(_SQLiteRepositoryBase):
    def __init__(self, *, db_path: Path, legacy_json_path: Path | None = None) -> None:
        self.legacy_json_path = legacy_json_path
        super().__init__(db_path=db_path)
        self._migrate_legacy_json_if_needed()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                error TEXT NOT NULL DEFAULT '',
                progress_pct INTEGER NOT NULL DEFAULT 0,
                current_file TEXT NOT NULL DEFAULT '',
                step TEXT NOT NULL DEFAULT 'pending',
                message TEXT NOT NULL DEFAULT '',
                logs_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(batch_jobs)").fetchall()
        }
        if "step" not in columns:
            connection.execute("ALTER TABLE batch_jobs ADD COLUMN step TEXT NOT NULL DEFAULT 'pending'")
        if "message" not in columns:
            connection.execute("ALTER TABLE batch_jobs ADD COLUMN message TEXT NOT NULL DEFAULT ''")
        if "logs_json" not in columns:
            connection.execute("ALTER TABLE batch_jobs ADD COLUMN logs_json TEXT NOT NULL DEFAULT '[]'")

    def create(self, request: BatchIndexRequest) -> BatchJobStatusResponse:
        now = _utc_now()
        job = BatchJobStatusResponse(
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
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO batch_jobs (
                    job_id, status, request_json, result_json, error, progress_pct, current_file, step, message, logs_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.status,
                    _json_dumps(request.model_dump(mode="json")),
                    None,
                    "",
                    0,
                    "",
                    "pending",
                    "job queued",
                    "[]",
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return job

    def mark_running(self, job_id: str) -> BatchJobStatusResponse:
        return self._update(job_id, status="running", step="RUNNING", message="job started")

    def mark_progress(self, job_id: str, *, progress_pct: int, current_file: str) -> BatchJobStatusResponse:
        return self._update(
            job_id,
            status="running",
            progress_pct=max(0, min(100, int(progress_pct))),
            current_file=current_file,
        )

    def mark_step(self, job_id: str, *, step: str, message: str = "", current_file: str | None = None) -> BatchJobStatusResponse:
        return self._update(
            job_id,
            status="running" if step not in {"COMPLETED", "FAILED", "CANCELLED"} else step.lower(),
            step=step,
            message=message,
            current_file=current_file,
        )

    def append_log(self, job_id: str, message: str) -> BatchJobStatusResponse:
        existing = self.get(job_id)
        if existing is None:
            raise LookupError(f"Unknown batch job: {job_id}")
        logs = [*existing.recent_logs, str(message or "").strip()]
        logs = [item for item in logs if item][-40:]
        return self._update(job_id, recent_logs=logs)

    def mark_completed(self, job_id: str, result: BatchIndexResponse) -> BatchJobStatusResponse:
        return self._update(
            job_id,
            status="completed",
            result=result,
            error="",
            progress_pct=100,
            current_file="",
            step="COMPLETED",
            message="batch completed",
        )

    def mark_failed(self, job_id: str, error: str) -> BatchJobStatusResponse:
        return self._update(job_id, status="failed", error=error, current_file="", step="FAILED", message=error)

    def mark_cancelled(self, job_id: str, result: BatchIndexResponse | None = None) -> BatchJobStatusResponse:
        existing = self.get(job_id)
        if existing is None:
            raise LookupError(f"Unknown batch job: {job_id}")
        if existing.status in {"completed", "failed", "cancelled"}:
            if result is not None:
                return self._update(job_id, result=result)
            return existing
        return self._update(job_id, status="cancelled", result=result, current_file="", step="CANCELLED", message="job cancelled")

    def get(self, job_id: str) -> BatchJobStatusResponse | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    def list_recent(self, limit: int = 20) -> list[BatchJobStatusResponse]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM batch_jobs
                ORDER BY updated_at DESC, created_at DESC, seq DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM batch_jobs")
            connection.commit()

    def is_cancelled(self, job_id: str) -> bool:
        existing = self.get(job_id)
        return bool(existing and existing.status == "cancelled")

    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result: BatchIndexResponse | None = None,
        error: str | None = None,
        progress_pct: int | None = None,
        current_file: str | None = None,
        step: str | None = None,
        message: str | None = None,
        recent_logs: list[str] | None = None,
    ) -> BatchJobStatusResponse:
        existing = self.get(job_id)
        if existing is None:
            raise LookupError(f"Unknown batch job: {job_id}")
        updated = existing.model_copy(
            update={
                "status": status or existing.status,
                "result": result if result is not None else existing.result,
                "error": error if error is not None else existing.error,
                "progress_pct": progress_pct if progress_pct is not None else existing.progress_pct,
                "current_file": current_file if current_file is not None else existing.current_file,
                "step": step if step is not None else existing.step,
                "message": message if message is not None else existing.message,
                "recent_logs": recent_logs if recent_logs is not None else existing.recent_logs,
                "updated_at": _utc_now(),
            }
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE batch_jobs
                SET status = ?, request_json = ?, result_json = ?, error = ?, progress_pct = ?, current_file = ?, step = ?, message = ?, logs_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    updated.status,
                    _json_dumps(updated.request.model_dump(mode="json")),
                    _json_dumps(updated.result.model_dump(mode="json")) if updated.result else None,
                    updated.error,
                    updated.progress_pct,
                    updated.current_file,
                    updated.step,
                    updated.message,
                    _json_dumps(updated.recent_logs),
                    updated.updated_at.isoformat(),
                    job_id,
                ),
            )
            connection.commit()
        return updated

    def _row_to_job(self, row: sqlite3.Row) -> BatchJobStatusResponse:
        return BatchJobStatusResponse(
            job_id=row["job_id"],
            status=row["status"],
            request=BatchIndexRequest.model_validate(_json_loads(row["request_json"])),
            result=BatchIndexResponse.model_validate(_json_loads(row["result_json"])) if row["result_json"] else None,
            error=row["error"] or "",
            progress_pct=int(row["progress_pct"] or 0),
            current_file=row["current_file"] or "",
            step=row["step"] or "pending",
            message=row["message"] or "",
            recent_logs=list(_json_loads(row["logs_json"])) if row["logs_json"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _migrate_legacy_json_if_needed(self) -> None:
        if self.legacy_json_path is None:
            return
        with self._lock, self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) AS count FROM batch_jobs").fetchone()["count"])
        if count > 0:
            return
        payload = load_json_file(self.legacy_json_path, default={"jobs": []})
        for item in payload.get("jobs", []):
            job = BatchJobStatusResponse(
                job_id=item["job_id"],
                status=item["status"],
                request=BatchIndexRequest.model_validate(item["request"]),
                result=BatchIndexResponse.model_validate(item["result"]) if item.get("result") else None,
                error=item.get("error", ""),
                progress_pct=int(item.get("progress_pct") or 0),
                current_file=item.get("current_file", ""),
                step=item.get("step", item["status"]),
                message=item.get("message", ""),
                recent_logs=list(item.get("recent_logs") or []),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO batch_jobs (
                        job_id, status, request_json, result_json, error, progress_pct, current_file, step, message, logs_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.status,
                        _json_dumps(job.request.model_dump(mode="json")),
                        _json_dumps(job.result.model_dump(mode="json")) if job.result else None,
                        job.error,
                        job.progress_pct,
                        job.current_file,
                        job.step,
                        job.message,
                        _json_dumps(job.recent_logs),
                        job.created_at.isoformat(),
                        job.updated_at.isoformat(),
                    ),
                )
                connection.commit()


class SQLiteActionRequestRepository(_SQLiteRepositoryBase):
    def __init__(self, *, db_path: Path, legacy_json_path: Path | None = None) -> None:
        self.legacy_json_path = legacy_json_path
        super().__init__(db_path=db_path)
        self._migrate_legacy_json_if_needed()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS action_requests (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def create(
        self,
        *,
        preview: OcpActionPreviewResponse,
        reason: str,
        requested_by: str,
        requested_roles: list[str],
        required_approvals: int,
        manifest_yaml: str = "",
        resource_version: str | None = None,
    ) -> OcpActionRequestRecord:
        now = _utc_now()
        record = OcpActionRequestRecord(
            request_id=f"action-{uuid4().hex}",
            status=OcpActionRequestStatus.PENDING,
            preview=preview,
            requested_by=requested_by,
            requested_roles=list(requested_roles),
            required_approvals=max(int(required_approvals), 1),
            approval_count=0,
            approver_ids=[],
            approver_role_map={},
            reason=reason,
            manifest_yaml=manifest_yaml,
            resource_version=resource_version,
            decision_note="",
            created_at=now,
            updated_at=now,
        )
        self._upsert(record)
        return record

    def get(self, request_id: str) -> OcpActionRequestRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM action_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return OcpActionRequestRecord.model_validate(_json_loads(row["payload_json"]))

    def approve(self, request_id: str, *, actor_id: str, actor_roles: list[str], decision_note: str = "") -> OcpActionRequestRecord:
        existing = self._require(request_id)
        if existing.status == OcpActionRequestStatus.REJECTED:
            raise ValueError("Rejected requests cannot be approved.")
        if actor_id in existing.approver_ids:
            raise ValueError("The same actor cannot approve the same request twice.")
        if existing.status == OcpActionRequestStatus.APPROVED:
            raise ValueError("This request is already fully approved.")
        next_approvers = [*existing.approver_ids, actor_id]
        updated = existing.model_copy(
            update={
                "status": OcpActionRequestStatus.APPROVED
                if len(next_approvers) >= existing.required_approvals
                else OcpActionRequestStatus.PENDING,
                "approval_count": len(next_approvers),
                "approver_ids": next_approvers,
                "approver_role_map": {**existing.approver_role_map, actor_id: list(actor_roles)},
                "decision_note": decision_note,
                "updated_at": _utc_now(),
            }
        )
        self._upsert(updated)
        return updated

    def reject(self, request_id: str, *, decision_note: str = "") -> OcpActionRequestRecord:
        existing = self._require(request_id)
        updated = existing.model_copy(
            update={
                "status": OcpActionRequestStatus.REJECTED,
                "decision_note": decision_note,
                "updated_at": _utc_now(),
            }
        )
        self._upsert(updated)
        return updated

    def list_recent(self, limit: int = 20) -> OcpActionRequestListResponse:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM action_requests
                ORDER BY updated_at DESC, created_at DESC, seq DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        return OcpActionRequestListResponse(
            items=[OcpActionRequestRecord.model_validate(_json_loads(row["payload_json"])) for row in rows]
        )

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM action_requests")
            connection.commit()

    def _require(self, request_id: str) -> OcpActionRequestRecord:
        record = self.get(request_id)
        if record is None:
            raise LookupError(f"Unknown action request: {request_id}")
        return record

    def _upsert(self, record: OcpActionRequestRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO action_requests (request_id, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    record.request_id,
                    record.status.value,
                    _json_dumps(record.model_dump(mode="json")),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
            connection.commit()

    def _migrate_legacy_json_if_needed(self) -> None:
        if self.legacy_json_path is None:
            return
        with self._lock, self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) AS count FROM action_requests").fetchone()["count"])
        if count > 0:
            return
        payload = load_json_file(self.legacy_json_path, default={"items": []})
        for item in payload.get("items", []):
            migrated = {key: value for key, value in dict(item).items() if key != "order"}
            self._upsert(OcpActionRequestRecord.model_validate(migrated))


class SQLiteActionExecutionRepository(_SQLiteRepositoryBase):
    def __init__(self, *, db_path: Path, legacy_json_path: Path | None = None) -> None:
        self.legacy_json_path = legacy_json_path
        super().__init__(db_path=db_path)
        self._migrate_legacy_json_if_needed()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS action_executions (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

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
        now = _utc_now()
        record = OcpActionExecutionRecord(
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
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO action_executions (execution_id, request_id, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.execution_id,
                    record.request_id,
                    record.status.value,
                    _json_dumps(record.model_dump(mode="json")),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return record

    def list_recent(self, limit: int = 20) -> OcpActionExecutionListResponse:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM action_executions
                ORDER BY updated_at DESC, created_at DESC, seq DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        return OcpActionExecutionListResponse(
            items=[OcpActionExecutionRecord.model_validate(_json_loads(row["payload_json"])) for row in rows]
        )

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM action_executions")
            connection.commit()

    def _migrate_legacy_json_if_needed(self) -> None:
        if self.legacy_json_path is None:
            return
        with self._lock, self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) AS count FROM action_executions").fetchone()["count"])
        if count > 0:
            return
        payload = load_json_file(self.legacy_json_path, default={"items": []})
        for item in payload.get("items", []):
            migrated = {key: value for key, value in dict(item).items() if key != "order"}
            record = OcpActionExecutionRecord.model_validate(migrated)
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO action_executions (
                        execution_id, request_id, status, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.execution_id,
                        record.request_id,
                        record.status.value,
                        _json_dumps(record.model_dump(mode="json")),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
                connection.commit()


class SQLiteActionAuditRepository(_SQLiteRepositoryBase):
    def __init__(self, *, db_path: Path, legacy_json_path: Path | None = None) -> None:
        self.legacy_json_path = legacy_json_path
        super().__init__(db_path=db_path)
        self._migrate_legacy_json_if_needed()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS action_audit (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    def create(
        self,
        *,
        event_type: OcpActionAuditEventType,
        actor_id: str,
        request_id: str,
        execution_id: str,
        action_type: OcpActionType,
        namespace: str,
        resource_name: str,
        risk_level: str,
        decision_note: str = "",
        details: dict | None = None,
    ) -> OcpActionAuditRecord:
        record = OcpActionAuditRecord(
            event_id=f"audit-{uuid4().hex}",
            event_type=event_type,
            actor_id=actor_id,
            request_id=request_id,
            execution_id=execution_id,
            action_type=action_type,
            namespace=namespace,
            resource_name=resource_name,
            risk_level=risk_level,
            decision_note=decision_note,
            details=dict(details or {}),
            created_at=_utc_now(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO action_audit (event_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    record.event_type.value,
                    _json_dumps(record.model_dump(mode="json")),
                    record.created_at.isoformat(),
                ),
            )
            connection.commit()
        return record

    def list_recent(self, limit: int = 20) -> OcpActionAuditListResponse:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM action_audit
                ORDER BY created_at DESC, seq DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        return OcpActionAuditListResponse(
            items=[OcpActionAuditRecord.model_validate(_json_loads(row["payload_json"])) for row in rows]
        )

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM action_audit")
            connection.commit()

    def _migrate_legacy_json_if_needed(self) -> None:
        if self.legacy_json_path is None:
            return
        with self._lock, self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) AS count FROM action_audit").fetchone()["count"])
        if count > 0:
            return
        payload = load_json_file(self.legacy_json_path, default={"items": []})
        for item in payload.get("items", []):
            migrated = {key: value for key, value in dict(item).items() if key != "order"}
            record = OcpActionAuditRecord.model_validate(migrated)
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO action_audit (event_id, event_type, payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        record.event_id,
                        record.event_type.value,
                        _json_dumps(record.model_dump(mode="json")),
                        record.created_at.isoformat(),
                    ),
                )
                connection.commit()




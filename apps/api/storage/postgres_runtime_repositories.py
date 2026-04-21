from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock

import psycopg2
import psycopg2.extras

from apps.api.schemas.recommendations import (
    MetricSnapshotListResponse,
    MetricSnapshotRecord,
    RecommendationListResponse,
    RecommendationRecord,
)
from apps.api.schemas.scm import (
    ScmConnectionCreateRequest,
    ScmConnectionListResponse,
    ScmConnectionRecord,
    ScmRepositoryCreateRequest,
    ScmRepositoryListResponse,
    ScmRepositoryRecord,
    ScmRepositoryUpdateRequest,
)
from apps.api.schemas.workspaces import (
    WorkspaceCreateRequest,
    WorkspaceModelProfile,
    WorkspaceModelProfileUpdateRequest,
    WorkspaceRecord,
    WorkspaceUpdateRequest,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class _PostgresRepositoryBase:
    def __init__(self, *, dsn: str) -> None:
        self.dsn = dsn
        self._lock = Lock()
        self._initialized = False

    @contextmanager
    def _connect(self):
        connection = psycopg2.connect(self.dsn)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ping(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    self._create_schema(cursor)
            self._initialized = True

    def _create_schema(self, cursor) -> None:
        raise NotImplementedError


class PostgresWorkspaceRepository(_PostgresRepositoryBase):
    def _create_schema(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL UNIQUE,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )

    def create_workspace(self, request: WorkspaceCreateRequest) -> WorkspaceRecord:
        self._ensure_initialized()
        now = _utc_now()
        normalized_slug = self._normalize_slug(request.slug or request.name)
        record = WorkspaceRecord(
            workspace_id=f"workspace-{normalized_slug}" if normalized_slug else "workspace",
            name=request.name,
            slug=normalized_slug,
            industry=request.industry,
            environment=request.environment,
            created_at=now,
            updated_at=now,
        )
        if record.workspace_id == "workspace":
            record = record.model_copy(update={"workspace_id": f"workspace-{now.timestamp():.0f}"})
        self._upsert(record)
        return record

    def list_workspaces(self) -> list[WorkspaceRecord]:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT payload_json
                    FROM workspaces
                    ORDER BY updated_at DESC, created_at DESC
                    """
                )
                rows = cursor.fetchall()
        return [WorkspaceRecord.model_validate(row["payload_json"]) for row in rows]

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT payload_json FROM workspaces WHERE workspace_id = %s",
                    (workspace_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return WorkspaceRecord.model_validate(row["payload_json"])

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
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM workspaces")

    def _upsert(self, record: WorkspaceRecord) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO workspaces (workspace_id, slug, payload_json, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        slug = EXCLUDED.slug,
                        payload_json = EXCLUDED.payload_json,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.workspace_id,
                        record.slug,
                        psycopg2.extras.Json(record.model_dump(mode="json")),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )

    @staticmethod
    def _normalize_slug(value: str) -> str:
        next_value = str(value or "").strip().lower()
        slug = "".join(character if character.isalnum() else "-" for character in next_value).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug or "workspace"


class PostgresWorkspaceModelProfileRepository(_PostgresRepositoryBase):
    def _create_schema(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_model_profiles (
                workspace_id TEXT PRIMARY KEY,
                payload_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )

    def get_profile(self, workspace_id: str) -> WorkspaceModelProfile:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT payload_json FROM workspace_model_profiles WHERE workspace_id = %s",
                    (workspace_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return WorkspaceModelProfile(workspace_id=workspace_id, updated_at=_utc_now())
        return WorkspaceModelProfile.model_validate(row["payload_json"])

    def put_profile(self, workspace_id: str, request: WorkspaceModelProfileUpdateRequest) -> WorkspaceModelProfile:
        self._ensure_initialized()
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
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO workspace_model_profiles (workspace_id, payload_json, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        payload_json = EXCLUDED.payload_json,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        workspace_id,
                        psycopg2.extras.Json(record.model_dump(mode="json")),
                        record.updated_at.isoformat(),
                    ),
                )
        return record

    def clear(self) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM workspace_model_profiles")


class PostgresMetricSnapshotRepository(_PostgresRepositoryBase):
    def _create_schema(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metric_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                connection_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                metric_key TEXT NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                unit TEXT NOT NULL,
                payload_json JSONB NOT NULL,
                collected_at TIMESTAMPTZ NOT NULL
            )
            """
        )

    def create(
        self,
        *,
        workspace_id: str,
        connection_id: str,
        namespace: str,
        metric_key: str,
        metric_value: float,
        unit: str = "",
    ) -> MetricSnapshotRecord:
        self._ensure_initialized()
        record = MetricSnapshotRecord(
            snapshot_id=f"metric-{_utc_now().timestamp():.0f}-{metric_key}",
            workspace_id=workspace_id,
            connection_id=connection_id,
            namespace=namespace,
            metric_key=metric_key,
            metric_value=float(metric_value),
            unit=unit,
            collected_at=_utc_now(),
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO metric_snapshots (
                        snapshot_id, workspace_id, connection_id, namespace, metric_key, metric_value, unit, payload_json, collected_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(snapshot_id) DO UPDATE SET
                        payload_json = EXCLUDED.payload_json,
                        collected_at = EXCLUDED.collected_at
                    """,
                    (
                        record.snapshot_id,
                        record.workspace_id,
                        record.connection_id,
                        record.namespace,
                        record.metric_key,
                        record.metric_value,
                        record.unit,
                        psycopg2.extras.Json(record.model_dump(mode="json")),
                        record.collected_at.isoformat(),
                    ),
                )
        return record

    def list_recent(self, *, workspace_id: str, limit: int = 20) -> MetricSnapshotListResponse:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT payload_json FROM metric_snapshots
                    WHERE workspace_id = %s
                    ORDER BY collected_at DESC
                    LIMIT %s
                    """,
                    (workspace_id, max(int(limit), 1)),
                )
                rows = cursor.fetchall()
        return MetricSnapshotListResponse(items=[MetricSnapshotRecord.model_validate(row["payload_json"]) for row in rows])

    def clear(self) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM metric_snapshots")


class PostgresRecommendationLogRepository(_PostgresRepositoryBase):
    def _create_schema(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                recommendation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                connection_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                recommendation_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                resource_kind TEXT NOT NULL,
                resource_name TEXT NOT NULL,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """
        )

    def create(
        self,
        *,
        workspace_id: str,
        connection_id: str,
        namespace: str,
        recommendation_type: str,
        risk_level: str,
        summary: str,
        rationale: str = "",
        resource_kind: str = "",
        resource_name: str = "",
    ) -> RecommendationRecord:
        self._ensure_initialized()
        record = RecommendationRecord(
            recommendation_id=f"rec-{_utc_now().timestamp():.0f}-{recommendation_type}",
            workspace_id=workspace_id,
            connection_id=connection_id,
            namespace=namespace,
            recommendation_type=recommendation_type,
            risk_level=risk_level,
            resource_kind=resource_kind,
            resource_name=resource_name,
            summary=summary,
            rationale=rationale,
            created_at=_utc_now(),
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO recommendation_logs (
                        recommendation_id, workspace_id, connection_id, namespace, recommendation_type, risk_level, resource_kind, resource_name, payload_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(recommendation_id) DO UPDATE SET
                        payload_json = EXCLUDED.payload_json,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        record.recommendation_id,
                        record.workspace_id,
                        record.connection_id,
                        record.namespace,
                        record.recommendation_type,
                        record.risk_level,
                        record.resource_kind,
                        record.resource_name,
                        psycopg2.extras.Json(record.model_dump(mode="json")),
                        record.created_at.isoformat(),
                    ),
                )
        return record

    def list_recent(self, *, workspace_id: str, limit: int = 20) -> RecommendationListResponse:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT payload_json FROM recommendation_logs
                    WHERE workspace_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (workspace_id, max(int(limit), 1)),
                )
                rows = cursor.fetchall()
        return RecommendationListResponse(
            items=[RecommendationRecord.model_validate(row["payload_json"]) for row in rows]
        )

    def clear(self) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM recommendation_logs")


class PostgresScmConnectionRepository(_PostgresRepositoryBase):
    def _create_schema(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scm_connections (
                scm_connection_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                host_url TEXT NOT NULL,
                auth_type TEXT NOT NULL,
                account_label TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )

    def create(self, workspace_id: str, request: ScmConnectionCreateRequest) -> ScmConnectionRecord:
        self._ensure_initialized()
        now = _utc_now()
        record = ScmConnectionRecord(
            scm_connection_id=f"scm-{workspace_id}-{request.provider}-{int(now.timestamp())}",
            workspace_id=workspace_id,
            provider=request.provider,
            host_url=request.host_url,
            auth_type=request.auth_type,
            account_label=request.account_label,
            login_name=request.login_name,
            scopes=list(request.scopes or []),
            secret_ref=request.secret_ref,
            status="connected",
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO scm_connections (
                        scm_connection_id, workspace_id, provider, host_url, auth_type, account_label, status, payload_json, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.scm_connection_id,
                        record.workspace_id,
                        record.provider,
                        record.host_url,
                        record.auth_type,
                        record.account_label,
                        record.status,
                        psycopg2.extras.Json(record.model_dump(mode="json")),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
        return record

    def list_by_workspace(self, workspace_id: str) -> ScmConnectionListResponse:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT payload_json FROM scm_connections
                    WHERE workspace_id = %s
                    ORDER BY updated_at DESC, created_at DESC
                    """,
                    (workspace_id,),
                )
                rows = cursor.fetchall()
        return ScmConnectionListResponse(items=[ScmConnectionRecord.model_validate(row["payload_json"]) for row in rows])

    def get(self, scm_connection_id: str) -> ScmConnectionRecord | None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT payload_json FROM scm_connections WHERE scm_connection_id = %s",
                    (scm_connection_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return ScmConnectionRecord.model_validate(row["payload_json"])

    def clear(self) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM scm_connections")


class PostgresScmRepositoryRepository(_PostgresRepositoryBase):
    def _create_schema(self, cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scm_repositories (
                repository_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                scm_connection_id TEXT NOT NULL,
                repo_full_name TEXT NOT NULL,
                default_branch TEXT NOT NULL,
                config_path TEXT NOT NULL,
                sync_status TEXT NOT NULL,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )

    def create(self, workspace_id: str, request: ScmRepositoryCreateRequest) -> ScmRepositoryRecord:
        self._ensure_initialized()
        now = _utc_now()
        record = ScmRepositoryRecord(
            repository_id=f"repo-{workspace_id}-{int(now.timestamp())}",
            workspace_id=workspace_id,
            scm_connection_id=request.scm_connection_id,
            repo_full_name=request.repo_full_name,
            default_branch=request.default_branch,
            config_path=request.config_path,
            delivery_mode=request.delivery_mode,
            manifest_kind=request.manifest_kind,
            target_cluster_url=request.target_cluster_url,
            target_namespace=request.target_namespace,
            auto_deploy_enabled=request.auto_deploy_enabled,
            sync_status="configured",
            created_at=now,
            updated_at=now,
        )
        self._upsert(record)
        return record

    def update(self, repository_id: str, request: ScmRepositoryUpdateRequest) -> ScmRepositoryRecord:
        existing = self.get(repository_id)
        if existing is None:
            raise LookupError("Repository not found.")
        updated = existing.model_copy(
            update={
                "default_branch": request.default_branch or existing.default_branch,
                "config_path": request.config_path or existing.config_path,
                "delivery_mode": request.delivery_mode or existing.delivery_mode,
                "manifest_kind": request.manifest_kind or existing.manifest_kind,
                "target_cluster_url": request.target_cluster_url or existing.target_cluster_url,
                "target_namespace": request.target_namespace or existing.target_namespace,
                "auto_deploy_enabled": existing.auto_deploy_enabled if request.auto_deploy_enabled is None else request.auto_deploy_enabled,
                "updated_at": _utc_now(),
            }
        )
        self._upsert(updated)
        return updated

    def list_by_workspace(self, workspace_id: str) -> ScmRepositoryListResponse:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT payload_json FROM scm_repositories
                    WHERE workspace_id = %s
                    ORDER BY updated_at DESC, created_at DESC
                    """,
                    (workspace_id,),
                )
                rows = cursor.fetchall()
        return ScmRepositoryListResponse(items=[ScmRepositoryRecord.model_validate(row["payload_json"]) for row in rows])

    def get(self, repository_id: str) -> ScmRepositoryRecord | None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT payload_json FROM scm_repositories WHERE repository_id = %s",
                    (repository_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return ScmRepositoryRecord.model_validate(row["payload_json"])

    def clear(self) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM scm_repositories")

    def _upsert(self, record: ScmRepositoryRecord) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO scm_repositories (
                        repository_id, workspace_id, scm_connection_id, repo_full_name, default_branch, config_path, sync_status, payload_json, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(repository_id) DO UPDATE SET
                        default_branch = EXCLUDED.default_branch,
                        config_path = EXCLUDED.config_path,
                        sync_status = EXCLUDED.sync_status,
                        payload_json = EXCLUDED.payload_json,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.repository_id,
                        record.workspace_id,
                        record.scm_connection_id,
                        record.repo_full_name,
                        record.default_branch,
                        record.config_path,
                        record.sync_status,
                        psycopg2.extras.Json(record.model_dump(mode="json")),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )

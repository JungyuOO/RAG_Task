from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from apps.api.storage.sqlite_runtime_repositories import (
    SQLiteActionAuditRepository,
    SQLiteActionExecutionRepository,
    SQLiteActionRequestRepository,
    SQLiteBatchJobRepository,
    SQLiteConnectionProfileStore,
)
from apps.api.schemas.batch_indexing import BatchIndexRequest
from apps.api.schemas.ocp_action_audit import OcpActionAuditEventType
from apps.api.schemas.ocp_action_executions import OcpActionExecutionStatus
from apps.api.schemas.ocp_action_requests import OcpActionRequestStatus
from apps.api.schemas.ocp_actions import OcpActionPreviewResponse, OcpActionType
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionProfile

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "sqlite_runtime_repositories"
FIXTURE_ROOT = "fixtures/generated-manual"


class SQLiteRuntimeRepositoriesTests(unittest.TestCase):
    def tearDown(self) -> None:
        if RESULT_DIR.exists():
            shutil.rmtree(RESULT_DIR, ignore_errors=True)

    def test_connection_profile_store_migrates_legacy_json(self) -> None:
        db_path = RESULT_DIR / "runtime.sqlite3"
        legacy_path = RESULT_DIR / "connection_profiles.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        profile = OcpConnectionProfile(
            connection_id="ocp-conn-1",
            display_name="demo",
            cluster_url="https://api.cluster.example.com",
            auth_mode=OcpAuthMode.TOKEN,
            verify_ssl=True,
            default_namespace="demo",
            username_hint="",
            secret_ref="ocp-secret-1",
            save_profile=False,
            status="connected",
            last_verified_at="2026-04-14T00:00:00+00:00",
        )
        legacy_path.write_text(
            json.dumps({profile.connection_id: profile.model_dump(mode="json")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        store = SQLiteConnectionProfileStore(db_path=db_path, legacy_json_path=legacy_path)
        loaded = store.get(profile.connection_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.cluster_url, profile.cluster_url)

    def test_batch_job_repository_persists_jobs(self) -> None:
        db_path = RESULT_DIR / "runtime.sqlite3"
        repository = SQLiteBatchJobRepository(db_path=db_path)
        created = repository.create(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=1))
        repository.mark_running(created.job_id)

        reloaded = SQLiteBatchJobRepository(db_path=db_path)
        job = reloaded.get(created.job_id)
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.status, "running")

    def test_action_request_repository_persists_requests(self) -> None:
        db_path = RESULT_DIR / "runtime.sqlite3"
        repository = SQLiteActionRequestRepository(db_path=db_path)
        preview = OcpActionPreviewResponse(
            connection_id="conn-1",
            action_type=OcpActionType.SCALE_DEPLOYMENT,
            namespace="demo",
            resource_name="demo-app",
            allowed=True,
            risk_level="medium",
            summary="scale preview",
            preview_command="oc scale deployment/demo-app -n demo --replicas=2",
            policy_checks=["Replica guardrail enabled."],
            blocked_reasons=[],
            validation_messages=[],
            next_step="preview",
        )
        created = repository.create(
            preview=preview,
            reason="scale out",
            requested_by="ui-local",
            requested_roles=["operator"],
            required_approvals=2,
        )
        repository.approve(created.request_id, actor_id="approver-a", actor_roles=["operator"], decision_note="approved-stage-1")
        repository.approve(created.request_id, actor_id="approver-b", actor_roles=["admin"], decision_note="approved-stage-2")

        reloaded = SQLiteActionRequestRepository(db_path=db_path)
        item = reloaded.get(created.request_id)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.status, OcpActionRequestStatus.APPROVED)
        self.assertEqual(item.preview.preview_command, preview.preview_command)

    def test_action_execution_repository_persists_preflight_checks(self) -> None:
        db_path = RESULT_DIR / "runtime.sqlite3"
        repository = SQLiteActionExecutionRepository(db_path=db_path)
        preview = OcpActionPreviewResponse(
            connection_id="conn-1",
            action_type=OcpActionType.ROLLOUT_RESTART,
            namespace="demo",
            resource_name="demo-app",
            allowed=True,
            risk_level="medium",
            summary="restart preview",
            preview_command="oc rollout restart deployment/demo-app -n demo",
            policy_checks=[],
            blocked_reasons=[],
            validation_messages=[],
            next_step="preview",
        )
        created = repository.create(
            request_id="action-1",
            status=OcpActionExecutionStatus.SUCCEEDED,
            execution_mode="dry_run",
            simulated=False,
            preview=preview,
            summary="dry-run complete",
            preflight_checks=["deployment exists", "paused=no"],
            output_lines=["ok"],
        )

        reloaded = SQLiteActionExecutionRepository(db_path=db_path)
        items = reloaded.list_recent(10).items
        self.assertGreaterEqual(len(items), 1)
        self.assertEqual(items[0].execution_id, created.execution_id)
        self.assertIn("paused=no", items[0].preflight_checks)

    def test_action_audit_repository_persists_records(self) -> None:
        db_path = RESULT_DIR / "runtime.sqlite3"
        repository = SQLiteActionAuditRepository(db_path=db_path)
        created = repository.create(
            event_type=OcpActionAuditEventType.REQUEST_CREATED,
            actor_id="ui-local",
            request_id="action-1",
            execution_id="",
            action_type=OcpActionType.LOG_BUNDLE,
            namespace="demo",
            resource_name="demo-pod",
            risk_level="low",
            details={"execution_mode": "read_only"},
        )

        reloaded = SQLiteActionAuditRepository(db_path=db_path)
        items = reloaded.list_recent(10).items
        self.assertGreaterEqual(len(items), 1)
        self.assertEqual(items[0].event_id, created.event_id)
        self.assertEqual(items[0].details["execution_mode"], "read_only")


if __name__ == "__main__":
    unittest.main()





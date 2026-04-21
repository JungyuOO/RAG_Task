from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.api.storage.action_request_repository import InMemoryActionRequestRepository
from apps.api.storage.batch_job_repository import InMemoryBatchJobRepository
from apps.api.schemas.batch_indexing import BatchIndexRequest
from apps.api.schemas.ocp_actions import OcpActionPreviewResponse, OcpActionType
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionProfile, OcpConnectionRequest
from apps.api.ocp.auth import InMemoryConnectionProfileStore, InMemoryConnectionSecretStore, OcpConnectionBroker

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "persistent_runtime_repositories"
FIXTURE_ROOT = "fixtures/generated-manual"


class PersistentRuntimeRepositoriesTests(unittest.TestCase):
    def tearDown(self) -> None:
        if RESULT_DIR.exists():
            shutil.rmtree(RESULT_DIR, ignore_errors=True)

    def test_connection_stores_persist_profiles_and_secrets(self) -> None:
        secret_path = RESULT_DIR / "connection_secrets.protected.json"
        profile_path = RESULT_DIR / "connection_profiles.json"

        secret_store = InMemoryConnectionSecretStore(storage_path=secret_path)
        profile_store = InMemoryConnectionProfileStore(storage_path=profile_path)
        broker = OcpConnectionBroker(secret_store=secret_store, profile_store=profile_store)
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        reloaded_secret_store = InMemoryConnectionSecretStore(storage_path=secret_path)
        reloaded_profile_store = InMemoryConnectionProfileStore(storage_path=profile_path)
        reloaded_broker = OcpConnectionBroker(secret_store=reloaded_secret_store, profile_store=reloaded_profile_store)
        reloaded_profile = reloaded_broker.get_profile(profile.connection_id)

        self.assertIsNotNone(reloaded_profile)
        assert reloaded_profile is not None
        runtime = reloaded_broker.build_runtime_config(reloaded_profile)
        self.assertEqual(runtime["token"], "sha256~abc")
        raw_secret_file = secret_path.read_text(encoding="utf-8")
        self.assertNotIn("sha256~abc", raw_secret_file)
        payload = json.loads(raw_secret_file)
        self.assertEqual(payload.get("format"), "dpapi-json-v1")

    def test_connection_secret_store_migrates_legacy_plaintext_file(self) -> None:
        secret_path = RESULT_DIR / "connection_secrets.protected.json"
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        secret_path.write_text(
            json.dumps(
                {
                    "ocp-secret-legacy": {
                        "auth_mode": "token",
                        "payload": {"token": "sha256~legacy"},
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        secret_store = InMemoryConnectionSecretStore(storage_path=secret_path)
        secret = secret_store.get("ocp-secret-legacy")
        self.assertIsNotNone(secret)
        assert secret is not None
        self.assertEqual(secret.payload["token"], "sha256~legacy")

        rewritten = secret_path.read_text(encoding="utf-8")
        self.assertNotIn("sha256~legacy", rewritten)
        self.assertIn("dpapi-json-v1", rewritten)

    def test_connection_secret_store_supports_env_key_backend(self) -> None:
        secret_path = RESULT_DIR / "connection_secrets.protected.json"
        profile_path = RESULT_DIR / "connection_profiles.json"

        with patch.dict(
            os.environ,
            {
                "RAG_TASK_SECRET_BACKEND": "env_key",
                "RAG_TASK_SECRET_MASTER_KEY": "unit-test-master-key-for-rag-task",
            },
            clear=False,
        ):
            secret_store = InMemoryConnectionSecretStore(storage_path=secret_path)
            profile_store = InMemoryConnectionProfileStore(storage_path=profile_path)
            broker = OcpConnectionBroker(secret_store=secret_store, profile_store=profile_store)
            profile = broker.create_profile(
                OcpConnectionRequest(
                    cluster_url="https://api.cluster.example.com",
                    auth_mode=OcpAuthMode.TOKEN,
                    token="sha256~envkey",
                    default_namespace="demo",
                )
            )

            reloaded_secret_store = InMemoryConnectionSecretStore(storage_path=secret_path)
            reloaded_profile_store = InMemoryConnectionProfileStore(storage_path=profile_path)
            reloaded_broker = OcpConnectionBroker(secret_store=reloaded_secret_store, profile_store=reloaded_profile_store)
            reloaded_profile = reloaded_broker.get_profile(profile.connection_id)
            self.assertIsNotNone(reloaded_profile)
            assert reloaded_profile is not None
            runtime = reloaded_broker.build_runtime_config(reloaded_profile)
            self.assertEqual(runtime["token"], "sha256~envkey")
            raw_secret_file = secret_path.read_text(encoding="utf-8")
            self.assertNotIn("sha256~envkey", raw_secret_file)
            payload = json.loads(raw_secret_file)
            self.assertEqual(payload.get("format"), "envkey-json-v1")

    def test_batch_job_repository_persists_jobs(self) -> None:
        storage_path = RESULT_DIR / "batch_jobs.json"
        repository = InMemoryBatchJobRepository(storage_path=storage_path)
        created = repository.create(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=1))
        repository.mark_running(created.job_id)

        reloaded = InMemoryBatchJobRepository(storage_path=storage_path)
        job = reloaded.get(created.job_id)
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.status, "running")

    def test_action_request_repository_persists_requests(self) -> None:
        storage_path = RESULT_DIR / "action_requests.json"
        repository = InMemoryActionRequestRepository(storage_path=storage_path)
        preview = OcpActionPreviewResponse(
            connection_id="conn-1",
            action_type=OcpActionType.SCALE_DEPLOYMENT,
            namespace="demo",
            resource_name="demo-app",
            allowed=True,
            risk_level="medium",
            summary="scale preview",
            preview_command="oc scale deployment/demo-app -n demo --replicas=2",
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

        reloaded = InMemoryActionRequestRepository(storage_path=storage_path)
        item = reloaded.get(created.request_id)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.preview.preview_command, preview.preview_command)


if __name__ == "__main__":
    unittest.main()





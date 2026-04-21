from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.scm import router as scm_router
from apps.api.routes.workspaces import router as workspaces_router
from apps.api.runtime import scm_connection_repository, scm_repository_repository, workspace_repository


class ScmRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(workspaces_router, prefix="/api/v1")
        self.app.include_router(scm_router, prefix="/api/v1")
        self.client = TestClient(self.app)
        scm_repository_repository.clear()
        scm_connection_repository.clear()
        workspace_repository.clear()

    def test_workspace_scoped_scm_connection_and_repository_flow(self) -> None:
        workspace = self.client.post(
            "/api/v1/workspaces",
            json={"name": "Customer SCM", "slug": "customer-scm"},
        ).json()
        workspace_id = workspace["workspace_id"]

        create_connection_response = self.client.post(
            f"/api/v1/workspaces/{workspace_id}/scm/connections",
            json={
                "provider": "github",
                "host_url": "https://github.com",
                "auth_type": "token",
                "account_label": "customer-admin",
            },
        )
        self.assertEqual(create_connection_response.status_code, 200)
        connection = create_connection_response.json()
        self.assertEqual(connection["workspace_id"], workspace_id)
        self.assertEqual(connection["provider"], "github")

        list_connection_response = self.client.get(f"/api/v1/workspaces/{workspace_id}/scm/connections")
        self.assertEqual(list_connection_response.status_code, 200)
        self.assertEqual(len(list_connection_response.json()["items"]), 1)

        create_repository_response = self.client.post(
            f"/api/v1/workspaces/{workspace_id}/scm/repositories",
            json={
                "scm_connection_id": connection["scm_connection_id"],
                "repo_full_name": "customer/platform-config",
                "default_branch": "main",
                "config_path": "deploy/config.yaml",
                "delivery_mode": "gitops_commit",
                "manifest_kind": "config_yaml",
                "target_cluster_url": "https://api.cluster.example.com:6443",
                "target_namespace": "payments",
                "auto_deploy_enabled": True,
            },
        )
        self.assertEqual(create_repository_response.status_code, 200)
        repository = create_repository_response.json()
        self.assertEqual(repository["workspace_id"], workspace_id)
        self.assertEqual(repository["config_path"], "deploy/config.yaml")
        self.assertEqual(repository["target_namespace"], "payments")

        update_repository_response = self.client.patch(
            f"/api/v1/workspaces/{workspace_id}/scm/repositories/{repository['repository_id']}",
            json={
                "default_branch": "release",
                "config_path": ".ocp/config.yaml",
                "manifest_kind": "helm_values",
                "auto_deploy_enabled": False,
            },
        )
        self.assertEqual(update_repository_response.status_code, 200)
        updated_repository = update_repository_response.json()
        self.assertEqual(updated_repository["default_branch"], "release")
        self.assertEqual(updated_repository["config_path"], ".ocp/config.yaml")
        self.assertEqual(updated_repository["manifest_kind"], "helm_values")
        self.assertFalse(updated_repository["auto_deploy_enabled"])

        list_repository_response = self.client.get(f"/api/v1/workspaces/{workspace_id}/scm/repositories")
        self.assertEqual(list_repository_response.status_code, 200)
        self.assertEqual(len(list_repository_response.json()["items"]), 1)

        plan_response = self.client.post(
            f"/api/v1/workspaces/{workspace_id}/scm/repositories/{repository['repository_id']}/deployment-plan",
            json={
                "resource_kind": "Deployment",
                "resource_name": "payments-api",
                "target_namespace": "payments",
                "replicas": 5,
                "reason": "traffic increase",
            },
        )
        self.assertEqual(plan_response.status_code, 200)
        plan = plan_response.json()
        self.assertEqual(plan["repository_id"], repository["repository_id"])
        self.assertEqual(plan["trigger_kind"], "gitops_sync")
        self.assertTrue(any("replicaCount" in item for item in plan["suggested_updates"]))

    def test_repository_creation_rejects_connection_from_other_workspace(self) -> None:
        workspace_a = self.client.post("/api/v1/workspaces", json={"name": "A", "slug": "a"}).json()
        workspace_b = self.client.post("/api/v1/workspaces", json={"name": "B", "slug": "b"}).json()

        connection = self.client.post(
            f"/api/v1/workspaces/{workspace_a['workspace_id']}/scm/connections",
            json={"provider": "gitlab", "account_label": "ops"},
        ).json()

        response = self.client.post(
            f"/api/v1/workspaces/{workspace_b['workspace_id']}/scm/repositories",
            json={
                "scm_connection_id": connection["scm_connection_id"],
                "repo_full_name": "group/project",
                "default_branch": "main",
                "config_path": "config.yaml",
            },
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

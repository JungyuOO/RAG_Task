from __future__ import annotations

import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.ocp.live_service import ConnectedOcpService
from apps.api.routes.auth import _broker
from apps.api.routes.auth import router as auth_router
from apps.api.routes.workspaces import router as workspaces_router
from apps.api.runtime import connected_ocp_service, metric_snapshot_repository, recommendation_log_repository, workspace_repository


class WorkspaceRecommendationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(auth_router, prefix="/api/v1/auth")
        self.app.include_router(workspaces_router, prefix="/api/v1")
        self.client = TestClient(self.app)
        _broker.profile_store.clear()
        _broker.secret_store.clear()
        workspace_repository.clear()
        metric_snapshot_repository.clear()
        recommendation_log_repository.clear()
        connected_ocp_service.transport = None

    def test_refresh_workspace_recommendations_creates_logs(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/proxy/api/v1/query_range"):
                return httpx.Response(
                    200,
                    json={"data": {"result": [{"values": [[1710000000, "1.5"], [1710000300, "2.0"]]}]}},
                )
            if path.endswith("/apis/apps/v1/namespaces/demo/deployments"):
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "metadata": {"name": "demo-app", "creationTimestamp": "2026-01-01T00:00:00Z"},
                                "kind": "Deployment",
                                "spec": {"replicas": 3},
                                "status": {"readyReplicas": 1},
                            }
                        ]
                    },
                )
            if path.endswith("/api/v1/namespaces"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "demo"}}]})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        connected_ocp_service.transport = httpx.MockTransport(handler)
        workspace = self.client.post(
            "/api/v1/workspaces",
            json={"name": "Customer A", "slug": "customer-a"},
        ).json()
        connect = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "workspace_id": workspace["workspace_id"],
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
                "display_name": "demo-cluster",
            },
        ).json()
        connection_id = connect["connection"]["connection_id"]

        response = self.client.post(
            f"/api/v1/workspaces/{workspace['workspace_id']}/recommendations/refresh",
            json={"connection_id": connection_id},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["workspace_id"], workspace["workspace_id"])


if __name__ == "__main__":
    unittest.main()

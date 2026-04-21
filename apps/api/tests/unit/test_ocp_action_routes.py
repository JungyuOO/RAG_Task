from __future__ import annotations

import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.actions import router
from apps.api.runtime import action_preview_service, connection_broker
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest


class OcpActionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        connection_broker.profile_store.clear()
        connection_broker.secret_store.clear()
        action_preview_service.transport = None
        self.client = TestClient(self.app)

    def test_preview_route_returns_action_preview(self) -> None:
        profile = connection_broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        response = self.client.post(
            "/api/v1/actions/preview",
            json={
                "connection_id": profile.connection_id,
                "action_type": "scale_deployment",
                "resource_name": "demo-app",
                "replicas": 2,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("oc scale", response.json()["preview_command"])

    def test_yaml_apply_preview_route_returns_dry_run_metadata(self) -> None:
        profile = connection_broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "10"},
                        "spec": {"replicas": 2},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "10"},
                    "spec": {"replicas": 3},
                },
            )

        action_preview_service.transport = httpx.MockTransport(handler)
        response = self.client.post(
            "/api/v1/actions/preview",
            json={
                "connection_id": profile.connection_id,
                "actor_roles": ["admin"],
                "action_type": "yaml_apply",
                "namespace": "demo",
                "resource_name": "web",
                "reason": "bump replicas",
                "manifest_yaml": (
                    "apiVersion: apps/v1\n"
                    "kind: Deployment\n"
                    "metadata:\n"
                    "  name: web\n"
                    "  namespace: demo\n"
                    "spec:\n"
                    "  replicas: 3\n"
                ),
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["dry_run_status"], "ok")
        self.assertEqual(payload["approval_strategy"], "single_approval")
        self.assertIn('"replicas": 3', payload["diff_unified"])


if __name__ == "__main__":
    unittest.main()




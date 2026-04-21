from __future__ import annotations

import json
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

from apps.api.routes.auth import _broker, _verifier, router
from apps.api.ocp.auth.verifier import OcpConnectionVerifier


class OcpAuthRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1/auth")
        self.client = TestClient(self.app)
        _broker.profile_store.clear()
        _broker.secret_store.clear()
        _verifier.transport = None

    def test_connect_test_disconnect_flow(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/namespaces":
                return httpx.Response(200, json={"items": [{"metadata": {"name": "demo"}}]})
            if request.url.path == "/apis/user.openshift.io/v1/users/~":
                return httpx.Response(200, json={"metadata": {"name": "alice"}, "groups": ["dev-team"]})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews":
                review = json.loads(request.content.decode("utf-8"))
                resource = (((review.get("spec") or {}).get("resourceAttributes") or {}).get("resource") or "")
                if resource == "deployments":
                    return httpx.Response(201, json={"status": {"allowed": True}})
                if resource == "rolebindings":
                    return httpx.Response(201, json={"status": {"allowed": False}})
                return httpx.Response(201, json={"status": {"allowed": False}})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews":
                return httpx.Response(
                    201,
                    json={
                        "status": {
                            "incomplete": True,
                            "evaluationError": "webhook authorizer timeout",
                            "resourceRules": [
                                {
                                    "verbs": ["patch"],
                                    "apiGroups": ["route.openshift.io"],
                                    "resources": ["routes"],
                                }
                            ]
                        }
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        _verifier.transport = httpx.MockTransport(handler)
        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "workspace_id": "workspace-customer-a",
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
                "display_name": "demo-cluster",
            },
        )
        self.assertEqual(connect_response.status_code, 200)
        payload = connect_response.json()
        self.assertTrue(payload["connected"])
        self.assertEqual(payload["connection"]["display_name"], "demo-cluster")
        self.assertEqual(payload["connection"]["workspace_id"], "workspace-customer-a")
        self.assertNotIn("token", payload["connection"])

        connection_id = payload["connection"]["connection_id"]
        test_response = self.client.post(
            "/api/v1/auth/ocp/test",
            json={"connection_id": connection_id},
        )
        self.assertEqual(test_response.status_code, 200)
        test_payload = test_response.json()
        self.assertTrue(test_payload["success"])
        self.assertEqual(test_payload["resolved_user"], "alice")
        self.assertEqual(test_payload["resolved_roles"], ["operator", "viewer"])
        self.assertTrue(test_payload["permission_hints"]["can_patch_deployments"])
        self.assertTrue(test_payload["permission_hints"]["can_manage_routes"])
        self.assertTrue(test_payload["rbac_rules_incomplete"])
        self.assertEqual(test_payload["rbac_evaluation_error"], "webhook authorizer timeout")
        self.assertEqual(test_payload["secret_backend"], "protected_file")
        self.assertEqual(test_payload["resolved_namespace"], "demo")

        refresh_response = self.client.post(
            "/api/v1/auth/ocp/lease/refresh",
            json={"connection_id": connection_id},
        )
        self.assertEqual(refresh_response.status_code, 200)
        refresh_payload = refresh_response.json()
        self.assertEqual(refresh_payload["secret_backend"], "protected_file")
        self.assertFalse(refresh_payload["secret_lease_renewable"])

        disconnect_response = self.client.post(
            "/api/v1/auth/ocp/disconnect",
            json={"connection_id": connection_id},
        )
        self.assertEqual(disconnect_response.status_code, 200)
        disconnect_payload = disconnect_response.json()
        self.assertFalse(disconnect_payload["connected"])

    def test_test_endpoint_404s_for_missing_connection(self) -> None:
        response = self.client.post(
            "/api/v1/auth/ocp/test",
            json={"connection_id": "missing"},
        )
        self.assertEqual(response.status_code, 404)

    def test_password_mode_test_returns_failure_payload(self) -> None:
        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "password",
                "username": "developer",
                "password": "secret",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        test_response = self.client.post(
            "/api/v1/auth/ocp/test",
            json={"connection_id": connection_id},
        )
        self.assertEqual(test_response.status_code, 200)
        payload = test_response.json()
        self.assertFalse(payload["success"])
        self.assertIn("token exchange", payload["error"].lower())

    def test_lease_status_endpoint_returns_scheduler_status(self) -> None:
        response = self.client.get("/api/v1/auth/ocp/lease/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("enabled", payload)
        self.assertIn("interval_seconds", payload)
        self.assertIn("alert_level", payload)
        self.assertIn("recent_failures", payload)


if __name__ == "__main__":
    unittest.main()





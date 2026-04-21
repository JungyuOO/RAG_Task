from __future__ import annotations

import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.actions import router
from apps.api.runtime import action_execution_service, action_request_service, connection_broker
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest


class OcpActionExecutionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        connection_broker.profile_store.clear()
        connection_broker.secret_store.clear()
        action_request_service.repository.clear()
        action_execution_service.repository.clear()
        action_request_service.preview_service.transport = None
        action_execution_service.transport = None
        self.client = TestClient(self.app)

    def test_execute_route_and_execution_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"spec": {"replicas": 2}, "status": {"readyReplicas": 2}})
            if request.method == "PATCH":
                return httpx.Response(200, json={"spec": {"replicas": 2}})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        action_execution_service.transport = httpx.MockTransport(handler)
        profile = connection_broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        created = self.client.post(
            "/api/v1/actions/requests",
            json={
                "connection_id": profile.connection_id,
                "actor_roles": ["operator"],
                "action_type": "scale_deployment",
                "resource_name": "demo-app",
                "replicas": 2,
                "reason": "scale out",
            },
        ).json()
        request_id = created["request_id"]

        self.client.post(
            f"/api/v1/actions/requests/{request_id}/approve",
            json={"decision_note": "approved-stage-1", "actor_id": "approver-a", "actor_roles": ["operator"]},
        )
        self.client.post(
            f"/api/v1/actions/requests/{request_id}/approve",
            json={"decision_note": "approved-stage-2", "actor_id": "approver-b", "actor_roles": ["admin"]},
        )
        execute_response = self.client.post(
            f"/api/v1/actions/requests/{request_id}/execute",
            json={"actor_id": "ui-local", "actor_roles": ["admin"], "execution_note": "dry-run from UI"},
        )
        self.assertEqual(execute_response.status_code, 200)
        self.assertEqual(execute_response.json()["status"], "succeeded")
        self.assertGreaterEqual(len(execute_response.json()["preflight_checks"]), 1)

        executions_response = self.client.get("/api/v1/actions/executions?limit=10")
        self.assertEqual(executions_response.status_code, 200)
        self.assertGreaterEqual(len(executions_response.json()["items"]), 1)

    def test_yaml_apply_request_approve_execute_route_flow(self) -> None:
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
            if request.method == "PATCH" and request.url.params.get("dryRun") == "All":
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "10"},
                        "spec": {"replicas": 3},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "11"},
                    "spec": {"replicas": 3},
                },
            )

        action_request_service.preview_service.transport = httpx.MockTransport(handler)
        action_execution_service.transport = httpx.MockTransport(handler)

        profile = connection_broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        created = self.client.post(
            "/api/v1/actions/requests",
            json={
                "connection_id": profile.connection_id,
                "actor_roles": ["admin"],
                "action_type": "yaml_apply",
                "namespace": "demo",
                "resource_name": "web",
                "reason": "apply deployment",
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
        self.assertEqual(created.status_code, 200)
        request_id = created.json()["request_id"]

        approve = self.client.post(
            f"/api/v1/actions/requests/{request_id}/approve",
            json={"decision_note": "approved", "actor_id": "approver-a", "actor_roles": ["admin"]},
        )
        self.assertEqual(approve.status_code, 200)
        self.assertEqual(approve.json()["status"], "approved")

        execute = self.client.post(
            f"/api/v1/actions/requests/{request_id}/execute",
            json={"actor_id": "ui-local", "actor_roles": ["admin"], "execution_note": "apply from UI"},
        )
        self.assertEqual(execute.status_code, 200)
        self.assertEqual(execute.json()["status"], "succeeded")
        self.assertEqual(execute.json()["execution_mode"], "real")


if __name__ == "__main__":
    unittest.main()




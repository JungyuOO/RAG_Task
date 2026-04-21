from __future__ import annotations

import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.auth import _broker
from apps.api.routes.auth import router as auth_router
from apps.api.routes.ocp import router as ocp_router
from apps.api.runtime import connected_ocp_service


class OcpLiveRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(auth_router, prefix="/api/v1/auth")
        self.app.include_router(ocp_router, prefix="/api/v1")
        self.client = TestClient(self.app)
        _broker.profile_store.clear()
        _broker.secret_store.clear()
        connected_ocp_service.transport = None

    def test_namespaces_resources_and_overview_flow(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/api/v1/namespaces"):
                return httpx.Response(
                    200,
                    json={"items": [{"metadata": {"name": "demo"}}, {"metadata": {"name": "openshift-monitoring"}}]},
                )
            if path.endswith("/api/v1/namespaces/demo/pods"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "pod-a", "namespace": "demo"}, "status": {"phase": "Running"}, "spec": {"nodeName": "node-1"}}]})
            if path.endswith("/api/v1/namespaces/demo/pods/pod-a"):
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "v1",
                        "kind": "Pod",
                        "metadata": {"name": "pod-a", "namespace": "demo"},
                        "spec": {"nodeName": "node-1"},
                        "status": {"phase": "Running"},
                    },
                )
            if path.endswith("/apis/apps/v1/namespaces/demo/deployments"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "deploy-a", "namespace": "demo"}, "status": {"readyReplicas": 1}, "spec": {"replicas": 1}}]})
            if path.endswith("/api/v1/namespaces/demo/services"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "svc-a", "namespace": "demo"}, "spec": {"type": "ClusterIP", "clusterIP": "10.0.0.1"}}]})
            if path.endswith("/apis/route.openshift.io/v1/namespaces/demo/routes"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "route-a", "namespace": "demo"}, "spec": {"host": "app.example.com", "to": {"name": "svc-a"}}}]})
            if path.endswith("/api/v1/namespaces/demo/events"):
                return httpx.Response(200, json={"items": [{"metadata": {"name": "event-a", "namespace": "demo"}, "type": "Warning", "reason": "BackOff", "involvedObject": {"name": "pod-a", "kind": "Pod"}}]})
            return httpx.Response(404, json={"message": "not found"})

        connected_ocp_service.transport = httpx.MockTransport(handler)
        connect_response = self.client.post(
            "/api/v1/auth/ocp/connect",
            json={
                "cluster_url": "https://api.cluster.example.com",
                "auth_mode": "token",
                "token": "sha256~abc",
                "default_namespace": "demo",
            },
        )
        connection_id = connect_response.json()["connection"]["connection_id"]

        namespaces_response = self.client.get(f"/api/v1/ocp/namespaces/{connection_id}")
        self.assertEqual(namespaces_response.status_code, 200)
        self.assertEqual(namespaces_response.json()["count"], 2)

        resources_response = self.client.get(f"/api/v1/ocp/resources/{connection_id}?resource=pods&namespace=demo")
        self.assertEqual(resources_response.status_code, 200)
        self.assertEqual(resources_response.json()["items"][0]["name"], "pod-a")

        detail_response = self.client.get(f"/api/v1/ocp/resource-detail/{connection_id}?resource=pods&namespace=demo&name=pod-a")
        self.assertEqual(detail_response.status_code, 200)
        self.assertIn("kind: Pod", detail_response.json()["manifest_yaml"])

        overview_response = self.client.get(f"/api/v1/ocp/overview/{connection_id}")
        self.assertEqual(overview_response.status_code, 200)
        overview = overview_response.json()
        self.assertEqual(overview["resource_counts"]["pods"], 1)
        self.assertEqual(overview["resource_counts"]["deployments"], 1)

    def test_password_connection_cannot_access_live_resources_yet(self) -> None:
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
        response = self.client.get(f"/api/v1/ocp/namespaces/{connection_id}")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()



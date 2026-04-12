from __future__ import annotations

import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_ocp import router as ocp_router
from app.dependencies import get_container


class _DisabledOcpClient:
    enabled = False
    base_url = ""
    default_namespace = ""


class _EnabledOcpClient:
    enabled = True
    base_url = "https://api.example.com:6443"
    default_namespace = "demo"

    async def list_namespaces(self) -> dict:
        return {
            "count": 2,
            "items": ["alpha", "demo"],
        }

    async def list_resources(self, resource: str, namespace: str | None = None) -> dict:
        if resource == "events":
            return {
                "resource": resource,
                "namespace": namespace or "demo",
                "count": 1,
                "items": [
                    {
                        "name": "pod-crash",
                        "namespace": namespace or "demo",
                        "kind": "Event",
                        "created_at": "",
                        "phase": "BackOff",
                        "type": "Warning",
                        "host": "Pod",
                        "to": "demo-pod",
                    }
                ],
            }
        return {
            "resource": resource,
            "namespace": namespace or "demo",
            "count": 1,
            "items": [
                {
                    "name": "pod-a",
                    "namespace": namespace or "demo",
                    "kind": "Pod",
                    "created_at": "",
                    "phase": "Running",
                    "node_name": "worker-1",
                }
            ],
        }

    async def get_resource_yaml(self, resource: str, name: str, namespace: str | None = None) -> dict:
        return {
            "resource": resource,
            "namespace": namespace or "demo",
            "name": name,
            "object": {"kind": "Pod", "metadata": {"name": name}},
        }


class OcpRoutesTests(unittest.TestCase):
    def _build_app(self, ocp_client) -> FastAPI:
        app = FastAPI()
        app.include_router(ocp_router)
        app.dependency_overrides[get_container] = lambda: SimpleNamespace(ocp_api_client=ocp_client)
        return app

    def test_resources_route_returns_503_when_client_not_configured(self) -> None:
        app = self._build_app(_DisabledOcpClient())
        client = TestClient(app)
        response = client.get("/api/ocp/resources?resource=pods&namespace=demo")
        self.assertEqual(response.status_code, 503)

    def test_resources_route_returns_payload(self) -> None:
        app = self._build_app(_EnabledOcpClient())
        client = TestClient(app)
        response = client.get("/api/ocp/resources?resource=pods&namespace=demo")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resource"], "pods")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["name"], "pod-a")

    def test_events_route_returns_payload(self) -> None:
        app = self._build_app(_EnabledOcpClient())
        client = TestClient(app)
        response = client.get("/api/ocp/resources?resource=events&namespace=demo")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resource"], "events")
        self.assertEqual(payload["items"][0]["type"], "Warning")

    def test_status_route_returns_configuration_summary(self) -> None:
        app = self._build_app(_EnabledOcpClient())
        client = TestClient(app)
        response = client.get("/api/ocp/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["base_url"], "https://api.example.com:6443")
        self.assertEqual(payload["default_namespace"], "demo")

    def test_namespaces_route_returns_payload(self) -> None:
        app = self._build_app(_EnabledOcpClient())
        client = TestClient(app)
        response = client.get("/api/ocp/namespaces")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["items"], ["alpha", "demo"])

    def test_resource_yaml_route_returns_payload(self) -> None:
        app = self._build_app(_EnabledOcpClient())
        client = TestClient(app)
        response = client.get("/api/ocp/resource-yaml?resource=pods&namespace=demo&name=pod-a")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resource"], "pods")
        self.assertEqual(payload["name"], "pod-a")
        self.assertEqual(payload["object"]["kind"], "Pod")


if __name__ == "__main__":
    unittest.main()

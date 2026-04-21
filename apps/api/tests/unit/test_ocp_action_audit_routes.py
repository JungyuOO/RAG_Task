from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.actions import router
from apps.api.runtime import action_audit_service, action_request_service, connection_broker
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest


class OcpActionAuditRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        connection_broker.profile_store.clear()
        connection_broker.secret_store.clear()
        action_request_service.repository.clear()
        action_audit_service.repository.clear()
        self.client = TestClient(self.app)

    def test_audit_route_returns_created_request_audit(self) -> None:
        profile = connection_broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        self.client.post(
            "/api/v1/actions/requests",
            json={
                "connection_id": profile.connection_id,
                "action_type": "scale_deployment",
                "resource_name": "demo-app",
                "replicas": 2,
                "reason": "need more capacity",
                "actor_id": "ui-local",
                "actor_roles": ["operator"],
            },
        )
        response = self.client.get("/api/v1/actions/audit?limit=10")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(response.json()["items"]), 1)


if __name__ == "__main__":
    unittest.main()




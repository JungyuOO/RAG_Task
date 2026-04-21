from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.actions import router
from apps.api.runtime import action_request_service, connection_broker
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest


class OcpActionRequestRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        connection_broker.profile_store.clear()
        connection_broker.secret_store.clear()
        action_request_service.repository.clear()
        self.client = TestClient(self.app)

    def test_create_list_and_dual_approve_action_request(self) -> None:
        profile = connection_broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        create_response = self.client.post(
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
        self.assertEqual(create_response.status_code, 200)
        request_id = create_response.json()["request_id"]
        self.assertEqual(create_response.json()["required_approvals"], 1)

        list_response = self.client.get("/api/v1/actions/requests?limit=10")
        self.assertEqual(list_response.status_code, 200)
        self.assertGreaterEqual(len(list_response.json()["items"]), 1)

        approve_response = self.client.post(
            f"/api/v1/actions/requests/{request_id}/approve",
            json={"decision_note": "approved-stage-1", "actor_id": "approver-a", "actor_roles": ["operator"]},
        )
        self.assertEqual(approve_response.status_code, 200)
        self.assertEqual(approve_response.json()["status"], "approved")
        self.assertEqual(approve_response.json()["approval_count"], 1)

        reject_response = self.client.post(
            f"/api/v1/actions/requests/{request_id}/reject",
            json={"decision_note": "rejected", "actor_id": "ui-local", "actor_roles": ["admin"]},
        )
        self.assertEqual(reject_response.status_code, 200)
        self.assertEqual(reject_response.json()["status"], "rejected")

    def test_same_actor_cannot_double_approve_via_route(self) -> None:
        profile = connection_broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        create_response = self.client.post(
            "/api/v1/actions/requests",
            json={
                "connection_id": profile.connection_id,
                "action_type": "scale_deployment",
                "resource_name": "demo-app",
                "replicas": 2,
                "reason": "need more capacity",
                "actor_id": "requester-a",
                "actor_roles": ["operator"],
            },
        )
        request_id = create_response.json()["request_id"]

        first_approve = self.client.post(
            f"/api/v1/actions/requests/{request_id}/approve",
            json={"decision_note": "approved", "actor_id": "same-actor", "actor_roles": ["operator"]},
        )
        self.assertEqual(first_approve.status_code, 200)

        second_approve = self.client.post(
            f"/api/v1/actions/requests/{request_id}/approve",
            json={"decision_note": "approved-again", "actor_id": "same-actor", "actor_roles": ["operator"]},
        )
        self.assertEqual(second_approve.status_code, 400)


if __name__ == "__main__":
    unittest.main()




from __future__ import annotations

import unittest

from apps.api.schemas.ocp_action_requests import OcpActionRequestCreateRequest, OcpActionRequestDecisionRequest
from apps.api.schemas.ocp_actions import OcpActionType
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest
from apps.api.ocp.action_request_service import OcpActionRequestService
from apps.api.ocp.auth import OcpConnectionBroker


class OcpActionRequestServiceTests(unittest.TestCase):
    def test_create_dual_approval_then_reject_and_list(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        service = OcpActionRequestService()
        created = service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=3,
                reason="scale out",
            ),
            broker,
        )
        self.assertEqual(created.status, "pending")
        self.assertEqual(created.required_approvals, 1)
        self.assertEqual(created.approval_count, 0)
        self.assertEqual(created.requested_by, "ui")

        approved_once = service.approve(
            created.request_id,
            OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved by operator a"),
        )
        self.assertEqual(approved_once.status, "approved")
        self.assertEqual(approved_once.approval_count, 1)
        self.assertEqual(approved_once.decision_note, "approved by operator a")
        self.assertEqual(approved_once.approver_ids, ["operator-a"])

        listed = service.list_recent(limit=10)
        self.assertGreaterEqual(len(listed.items), 1)

        rejected = service.reject(created.request_id, OcpActionRequestDecisionRequest(actor_roles=["admin"], decision_note="rejected later"))
        self.assertEqual(rejected.status, "rejected")

    def test_create_rejects_blocked_policy_request(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="openshift-monitoring",
            )
        )
        service = OcpActionRequestService()
        with self.assertRaises(ValueError):
            service.create(
                OcpActionRequestCreateRequest(
                    connection_id=profile.connection_id,
                    actor_roles=["operator"],
                    action_type=OcpActionType.SCALE_DEPLOYMENT,
                    resource_name="prometheus",
                    replicas=20,
                    reason="",
                ),
                broker,
            )

    def test_same_actor_cannot_approve_twice(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        service = OcpActionRequestService()
        created = service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_id="requester-a",
                actor_roles=["operator"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=3,
                reason="scale out safely",
            ),
            broker,
        )

        service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="first"))
        with self.assertRaises(ValueError):
            service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="duplicate"))

    def test_create_rejects_missing_actor_roles(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        service = OcpActionRequestService()
        with self.assertRaises(ValueError):
            service.create(
                OcpActionRequestCreateRequest(
                    connection_id=profile.connection_id,
                    action_type=OcpActionType.SCALE_DEPLOYMENT,
                    resource_name="demo-app",
                    replicas=2,
                    reason="scale out",
                ),
                broker,
            )

    def test_break_glass_request_uses_admin_only_preview(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="prod-payments",
            )
        )
        broker.profile_store.put(
            profile.model_copy(
                update={
                    "metadata": {
                        "resolved_roles": ["admin"],
                        "permission_hints": {
                            "can_patch_deployments": True,
                            "has_cluster_admin_like_access": True,
                        },
                    }
                }
            )
        )

        service = OcpActionRequestService()
        created = service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="payments-api",
                replicas=12,
                reason="incident scale-out",
                break_glass=True,
                break_glass_reason="incident mitigation requires temporary emergency capacity increase",
                break_glass_ticket="INC-2048",
            ),
            broker,
        )

        self.assertTrue(created.preview.break_glass)
        self.assertEqual(created.preview.break_glass_ticket, "INC-2048")
        self.assertEqual(created.preview.requester_roles, ["admin"])


if __name__ == "__main__":
    unittest.main()





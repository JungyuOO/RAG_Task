from __future__ import annotations

import unittest

import httpx

from apps.api.schemas.ocp_actions import OcpActionPreviewRequest, OcpActionType
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest
from apps.api.ocp.action_preview_service import OcpActionPreviewService
from apps.api.ocp.auth import OcpConnectionBroker


class OcpActionPreviewServiceTests(unittest.TestCase):
    @staticmethod
    def _yaml_apply_manifest() -> str:
        return (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: web\n"
            "  namespace: demo\n"
            "spec:\n"
            "  replicas: 3\n"
        )

    @staticmethod
    def _yaml_apply_manifest_with_server_fields() -> str:
        return (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: web\n"
            "  namespace: demo\n"
            "  resourceVersion: \"10\"\n"
            "  managedFields:\n"
            "    - manager: kube-controller-manager\n"
            "spec:\n"
            "  replicas: 3\n"
            "status:\n"
            "  readyReplicas: 2\n"
        )

    def test_scale_preview(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=3,
                reason="scale out",
            ),
            broker,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.risk_level, "medium")
        self.assertEqual(result.required_approvals, 1)
        self.assertEqual(result.approval_strategy, "single_approval")
        self.assertIn("oc scale", result.preview_command)

    def test_restart_preview_uses_default_namespace(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="demo-app",
                reason="restart deployment",
            ),
            broker,
        )
        self.assertEqual(result.namespace, "demo")
        self.assertEqual(result.required_approvals, 2)
        self.assertIn("rollout restart", result.preview_command)

    def test_scale_preview_is_blocked_for_protected_namespace_without_reason(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="openshift-monitoring",
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="prometheus",
                replicas=20,
                reason="",
            ),
            broker,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.risk_level, "high")
        self.assertGreaterEqual(len(result.blocked_reasons), 1)

    def test_production_like_namespace_requires_longer_reason(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="prod-payments",
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="payments-api",
                reason="short",
            ),
            broker,
        )
        self.assertFalse(result.allowed)
        self.assertIn("10 characters", " ".join(result.blocked_reasons))

    def test_production_like_namespace_requires_admin_like_connection_rbac(self) -> None:
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
                            "has_cluster_admin_like_access": False,
                            "has_namespace_admin_like_access": False,
                            "can_manage_cluster_rolebindings": False,
                        },
                    }
                }
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="payments-api",
                reason="restart after validated production rollout issue",
            ),
            broker,
        )
        self.assertFalse(result.allowed)
        self.assertIn("admin-like RBAC evidence", " ".join(result.blocked_reasons))

    def test_production_like_namespace_allows_admin_like_connection_rbac(self) -> None:
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
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="payments-api",
                reason="restart after validated production rollout issue",
            ),
            broker,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.requester_roles, ["admin"])

    def test_production_like_namespace_allows_clusterrole_management_signal(self) -> None:
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
                            "has_cluster_admin_like_access": False,
                            "can_manage_clusterroles": True,
                        },
                    }
                }
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="payments-api",
                reason="restart after validated production rollout issue",
            ),
            broker,
        )
        self.assertTrue(result.allowed)

    def test_preview_blocks_requester_with_insufficient_roles(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["viewer"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=2,
                reason="scale safely now",
            ),
            broker,
        )
        self.assertFalse(result.allowed)
        self.assertIn("requester roles", " ".join(result.blocked_reasons))

    def test_preview_blocks_when_connection_permissions_do_not_allow_action(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        broker.profile_store.put(
            profile.model_copy(
                update={
                    "metadata": {
                        "resolved_roles": ["viewer"],
                        "permission_hints": {
                            "can_patch_deployments": False,
                            "can_get_pod_logs": True,
                        },
                        "rbac_rules_incomplete": True,
                        "rbac_evaluation_error": "webhook authorizer timeout",
                    }
                }
            )
        )
        service = OcpActionPreviewService()
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["viewer"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=2,
                reason="scale safely now",
            ),
            broker,
        )
        self.assertFalse(result.allowed)
        self.assertIn("deployment patch permission", " ".join(result.blocked_reasons))
        self.assertIn("RBAC evaluation note", " ".join(result.validation_messages))

    def test_break_glass_scale_preview_requires_admin_and_ticket(self) -> None:
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
        service = OcpActionPreviewService()
        blocked = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="payments-api",
                replicas=12,
                reason="incident scale-out",
                break_glass=True,
                break_glass_reason="too short",
            ),
            broker,
        )
        self.assertFalse(blocked.allowed)
        self.assertIn("ticket or incident reference", " ".join(blocked.blocked_reasons))

        allowed = service.build_preview(
            OcpActionPreviewRequest(
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
        self.assertTrue(allowed.allowed)
        self.assertTrue(allowed.break_glass)
        self.assertEqual(allowed.approval_strategy, "break_glass_dual_approval")
        self.assertIn("replica target >10 preview guardrail", " ".join(allowed.validation_messages))

    def test_yaml_apply_preview_ok(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
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
            self.assertEqual(request.method, "PATCH")
            self.assertEqual(request.url.params.get("dryRun"), "All")
            self.assertEqual(request.url.params.get("fieldManager"), "cywell-copilot")
            self.assertEqual(request.headers.get("Content-Type"), "application/apply-patch+yaml")
            return httpx.Response(
                200,
                json={
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "10"},
                    "spec": {"replicas": 3},
                },
            )

        service = OcpActionPreviewService(transport=httpx.MockTransport(handler))
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.YAML_APPLY,
                namespace="demo",
                resource_name="web",
                reason="bump replicas",
                metadata={"kind": "deployments"},
                manifest_yaml=self._yaml_apply_manifest(),
            ),
            broker,
        )

        self.assertTrue(result.allowed)
        self.assertEqual(result.dry_run_status, "ok")
        self.assertEqual(result.approval_strategy, "single_approval")
        self.assertIn('"replicas": 3', result.diff_unified)

    def test_yaml_apply_preview_rejected(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
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
            return httpx.Response(422, json={"message": "invalid manifest: spec.replicas must be >= 0"})

        service = OcpActionPreviewService(transport=httpx.MockTransport(handler))
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.YAML_APPLY,
                namespace="demo",
                resource_name="web",
                reason="bad edit",
                metadata={"kind": "deployments"},
                manifest_yaml=self._yaml_apply_manifest(),
            ),
            broker,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.dry_run_status, "rejected")
        self.assertTrue(any("replicas must be" in item for item in result.dry_run_messages))

    def test_yaml_apply_preview_strips_server_managed_fields(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
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
            self.assertEqual(request.method, "PATCH")
            body = request.content.decode("utf-8")
            self.assertNotIn("managedFields", body)
            self.assertNotIn("resourceVersion", body)
            self.assertNotIn("status:", body)
            return httpx.Response(
                200,
                json={
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "11"},
                    "spec": {"replicas": 3},
                },
            )

        service = OcpActionPreviewService(transport=httpx.MockTransport(handler))
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.YAML_APPLY,
                namespace="demo",
                resource_name="web",
                reason="sanitize manifest",
                metadata={"kind": "deployments"},
                manifest_yaml=self._yaml_apply_manifest_with_server_fields(),
            ),
            broker,
        )

        self.assertTrue(result.allowed)
        self.assertIn("removed automatically", " ".join(result.validation_messages))

    def test_yaml_apply_preview_stringifies_annotations_and_env_values(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        manifest_yaml = (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: web\n"
            "  namespace: demo\n"
            "  annotations:\n"
            "    deployment.kubernetes.io/revision: 1\n"
            "spec:\n"
            "  replicas: 2\n"
            "  template:\n"
            "    metadata:\n"
            "      labels:\n"
            "        app: web\n"
            "    spec:\n"
            "      containers:\n"
            "        - name: app\n"
            "          image: nginx\n"
            "          env:\n"
            "            - name: K_SINK_TIMEOUT\n"
            "              value: 30\n"
            "            - name: METRICS_PROMETHEUS_PORT\n"
            "              value: 9000\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "10"},
                        "spec": {"replicas": 1},
                    },
                )
            body = request.content.decode("utf-8")
            self.assertIn("deployment.kubernetes.io/revision: '1'", body)
            self.assertIn("value: '30'", body)
            self.assertIn("value: '9000'", body)
            return httpx.Response(
                200,
                json={
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "11"},
                    "spec": {"replicas": 2},
                },
            )

        service = OcpActionPreviewService(transport=httpx.MockTransport(handler))
        result = service.build_preview(
            OcpActionPreviewRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.YAML_APPLY,
                namespace="demo",
                resource_name="web",
                reason="string normalization",
                metadata={"kind": "deployments"},
                manifest_yaml=manifest_yaml,
            ),
            broker,
        )

        self.assertTrue(result.allowed)


if __name__ == "__main__":
    unittest.main()





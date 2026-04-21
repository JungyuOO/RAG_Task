from __future__ import annotations

import unittest
import httpx

from apps.api.schemas.ocp_action_execute import OcpActionExecuteRequest
from apps.api.schemas.ocp_action_requests import OcpActionRequestCreateRequest, OcpActionRequestDecisionRequest
from apps.api.schemas.ocp_actions import OcpActionType
from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest
from apps.api.ocp.action_execution_service import OcpActionExecutionService
from apps.api.ocp.action_preview_service import OcpActionPreviewService
from apps.api.ocp.action_request_service import OcpActionRequestService
from apps.api.ocp.auth import OcpConnectionBroker


class OcpActionExecutionServiceTests(unittest.TestCase):
    def test_execute_only_after_approval(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        request_service = OcpActionRequestService()
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=2,
                reason="scale out",
            ),
            broker,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"spec": {"replicas": 2}, "status": {"readyReplicas": 2}})
            if request.method == "PATCH":
                return httpx.Response(200, json={"spec": {"replicas": 2}})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(ValueError):
            execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))

        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved-a"))
        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))
        self.assertEqual(executed.status, "succeeded")
        self.assertFalse(executed.simulated)
        self.assertEqual(executed.execution_mode, "real")
        self.assertGreaterEqual(len(executed.preflight_checks), 1)
        self.assertIn("API PATCH succeeded", executed.output_lines[0])

    def test_execute_scale_uses_real_dry_run(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                self.assertIn("/deployments/demo-app", str(request.url))
                return httpx.Response(200, json={"spec": {"replicas": 2}, "status": {"readyReplicas": 2}})
            self.assertEqual(request.method, "PATCH")
            self.assertNotIn("dryRun=All", str(request.url))
            return httpx.Response(200, json={"spec": {"replicas": 3}})

        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        request_service = OcpActionRequestService()
        created = request_service.create(
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
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved-a"))

        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )
        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))
        self.assertFalse(executed.simulated)
        self.assertEqual(executed.execution_mode, "real")
        self.assertIn("target deployment/demo-app exists in namespace demo", executed.preflight_checks[0])
        self.assertIn("Server-confirmed replicas: 3", executed.output_lines[-1])

    def test_execute_scale_blocks_large_replica_delta(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"spec": {"replicas": 0}, "status": {"readyReplicas": 0}})
            raise AssertionError("PATCH should not run when preflight blocks the request")

        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        request_service = OcpActionRequestService()
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=6,
                reason="big scale jump",
            ),
            broker,
        )
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved-a"))
        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )

        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))
        self.assertEqual(executed.status, "failed")
        self.assertIn("Replica delta above 5", executed.error)
        self.assertGreaterEqual(len(executed.preflight_checks), 3)

    def test_break_glass_execute_allows_large_replica_delta_dry_run(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "metadata": {"generation": 1},
                        "spec": {"replicas": 0},
                        "status": {"observedGeneration": 1, "readyReplicas": 0, "availableReplicas": 0, "updatedReplicas": 0},
                    },
                )
            if request.method == "PATCH":
                return httpx.Response(200, json={"spec": {"replicas": 12}})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

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
        request_service = OcpActionRequestService()
        created = request_service.create(
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
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="admin-a", actor_roles=["admin"], decision_note="approved-a"))
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="admin-b", actor_roles=["admin"], decision_note="approved-b"))

        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )
        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))

        self.assertEqual(executed.status, "succeeded")
        self.assertTrue(any("break-glass" in item for item in executed.preflight_checks))
        self.assertIn("Break-glass override acknowledged", executed.output_lines[0])

    def test_execute_restart_blocks_paused_deployment(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"spec": {"paused": True, "replicas": 2}, "status": {"readyReplicas": 2}})
            raise AssertionError("PATCH should not run when paused deployment blocks restart")

        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        request_service = OcpActionRequestService()
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="demo-app",
                reason="restart for refresh",
            ),
            broker,
        )
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved-a"))
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-b", actor_roles=["admin"], decision_note="approved-b"))
        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )

        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))
        self.assertEqual(executed.status, "failed")
        self.assertIn("Paused deployments are blocked", executed.error)
        self.assertIn("paused=yes", " ".join(executed.preflight_checks))

    def test_execute_restart_blocks_unhealthy_deployment(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and "/deployments/demo-app" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "metadata": {"generation": 3},
                        "spec": {"paused": False, "replicas": 2},
                        "status": {
                            "observedGeneration": 3,
                            "readyReplicas": 1,
                            "availableReplicas": 1,
                            "unavailableReplicas": 1,
                            "updatedReplicas": 2,
                            "conditions": [
                                {"type": "Available", "status": "False", "reason": "MinimumReplicasUnavailable"},
                                {"type": "Progressing", "status": "True", "reason": "ReplicaSetUpdated"},
                            ],
                        },
                    },
                )
            if request.method == "GET" and "/poddisruptionbudgets" in str(request.url):
                return httpx.Response(200, json={"items": []})
            raise AssertionError("PATCH should not run when unhealthy deployment blocks restart")

        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        request_service = OcpActionRequestService()
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="demo-app",
                reason="restart for refresh",
            ),
            broker,
        )
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved-a"))
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-b", actor_roles=["admin"], decision_note="approved-b"))
        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )

        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))
        self.assertEqual(executed.status, "failed")
        self.assertIn("not currently healthy", executed.error)
        self.assertIn("unavailable=1", " ".join(executed.preflight_checks))

    def test_execute_restart_blocks_when_matching_pdb_has_zero_disruptions(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and "/deployments/demo-app" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "metadata": {"generation": 1},
                        "spec": {
                            "paused": False,
                            "replicas": 2,
                            "selector": {"matchLabels": {"app": "demo-app"}},
                            "template": {"metadata": {"labels": {"app": "demo-app"}}},
                        },
                        "status": {
                            "observedGeneration": 1,
                            "readyReplicas": 2,
                            "availableReplicas": 2,
                            "unavailableReplicas": 0,
                            "updatedReplicas": 2,
                            "conditions": [
                                {"type": "Available", "status": "True", "reason": "MinimumReplicasAvailable"},
                                {"type": "Progressing", "status": "True", "reason": "NewReplicaSetAvailable"},
                            ],
                        },
                    },
                )
            if request.method == "GET" and "/poddisruptionbudgets" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "metadata": {"name": "demo-pdb"},
                                "spec": {"selector": {"matchLabels": {"app": "demo-app"}}},
                                "status": {"disruptionsAllowed": 0},
                            }
                        ]
                    },
                )
            raise AssertionError("PATCH should not run when PDB blocks restart")

        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        request_service = OcpActionRequestService()
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.ROLLOUT_RESTART,
                resource_name="demo-app",
                reason="restart for refresh",
            ),
            broker,
        )
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved-a"))
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-b", actor_roles=["admin"], decision_note="approved-b"))
        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )

        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))
        self.assertEqual(executed.status, "failed")
        self.assertIn("zero disruptions", executed.error)
        self.assertIn("demo-pdb", " ".join(executed.preflight_checks))

    def test_execute_blocks_insufficient_executor_role(self) -> None:
        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        request_service = OcpActionRequestService()
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["operator"],
                action_type=OcpActionType.SCALE_DEPLOYMENT,
                resource_name="demo-app",
                replicas=2,
                reason="scale out",
            ),
            broker,
        )
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="operator-a", actor_roles=["operator"], decision_note="approved-a"))

        execution_service = OcpActionExecutionService(request_service=request_service, broker=broker)
        with self.assertRaises(ValueError):
            execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["viewer"]))

    def test_execute_yaml_apply_real_patch(self) -> None:
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, str(request.url)))
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
            if request.method == "PATCH":
                self.assertEqual(request.url.params.get("fieldManager"), "cywell-copilot")
                self.assertIsNone(request.url.params.get("dryRun"))
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
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        preview_service = OcpActionPreviewService(transport=httpx.MockTransport(handler))
        request_service = OcpActionRequestService(preview_service=preview_service)
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.YAML_APPLY,
                namespace="demo",
                resource_name="web",
                reason="apply deployment",
                manifest_yaml=(
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
                ),
            ),
            broker,
        )
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="admin-a", actor_roles=["admin"], decision_note="approved"))

        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )
        executed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))

        self.assertEqual(executed.status, "succeeded")
        self.assertEqual(executed.execution_mode, "real")
        self.assertIn("API PATCH succeeded", executed.output_lines[0])

    def test_execute_yaml_apply_force_after_conflict(self) -> None:
        state = {"preview_patch": 0, "execute_patch": 0}

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
                state["preview_patch"] += 1
                return httpx.Response(
                    200,
                    json={
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "10"},
                        "spec": {"replicas": 3},
                    },
                )
            state["execute_patch"] += 1
            if state["execute_patch"] == 1:
                return httpx.Response(409, json={"message": "field ownership conflict"})
            self.assertEqual(request.url.params.get("force"), "true")
            return httpx.Response(
                200,
                json={
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "web", "namespace": "demo", "resourceVersion": "11"},
                    "spec": {"replicas": 3},
                },
            )

        broker = OcpConnectionBroker()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )
        preview_service = OcpActionPreviewService(transport=httpx.MockTransport(handler))
        request_service = OcpActionRequestService(preview_service=preview_service)
        created = request_service.create(
            OcpActionRequestCreateRequest(
                connection_id=profile.connection_id,
                actor_roles=["admin"],
                action_type=OcpActionType.YAML_APPLY,
                namespace="demo",
                resource_name="web",
                reason="apply deployment",
                manifest_yaml=(
                    "apiVersion: apps/v1\n"
                    "kind: Deployment\n"
                    "metadata:\n"
                    "  name: web\n"
                    "  namespace: demo\n"
                    "spec:\n"
                    "  replicas: 3\n"
                ),
            ),
            broker,
        )
        request_service.approve(created.request_id, OcpActionRequestDecisionRequest(actor_id="admin-a", actor_roles=["admin"], decision_note="approved"))

        execution_service = OcpActionExecutionService(
            request_service=request_service,
            broker=broker,
            transport=httpx.MockTransport(handler),
        )
        failed = execution_service.execute(created.request_id, OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"]))
        self.assertEqual(failed.status, "failed")
        self.assertIn("field ownership conflict", failed.error)

        succeeded = execution_service.execute(
            created.request_id,
            OcpActionExecuteRequest(actor_id="ui-local", actor_roles=["admin"], force=True),
        )
        self.assertEqual(succeeded.status, "succeeded")
        self.assertIn("Force apply: yes", succeeded.output_lines[-1])


if __name__ == "__main__":
    unittest.main()





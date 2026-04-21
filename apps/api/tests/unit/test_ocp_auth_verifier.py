from __future__ import annotations

import json
import unittest

import httpx

from apps.api.schemas.ocp_auth import OcpAuthMode, OcpConnectionRequest
from apps.api.ocp.auth import OcpConnectionBroker, OcpConnectionVerifier


class OcpAuthVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_token_success(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["Authorization"], "Bearer sha256~abc")
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
                                },
                                {
                                    "verbs": ["create"],
                                    "apiGroups": [""],
                                    "resources": ["pods/exec"],
                                },
                                {
                                    "verbs": ["get"],
                                    "apiGroups": [""],
                                    "resources": ["secrets"],
                                },
                            ]
                        }
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        broker = OcpConnectionBroker()
        verifier = OcpConnectionVerifier(transport=httpx.MockTransport(handler))
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
            )
        )

        result = await verifier.verify(profile, broker.build_runtime_config(profile), broker)

        self.assertTrue(result.success)
        self.assertEqual(result.resolved_user, "alice")
        self.assertEqual(result.identity_source, "user_api")
        self.assertEqual(result.resolved_groups, ["dev-team"])
        self.assertEqual(result.resolved_roles, ["operator", "viewer"])
        self.assertTrue(any(item.startswith("app_role_operator=") for item in result.rbac_evidence))
        self.assertTrue(result.permission_hints["can_patch_deployments"])
        self.assertTrue(result.permission_hints["can_manage_routes"])
        self.assertTrue(result.permission_hints["can_exec_pods"])
        self.assertTrue(result.permission_hints["can_read_secrets"])
        self.assertFalse(result.permission_hints["can_create_rolebindings"])
        self.assertGreaterEqual(len(result.rbac_evidence), 2)
        self.assertTrue(result.rbac_rules_incomplete)
        self.assertEqual(result.rbac_evaluation_error, "webhook authorizer timeout")
        self.assertEqual(result.secret_backend, "protected_file")
        self.assertFalse(result.secret_lease_renewable)
        self.assertEqual(result.resolved_namespace, "demo")

    async def test_verify_token_with_restricted_namespace_listing_is_still_success(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/namespaces":
                return httpx.Response(403, json={"message": "forbidden"})
            if request.url.path == "/apis/user.openshift.io/v1/users/~":
                return httpx.Response(200, json={"metadata": {"name": "viewer-user"}, "groups": ["viewers"]})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews":
                return httpx.Response(201, json={"status": {"allowed": False}})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews":
                return httpx.Response(201, json={"status": {"resourceRules": []}})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        broker = OcpConnectionBroker()
        verifier = OcpConnectionVerifier(transport=httpx.MockTransport(handler))
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~abc",
                default_namespace="demo",
            )
        )

        result = await verifier.verify(profile, broker.build_runtime_config(profile), broker)

        self.assertTrue(result.success)
        self.assertEqual(result.resolved_user, "viewer-user")
        self.assertEqual(result.resolved_roles, ["viewer"])
        self.assertFalse(result.permission_hints["can_patch_deployments"])
        self.assertFalse(result.rbac_rules_incomplete)
        self.assertEqual(result.resolved_namespace, "demo")
        self.assertIn("restricted", result.message.lower())

    async def test_verify_password_mode_returns_failure_until_exchange_hook_exists(self) -> None:
        broker = OcpConnectionBroker()
        verifier = OcpConnectionVerifier()
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.PASSWORD,
                username="developer",
                password="secret",
            )
        )

        result = await verifier.verify(profile, broker.build_runtime_config(profile), broker)

        self.assertFalse(result.success)
        self.assertIn("token exchange", result.error.lower())

    async def test_verify_token_detects_cluster_admin_like_rules(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/namespaces":
                return httpx.Response(200, json={"items": [{"metadata": {"name": "prod-core"}}]})
            if request.url.path == "/apis/user.openshift.io/v1/users/~":
                return httpx.Response(200, json={"metadata": {"name": "platform-admin"}, "groups": ["system:masters"]})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews":
                return httpx.Response(201, json={"status": {"allowed": True}})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews":
                return httpx.Response(
                    201,
                    json={
                        "status": {
                            "resourceRules": [
                                {
                                    "verbs": ["*"],
                                    "apiGroups": ["*"],
                                    "resources": ["*"],
                                }
                            ],
                            "nonResourceRules": [
                                {
                                    "verbs": ["*"],
                                    "nonResourceURLs": ["*"],
                                }
                            ],
                        }
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        broker = OcpConnectionBroker()
        verifier = OcpConnectionVerifier(transport=httpx.MockTransport(handler))
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~cluster-admin",
            )
        )

        result = await verifier.verify(profile, broker.build_runtime_config(profile), broker)

        self.assertTrue(result.success)
        self.assertEqual(result.resolved_roles, ["admin", "operator", "viewer"])
        self.assertTrue(any(item.startswith("app_role_admin=") for item in result.rbac_evidence))
        self.assertTrue(result.permission_hints["has_cluster_admin_like_access"])
        self.assertTrue(result.permission_hints["has_namespace_admin_like_access"])
        self.assertTrue(result.permission_hints["can_manage_clusterroles"])
        self.assertTrue(result.permission_hints["can_manage_cluster_rolebindings"])
        self.assertTrue(result.permission_hints["can_create_projects"])
        self.assertTrue(result.permission_hints["can_manage_configmaps"])
        self.assertTrue(result.permission_hints["can_read_pods"])

    async def test_verify_token_read_only_access_stays_viewer(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/namespaces":
                return httpx.Response(200, json={"items": [{"metadata": {"name": "team-a"}}]})
            if request.url.path == "/apis/user.openshift.io/v1/users/~":
                return httpx.Response(200, json={"metadata": {"name": "readonly-user"}, "groups": ["viewers"]})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews":
                review = json.loads(request.content.decode("utf-8"))
                attributes = ((review.get("spec") or {}).get("resourceAttributes") or {})
                resource = attributes.get("resource") or ""
                verb = attributes.get("verb") or ""
                if resource == "pods" and verb in {"get", "list"}:
                    return httpx.Response(201, json={"status": {"allowed": True}})
                if resource == "deployments" and verb in {"get", "list"}:
                    return httpx.Response(201, json={"status": {"allowed": True}})
                if resource == "services" and verb in {"get", "list"}:
                    return httpx.Response(201, json={"status": {"allowed": True}})
                if resource == "events" and verb == "list":
                    return httpx.Response(201, json={"status": {"allowed": True}})
                return httpx.Response(201, json={"status": {"allowed": False}})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews":
                return httpx.Response(
                    201,
                    json={
                        "status": {
                            "resourceRules": [
                                {
                                    "verbs": ["get", "list"],
                                    "apiGroups": ["apps"],
                                    "resources": ["deployments"],
                                },
                                {
                                    "verbs": ["get", "list"],
                                    "apiGroups": [""],
                                    "resources": ["services", "configmaps"],
                                },
                                {
                                    "verbs": ["get", "list"],
                                    "apiGroups": [""],
                                    "resources": ["serviceaccounts"],
                                },
                                {
                                    "verbs": ["get", "list"],
                                    "apiGroups": ["rbac.authorization.k8s.io"],
                                    "resources": ["roles"],
                                },
                                {
                                    "verbs": ["get", "list"],
                                    "apiGroups": ["route.openshift.io"],
                                    "resources": ["routes"],
                                },
                            ]
                        }
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        broker = OcpConnectionBroker()
        verifier = OcpConnectionVerifier(transport=httpx.MockTransport(handler))
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~readonly",
            )
        )

        result = await verifier.verify(profile, broker.build_runtime_config(profile), broker)

        self.assertTrue(result.success)
        self.assertEqual(result.resolved_roles, ["viewer"])
        self.assertTrue(result.permission_hints["can_read_deployments"])
        self.assertTrue(result.permission_hints["can_read_services"])
        self.assertTrue(result.permission_hints["can_read_serviceaccounts"])
        self.assertTrue(result.permission_hints["can_read_roles"])
        self.assertTrue(result.permission_hints["can_read_routes"])
        self.assertTrue(result.permission_hints["can_read_configmaps"])
        self.assertIn("app_role_viewer=read_only_workload_access", result.rbac_evidence)

    async def test_verify_token_detects_clusterrole_management_as_admin_like(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/namespaces":
                return httpx.Response(200, json={"items": [{"metadata": {"name": "team-a"}}]})
            if request.url.path == "/apis/user.openshift.io/v1/users/~":
                return httpx.Response(200, json={"metadata": {"name": "rbac-admin"}, "groups": ["platform-team"]})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews":
                return httpx.Response(201, json={"status": {"allowed": False}})
            if request.url.path == "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews":
                return httpx.Response(
                    201,
                    json={
                        "status": {
                            "resourceRules": [
                                {
                                    "verbs": ["create", "update"],
                                    "apiGroups": ["rbac.authorization.k8s.io"],
                                    "resources": ["clusterroles"],
                                }
                            ]
                        }
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        broker = OcpConnectionBroker()
        verifier = OcpConnectionVerifier(transport=httpx.MockTransport(handler))
        profile = broker.create_profile(
            OcpConnectionRequest(
                cluster_url="https://api.cluster.example.com",
                auth_mode=OcpAuthMode.TOKEN,
                token="sha256~rbac-admin",
            )
        )

        result = await verifier.verify(profile, broker.build_runtime_config(profile), broker)

        self.assertTrue(result.success)
        self.assertEqual(result.resolved_roles, ["admin", "operator", "viewer"])
        self.assertTrue(result.permission_hints["can_manage_clusterroles"])
        self.assertTrue(result.permission_hints["has_cluster_admin_like_access"])
        self.assertTrue(any(item.startswith("app_role_admin=") for item in result.rbac_evidence))


if __name__ == "__main__":
    unittest.main()





from __future__ import annotations

from typing import Any

import httpx

from apps.api.schemas.auth import OcpConnectionProfile, OcpConnectionTestResult
from apps.api.ocp.auth.broker import OcpConnectionBroker


class OcpConnectionVerifier:
    """Performs lightweight live verification for OCP connection profiles."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 10.0) -> None:
        self.transport = transport
        self.timeout = timeout

    async def verify(
        self,
        profile: OcpConnectionProfile,
        runtime: dict[str, Any],
        broker: OcpConnectionBroker,
    ) -> OcpConnectionTestResult:
        auth_mode = str(runtime.get("auth_mode") or "")
        if auth_mode == "token":
            return await self._verify_token(profile, runtime, broker)
        if auth_mode == "password":
            return broker.build_failure_result(
                profile,
                error="Password-based token exchange hook is not enabled yet. Use token mode for now.",
            )
        return broker.build_failure_result(
            profile,
            error="OAuth web flow is planned but not implemented yet.",
        )

    async def _verify_token(
        self,
        profile: OcpConnectionProfile,
        runtime: dict[str, Any],
        broker: OcpConnectionBroker,
    ) -> OcpConnectionTestResult:
        token = str(runtime.get("token") or "").strip()
        if not token:
            return broker.build_failure_result(profile, error="Missing bearer token.")

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                verify=profile.verify_ssl,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                namespace_response = await client.get(
                    f"{profile.cluster_url}/api/v1/namespaces?limit=1",
                    headers=headers,
                )
                identity = await self._resolve_identity(client, profile, headers)
        except httpx.HTTPError as exc:
            return broker.build_failure_result(profile, error=f"Network error while verifying cluster: {exc}")

        if namespace_response.status_code == 200:
            payload = namespace_response.json() if namespace_response.content else {}
            items = payload.get("items", []) if isinstance(payload, dict) else []
            discovered_namespace = ""
            if items:
                discovered_namespace = str(((items[0] or {}).get("metadata") or {}).get("name") or "")
            resolved_namespace = profile.default_namespace or discovered_namespace
            permission_hints, rbac_evidence, rules_metadata = await self._resolve_permission_hints(
                profile=profile,
                namespace=resolved_namespace,
                headers=headers,
                token_client_kwargs={
                    "verify": profile.verify_ssl,
                    "timeout": self.timeout,
                    "transport": self.transport,
                },
                can_read_namespaces=True,
            )
            resolved_roles, role_evidence = self._resolve_app_roles(
                resolved_groups=identity["resolved_groups"],
                permission_hints=permission_hints,
            )
            rbac_evidence.extend(role_evidence)
            secret_status = broker.describe_secret(profile, refresh=True, auto_renew=True)
            return broker.build_success_result(
                profile,
                resolved_user=identity["resolved_user"],
                resolved_groups=identity["resolved_groups"],
                resolved_roles=resolved_roles,
                identity_source=identity["identity_source"],
                permission_hints=permission_hints,
                rbac_evidence=rbac_evidence,
                rbac_rules_incomplete=rules_metadata["rbac_rules_incomplete"],
                rbac_evaluation_error=rules_metadata["rbac_evaluation_error"],
                secret_status=secret_status,
                resolved_namespace=resolved_namespace,
                message="Connected to cluster and verified namespace visibility.",
            )

        if namespace_response.status_code == 403:
            resolved_namespace = profile.default_namespace
            permission_hints, rbac_evidence, rules_metadata = await self._resolve_permission_hints(
                profile=profile,
                namespace=resolved_namespace,
                headers=headers,
                token_client_kwargs={
                    "verify": profile.verify_ssl,
                    "timeout": self.timeout,
                    "transport": self.transport,
                },
                can_read_namespaces=False,
            )
            resolved_roles, role_evidence = self._resolve_app_roles(
                resolved_groups=identity["resolved_groups"],
                permission_hints=permission_hints,
            )
            rbac_evidence.extend(role_evidence)
            secret_status = broker.describe_secret(profile, refresh=True, auto_renew=True)
            return broker.build_success_result(
                profile,
                resolved_user=identity["resolved_user"],
                resolved_groups=identity["resolved_groups"],
                resolved_roles=resolved_roles,
                identity_source=identity["identity_source"],
                permission_hints=permission_hints,
                rbac_evidence=rbac_evidence,
                rbac_rules_incomplete=rules_metadata["rbac_rules_incomplete"],
                rbac_evaluation_error=rules_metadata["rbac_evaluation_error"],
                secret_status=secret_status,
                resolved_namespace=resolved_namespace,
                message="Authenticated, but namespace listing is restricted. Connection is still usable.",
            )

        if namespace_response.status_code == 401:
            return broker.build_failure_result(profile, error="Authentication failed (401). Token may be invalid or expired.")

        return broker.build_failure_result(
            profile,
            error=f"Unexpected verification response: HTTP {namespace_response.status_code}",
        )

    async def _resolve_identity(
        self,
        client: httpx.AsyncClient,
        profile: OcpConnectionProfile,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        try:
            response = await client.get(f"{profile.cluster_url}/apis/user.openshift.io/v1/users/~", headers=headers)
        except httpx.HTTPError:
            return {
                "resolved_user": profile.username_hint,
                "resolved_groups": [],
                "identity_source": "username_hint",
            }

        if response.status_code != 200:
            return {
                "resolved_user": profile.username_hint,
                "resolved_groups": [],
                "identity_source": "username_hint",
            }

        payload = response.json() if response.content else {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        groups = payload.get("groups", []) if isinstance(payload, dict) else []
        return {
            "resolved_user": str(metadata.get("name") or profile.username_hint or ""),
            "resolved_groups": [str(group) for group in groups if str(group).strip()],
            "identity_source": "user_api",
        }

    async def _resolve_permission_hints(
        self,
        *,
        profile: OcpConnectionProfile,
        namespace: str,
        headers: dict[str, str],
        token_client_kwargs: dict[str, Any],
        can_read_namespaces: bool,
    ) -> tuple[dict[str, bool], list[str], dict[str, Any]]:
        permission_hints = {
            "can_read_namespaces": can_read_namespaces,
            "can_read_pods": False,
            "can_read_deployments": False,
            "can_read_services": False,
            "can_read_serviceaccounts": False,
            "can_patch_deployments": False,
            "can_get_pod_logs": False,
            "can_list_events": False,
            "can_create_rolebindings": False,
            "can_read_routes": False,
            "can_manage_routes": False,
            "can_exec_pods": False,
            "can_read_secrets": False,
            "can_read_configmaps": False,
            "can_manage_configmaps": False,
            "can_read_roles": False,
            "can_manage_roles": False,
            "can_read_rolebindings": False,
            "can_manage_rolebindings": False,
            "can_read_clusterroles": False,
            "can_manage_clusterroles": False,
            "can_bind_cluster_roles": False,
            "can_escalate_cluster_roles": False,
            "can_manage_cluster_rolebindings": False,
            "can_manage_serviceaccounts": False,
            "can_create_projects": False,
            "has_namespace_admin_like_access": False,
            "has_cluster_admin_like_access": False,
        }
        rbac_evidence = [
            f"namespace_visibility={'allowed' if can_read_namespaces else 'restricted'}",
        ]
        rule_hints, rule_metadata = await self._resolve_rule_hints(
            profile=profile,
            headers=headers,
            token_client_kwargs=token_client_kwargs,
            namespace=namespace,
        )
        permission_hints.update(rule_hints)
        if namespace:
            permission_hints["can_read_pods"] = await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="get",
                resource="pods",
                group="",
            ) or await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="list",
                resource="pods",
                group="",
            )
            permission_hints["can_read_deployments"] = permission_hints.get("can_read_deployments", False) or await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="get",
                resource="deployments",
                group="apps",
            ) or await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="list",
                resource="deployments",
                group="apps",
            )
            permission_hints["can_read_services"] = permission_hints.get("can_read_services", False) or await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="get",
                resource="services",
                group="",
            ) or await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="list",
                resource="services",
                group="",
            )
            permission_hints["can_patch_deployments"] = await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="patch",
                resource="deployments",
                group="apps",
            )
            permission_hints["can_get_pod_logs"] = await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="get",
                resource="pods/log",
                group="",
            )
            permission_hints["can_list_events"] = await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="list",
                resource="events",
                group="",
            )
            permission_hints["can_create_rolebindings"] = await self._check_access(
                profile=profile,
                headers=headers,
                token_client_kwargs=token_client_kwargs,
                namespace=namespace,
                verb="create",
                resource="rolebindings",
                group="rbac.authorization.k8s.io",
            )
        for key, value in permission_hints.items():
            rbac_evidence.append(f"{key}={'yes' if value else 'no'}")
        if rule_metadata["rbac_rules_incomplete"]:
            rbac_evidence.append("rules_review_incomplete=yes")
        if rule_metadata["rbac_evaluation_error"]:
            rbac_evidence.append(f"rules_review_error={rule_metadata['rbac_evaluation_error']}")
        return permission_hints, rbac_evidence, rule_metadata

    @staticmethod
    def _resolve_app_roles(
        *,
        resolved_groups: list[str],
        permission_hints: dict[str, bool],
    ) -> tuple[list[str], list[str]]:
        roles = {"viewer"}
        evidence = ["app_role_viewer=baseline"]
        normalized_groups = {group.casefold() for group in resolved_groups}
        admin_group_markers = {
            "cluster-admins",
            "admins",
            "system:cluster-admins",
            "system:masters",
            "dedicated-admins",
            "platform-admins",
            "ocp-admins",
        }
        if any(marker in normalized_groups for marker in admin_group_markers):
            roles.update({"operator", "admin"})
            evidence.extend(["app_role_operator=admin_group_marker", "app_role_admin=admin_group_marker"])
            return sorted(roles), evidence
        if (
            permission_hints.get("has_cluster_admin_like_access")
            or permission_hints.get("can_manage_clusterroles")
            or permission_hints.get("can_manage_cluster_rolebindings")
            or permission_hints.get("can_bind_cluster_roles")
            or permission_hints.get("can_escalate_cluster_roles")
            or permission_hints.get("can_create_projects")
        ):
            roles.update({"operator", "admin"})
            if permission_hints.get("has_cluster_admin_like_access"):
                signal = "cluster_admin_like_access"
            elif permission_hints.get("can_manage_clusterroles"):
                signal = "cluster_role_management"
            elif permission_hints.get("can_manage_cluster_rolebindings"):
                signal = "cluster_rolebinding_management"
            elif permission_hints.get("can_bind_cluster_roles"):
                signal = "cluster_role_bind"
            elif permission_hints.get("can_escalate_cluster_roles"):
                signal = "cluster_role_escalate"
            else:
                signal = "project_creation"
            evidence.extend([f"app_role_operator={signal}", f"app_role_admin={signal}"])
            return sorted(roles), evidence
        if (
            permission_hints.get("has_namespace_admin_like_access")
            or permission_hints.get("can_manage_rolebindings")
            or permission_hints.get("can_manage_roles")
        ):
            signal = (
                "namespace_admin_like_access"
                if permission_hints.get("has_namespace_admin_like_access")
                else "role_management"
                if permission_hints.get("can_manage_roles")
                else "rolebinding_management"
            )
            roles.update({"operator", "admin"})
            evidence.extend([f"app_role_operator={signal}", f"app_role_admin={signal}"])
            return sorted(roles), evidence

        operator_signals: list[str] = []
        if permission_hints.get("can_patch_deployments"):
            operator_signals.append("patch_deployments")
        if permission_hints.get("can_manage_routes"):
            operator_signals.append("manage_routes")
        if permission_hints.get("can_exec_pods"):
            operator_signals.append("exec_pods")
        if permission_hints.get("can_manage_configmaps"):
            operator_signals.append("manage_configmaps")
        if operator_signals:
            roles.add("operator")
            evidence.append(f"app_role_operator={'+'.join(operator_signals[:3])}")
            return sorted(roles), evidence

        if (
            permission_hints.get("can_read_pods")
            or permission_hints.get("can_read_deployments")
            or permission_hints.get("can_list_events")
            or permission_hints.get("can_read_services")
            or permission_hints.get("can_read_serviceaccounts")
            or permission_hints.get("can_read_roles")
            or permission_hints.get("can_read_clusterroles")
            or permission_hints.get("can_read_routes")
            or permission_hints.get("can_read_configmaps")
        ):
            evidence.append("app_role_viewer=read_only_workload_access")
        return sorted(roles), evidence

    async def _resolve_rule_hints(
        self,
        *,
        profile: OcpConnectionProfile,
        headers: dict[str, str],
        token_client_kwargs: dict[str, Any],
        namespace: str,
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        def empty_rule_hints() -> dict[str, bool]:
            return {
                "can_read_routes": False,
                "can_manage_routes": False,
                "can_exec_pods": False,
                "can_read_secrets": False,
                "can_read_configmaps": False,
                "can_manage_configmaps": False,
                "can_read_roles": False,
                "can_manage_roles": False,
                "can_read_rolebindings": False,
                "can_manage_rolebindings": False,
                "can_read_clusterroles": False,
                "can_manage_clusterroles": False,
                "can_bind_cluster_roles": False,
                "can_escalate_cluster_roles": False,
                "can_manage_cluster_rolebindings": False,
                "can_read_serviceaccounts": False,
                "can_manage_serviceaccounts": False,
                "can_create_projects": False,
                "can_read_deployments": False,
                "can_read_services": False,
                "has_namespace_admin_like_access": False,
                "has_cluster_admin_like_access": False,
            }

        if not namespace:
            return (
                empty_rule_hints(),
                {"rbac_rules_incomplete": False, "rbac_evaluation_error": ""},
            )

        review_payload = {
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectRulesReview",
            "spec": {
                "namespace": namespace,
            },
        }
        async with httpx.AsyncClient(**token_client_kwargs) as client:
            try:
                response = await client.post(
                    f"{profile.cluster_url}/apis/authorization.k8s.io/v1/selfsubjectrulesreviews",
                    headers={**headers, "Content-Type": "application/json"},
                    json=review_payload,
                )
            except httpx.HTTPError:
                return (
                    empty_rule_hints(),
                    {"rbac_rules_incomplete": True, "rbac_evaluation_error": "rules_review_network_error"},
                )
        if response.status_code not in {200, 201}:
            return (
                empty_rule_hints(),
                {"rbac_rules_incomplete": True, "rbac_evaluation_error": f"rules_review_http_{response.status_code}"},
            )
        payload = response.json() if response.content else {}
        status = payload.get("status", {}) if isinstance(payload, dict) else {}
        resource_rules = status.get("resourceRules", []) if isinstance(status, dict) else []
        non_resource_rules = status.get("nonResourceRules", []) if isinstance(status, dict) else []
        rules_incomplete = bool(status.get("incomplete") or False) if isinstance(status, dict) else False
        evaluation_error = str(status.get("evaluationError") or "") if isinstance(status, dict) else ""

        def has_rule(*, verb: str, resource: str, group: str = "") -> bool:
            for rule in resource_rules:
                verbs = [str(item).casefold() for item in rule.get("verbs", [])]
                resources = [str(item).casefold() for item in rule.get("resources", [])]
                api_groups = [str(item).casefold() for item in rule.get("apiGroups", [])]
                if "*" not in verbs and verb.casefold() not in verbs:
                    continue
                if "*" not in resources and resource.casefold() not in resources:
                    continue
                normalized_group = group.casefold()
                if "*" not in api_groups and normalized_group not in api_groups:
                    continue
                return True
            return False

        def has_non_resource_rule(*, verb: str, path: str = "*") -> bool:
            for rule in non_resource_rules:
                verbs = [str(item).casefold() for item in rule.get("verbs", [])]
                urls = [str(item).casefold() for item in rule.get("nonResourceURLs", [])]
                if "*" not in verbs and verb.casefold() not in verbs:
                    continue
                normalized_path = path.casefold()
                if "*" not in urls and normalized_path not in urls:
                    continue
                return True
            return False

        def has_any_rule(*, verbs: tuple[str, ...], resource: str, group: str = "") -> bool:
            return any(has_rule(verb=verb, resource=resource, group=group) for verb in verbs)

        read_verbs = ("get", "list", "watch")
        write_verbs = ("create", "patch", "update", "delete")

        can_read_configmaps = has_any_rule(verbs=read_verbs, resource="configmaps", group="")
        can_manage_configmaps = has_any_rule(verbs=write_verbs, resource="configmaps", group="")
        can_read_roles = has_any_rule(verbs=read_verbs, resource="roles", group="rbac.authorization.k8s.io")
        can_manage_roles = has_any_rule(verbs=write_verbs, resource="roles", group="rbac.authorization.k8s.io")
        can_read_rolebindings = has_any_rule(verbs=read_verbs, resource="rolebindings", group="rbac.authorization.k8s.io")
        can_manage_rolebindings = has_any_rule(verbs=write_verbs, resource="rolebindings", group="rbac.authorization.k8s.io")
        can_read_clusterroles = has_any_rule(verbs=read_verbs, resource="clusterroles", group="rbac.authorization.k8s.io")
        can_manage_clusterroles = has_any_rule(verbs=write_verbs, resource="clusterroles", group="rbac.authorization.k8s.io")
        can_bind_cluster_roles = has_rule(verb="bind", resource="clusterroles", group="rbac.authorization.k8s.io")
        can_escalate_cluster_roles = has_rule(verb="escalate", resource="clusterroles", group="rbac.authorization.k8s.io")
        can_manage_cluster_rolebindings = has_any_rule(
            verbs=write_verbs,
            resource="clusterrolebindings",
            group="rbac.authorization.k8s.io",
        )
        can_read_serviceaccounts = has_any_rule(verbs=read_verbs, resource="serviceaccounts", group="")
        can_manage_serviceaccounts = has_any_rule(verbs=write_verbs, resource="serviceaccounts", group="")
        can_create_projects = (
            has_rule(verb="create", resource="projectrequests", group="project.openshift.io")
            or has_rule(verb="create", resource="namespaces", group="")
        )
        has_cluster_admin_like_access = (
            has_rule(verb="*", resource="*", group="*")
            or can_manage_clusterroles
            or can_manage_cluster_rolebindings
            or can_bind_cluster_roles
            or can_escalate_cluster_roles
            or can_create_projects
            or has_non_resource_rule(verb="*", path="*")
        )
        can_read_routes = has_any_rule(verbs=read_verbs, resource="routes", group="route.openshift.io")
        can_manage_routes = has_any_rule(verbs=write_verbs, resource="routes", group="route.openshift.io")
        can_exec_pods = has_rule(verb="create", resource="pods/exec", group="")
        can_read_secrets = has_rule(verb="get", resource="secrets", group="") or has_rule(
            verb="list",
            resource="secrets",
            group="",
        )
        can_read_deployments = has_any_rule(verbs=read_verbs, resource="deployments", group="apps")
        can_read_services = has_any_rule(verbs=read_verbs, resource="services", group="")
        has_namespace_admin_like_access = (can_manage_rolebindings or can_manage_roles) and (
            can_manage_configmaps or can_read_secrets or can_manage_routes or can_exec_pods or can_manage_serviceaccounts
        )

        return (
            {
                "can_read_routes": can_read_routes,
                "can_read_configmaps": can_read_configmaps,
                "can_manage_configmaps": can_manage_configmaps,
                "can_read_roles": can_read_roles,
                "can_manage_roles": can_manage_roles,
                "can_read_rolebindings": can_read_rolebindings,
                "can_manage_rolebindings": can_manage_rolebindings,
                "can_read_clusterroles": can_read_clusterroles,
                "can_manage_clusterroles": can_manage_clusterroles,
                "can_bind_cluster_roles": can_bind_cluster_roles,
                "can_escalate_cluster_roles": can_escalate_cluster_roles,
                "can_manage_cluster_rolebindings": can_manage_cluster_rolebindings,
                "can_manage_routes": can_manage_routes,
                "can_exec_pods": can_exec_pods,
                "can_read_secrets": can_read_secrets,
                "can_read_serviceaccounts": can_read_serviceaccounts,
                "can_manage_serviceaccounts": can_manage_serviceaccounts,
                "can_create_projects": can_create_projects,
                "can_read_deployments": can_read_deployments,
                "can_read_services": can_read_services,
                "has_namespace_admin_like_access": has_namespace_admin_like_access,
                "has_cluster_admin_like_access": has_cluster_admin_like_access,
            },
            {
                "rbac_rules_incomplete": rules_incomplete,
                "rbac_evaluation_error": evaluation_error,
            },
        )

    async def _check_access(
        self,
        *,
        profile: OcpConnectionProfile,
        headers: dict[str, str],
        token_client_kwargs: dict[str, Any],
        namespace: str,
        verb: str,
        resource: str,
        group: str = "",
    ) -> bool:
        review_payload = {
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectAccessReview",
            "spec": {
                "resourceAttributes": {
                    "namespace": namespace,
                    "verb": verb,
                    "group": group,
                    "resource": resource,
                }
            },
        }
        async with httpx.AsyncClient(**token_client_kwargs) as client:
            try:
                response = await client.post(
                    f"{profile.cluster_url}/apis/authorization.k8s.io/v1/selfsubjectaccessreviews",
                    headers={**headers, "Content-Type": "application/json"},
                    json=review_payload,
                )
            except httpx.HTTPError:
                return False
        if response.status_code != 201 and response.status_code != 200:
            return False
        payload = response.json() if response.content else {}
        status = payload.get("status", {}) if isinstance(payload, dict) else {}
        return bool(status.get("allowed"))




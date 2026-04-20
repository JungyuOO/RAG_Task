from __future__ import annotations

import difflib
import json

import httpx
import yaml

from apps.api.api.schemas.actions import OcpActionPreviewRequest, OcpActionPreviewResponse, OcpActionType
from apps.api.integrations.ocp.manifest_sanitizer import dump_manifest_yaml, sanitize_manifest_for_apply
from apps.api.integrations.ocp.action_policy_service import OcpActionPolicyService
from apps.api.integrations.ocp.auth import OcpConnectionBroker


class OcpActionPreviewService:
    """Preview-only safe action planner for allowlisted OCP operations."""

    def __init__(
        self,
        *,
        policy_service: OcpActionPolicyService | None = None,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.policy_service = policy_service or OcpActionPolicyService()
        self.transport = transport
        self.timeout = timeout

    def build_preview(self, request: OcpActionPreviewRequest, broker: OcpConnectionBroker) -> OcpActionPreviewResponse:
        profile = broker.get_profile(request.connection_id)
        if profile is None:
            raise LookupError(f"Unknown connection_id={request.connection_id}")

        namespace = request.namespace or profile.default_namespace
        if not namespace:
            raise ValueError("namespace is required")

        connection_metadata = dict(profile.metadata or {})
        resolved_roles = [str(role) for role in connection_metadata.get("resolved_roles", []) if str(role).strip()]
        permission_hints = {
            str(key): bool(value)
            for key, value in dict(connection_metadata.get("permission_hints") or {}).items()
        }
        rbac_rules_incomplete = bool(connection_metadata.get("rbac_rules_incomplete") or False)
        rbac_evaluation_error = str(connection_metadata.get("rbac_evaluation_error") or "")

        validation_messages: list[str] = []
        summary = ""
        preview_command = ""
        risk_level = "low"
        next_step = "Preview only. Approval + execution wiring comes later."
        if request.break_glass:
            next_step = "Break-glass requested. Admin-only approval and audited dry-run execution are required."

        if request.action_type == OcpActionType.YAML_APPLY:
            runtime = broker.build_runtime_config(profile)
            if runtime.get("exchange_required"):
                raise ValueError("This connection requires token exchange and cannot run yaml_apply preview yet.")

            manifest = sanitize_manifest_for_apply(self._parse_manifest_yaml(request.manifest_yaml))
            sanitized_manifest_yaml = dump_manifest_yaml(manifest)
            manifest_kind = str(manifest.get("kind") or "").strip()
            metadata = dict(manifest.get("metadata") or {})
            manifest_namespace = str(metadata.get("namespace") or namespace).strip()
            manifest_name = str(metadata.get("name") or request.resource_name).strip()
            if not manifest_kind or not manifest_namespace or not manifest_name:
                raise ValueError("manifest_yaml must include kind, metadata.name, and metadata.namespace.")
            if request.namespace and manifest_namespace != request.namespace:
                raise ValueError("manifest namespace does not match request namespace.")
            if request.resource_name and manifest_name != request.resource_name:
                raise ValueError("manifest resource_name does not match request resource_name.")

            normalized_kind = self._normalize_kind(manifest_kind)
            request = request.model_copy(
                update={
                    "namespace": manifest_namespace,
                    "resource_name": manifest_name,
                    "metadata": {**dict(request.metadata or {}), "kind": normalized_kind},
                }
            )
            namespace = manifest_namespace

            baseline = self._get_resource(runtime, manifest_kind, namespace, manifest_name)
            dry_run_response = self._patch_yaml_apply_dry_run(
                runtime,
                manifest_kind,
                namespace,
                manifest_name,
                sanitized_manifest_yaml,
            )
            dry_run_status = "ok"
            dry_run_messages: list[str] = []
            allowed = True
            after = dry_run_response.json() if dry_run_response.content and dry_run_response.status_code < 400 else {}
            if dry_run_response.status_code >= 400:
                dry_run_status = "rejected"
                allowed = False
                message = self._extract_error_message(dry_run_response)
                dry_run_messages.append(message)
                validation_messages.append(message)
            diff_unified = self._build_unified_diff(baseline, after)
            summary = f"{namespace} namespace의 {normalized_kind}/{manifest_name} YAML apply preview입니다."
            preview_command = f"oc apply --server-side --dry-run=server -f - -n {namespace}"
            risk_level = "medium"
            next_step = "Review the dry-run diff, then create/approve/execute the yaml_apply request."

            if request.reason:
                validation_messages.append(f"Reason: {request.reason}")
            if sanitized_manifest_yaml != request.manifest_yaml:
                validation_messages.append("Server-managed manifest fields were removed automatically before dry-run.")
            if request.break_glass:
                validation_messages.append("Break-glass override requested for incident/emergency handling.")
                if request.break_glass_ticket:
                    validation_messages.append(f"Break-glass ticket: {request.break_glass_ticket}")
            if resolved_roles:
                validation_messages.append(f"Connection-backed roles: {', '.join(resolved_roles)}")
            if rbac_rules_incomplete:
                validation_messages.append("RBAC rules review was marked incomplete; final authorization is still confirmed by API dry-run/SSAR.")
            if rbac_evaluation_error:
                validation_messages.append(f"RBAC evaluation note: {rbac_evaluation_error}")

            preview = OcpActionPreviewResponse(
                connection_id=request.connection_id,
                action_type=request.action_type,
                namespace=namespace,
                resource_name=manifest_name,
                allowed=allowed,
                risk_level=risk_level,
                summary=summary,
                preview_command=preview_command,
                break_glass=request.break_glass,
                break_glass_reason=request.break_glass_reason,
                break_glass_ticket=request.break_glass_ticket,
                required_approvals=1,
                approval_strategy="single_approval",
                policy_checks=[],
                blocked_reasons=[],
                validation_messages=validation_messages,
                diff_unified=diff_unified,
                dry_run_status=dry_run_status,
                dry_run_messages=dry_run_messages,
                next_step=next_step,
            )
            return self.policy_service.apply(request, preview, connection_metadata=connection_metadata)

        if request.action_type == OcpActionType.SCALE_DEPLOYMENT:
            if not request.resource_name:
                raise ValueError("resource_name is required for scale_deployment")
            if request.replicas < 0:
                raise ValueError("replicas must be zero or positive")
            if permission_hints and not permission_hints.get("can_patch_deployments", False):
                validation_messages.append("Connected identity does not currently have deployment patch access in this namespace.")
            summary = f"{namespace} namespace의 deployment/{request.resource_name} 를 replicas={request.replicas} 로 조정할 예정입니다."
            preview_command = f"oc scale deployment/{request.resource_name} -n {namespace} --replicas={request.replicas}"
            validation_messages.append("Replica count will change workload capacity.")
            validation_messages.append("Use during low-traffic windows when possible.")
            risk_level = "medium"
        elif request.action_type == OcpActionType.ROLLOUT_RESTART:
            if not request.resource_name:
                raise ValueError("resource_name is required for rollout_restart")
            if permission_hints and not permission_hints.get("can_patch_deployments", False):
                validation_messages.append("Connected identity does not currently have deployment patch access in this namespace.")
            summary = f"{namespace} namespace의 deployment/{request.resource_name} 에 rollout restart를 수행할 예정입니다."
            preview_command = f"oc rollout restart deployment/{request.resource_name} -n {namespace}"
            validation_messages.append("Existing pods will be recreated in rollout order.")
            validation_messages.append("Check deployment health and disruption budget first.")
            risk_level = "medium"
        elif request.action_type == OcpActionType.LOG_BUNDLE:
            if not request.resource_name:
                raise ValueError("resource_name is required for log_bundle")
            if permission_hints and not permission_hints.get("can_get_pod_logs", False):
                validation_messages.append("Connected identity does not currently have pod log access in this namespace.")
            summary = f"{namespace} namespace의 pod/{request.resource_name} 로그 수집 번들을 준비합니다."
            preview_command = f"oc logs pod/{request.resource_name} -n {namespace} --all-containers=true"
            validation_messages.append("Log collection may expose sensitive data; confirm audience before exporting.")
            validation_messages.append("Bundle/export target should be auditable.")
            risk_level = "low"
        else:
            raise ValueError(f"Unsupported action_type={request.action_type}")

        if request.reason:
            validation_messages.append(f"Reason: {request.reason}")
        if request.break_glass:
            validation_messages.append("Break-glass override requested for incident/emergency handling.")
            if request.break_glass_ticket:
                validation_messages.append(f"Break-glass ticket: {request.break_glass_ticket}")
        if resolved_roles:
            validation_messages.append(f"Connection-backed roles: {', '.join(resolved_roles)}")
        if rbac_rules_incomplete:
            validation_messages.append("RBAC rules review was marked incomplete; final authorization is still confirmed by API dry-run/SSAR.")
        if rbac_evaluation_error:
            validation_messages.append(f"RBAC evaluation note: {rbac_evaluation_error}")

        preview = OcpActionPreviewResponse(
            connection_id=request.connection_id,
            action_type=request.action_type,
            namespace=namespace,
            resource_name=request.resource_name,
            allowed=True,
            risk_level=risk_level,
            summary=summary,
            preview_command=preview_command,
            break_glass=request.break_glass,
            break_glass_reason=request.break_glass_reason,
            break_glass_ticket=request.break_glass_ticket,
            required_approvals=1,
            approval_strategy="single_approval",
            policy_checks=[],
            blocked_reasons=[],
            validation_messages=validation_messages,
            next_step=next_step,
        )
        return self.policy_service.apply(request, preview, connection_metadata=connection_metadata)

    def _request(
        self,
        *,
        method: str,
        runtime: dict,
        path: str,
        params: dict | None = None,
        headers: dict | None = None,
        content: str | None = None,
    ) -> httpx.Response:
        merged_headers = {
            "Authorization": f"Bearer {runtime.get('token', '')}",
            "Accept": "application/json, text/plain;q=0.9",
            **(headers or {}),
        }
        with httpx.Client(
            verify=bool(runtime.get("verify_ssl", True)),
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            return client.request(
                method=method,
                url=f"{runtime['base_url']}{path}",
                params=params,
                headers=merged_headers,
                content=content,
            )

    def _get_resource(self, runtime: dict, manifest_kind: str, namespace: str, name: str) -> dict:
        response = self._request(
            method="GET",
            runtime=runtime,
            path=self._resource_path(manifest_kind, namespace, name),
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def _patch_yaml_apply_dry_run(
        self,
        runtime: dict,
        manifest_kind: str,
        namespace: str,
        name: str,
        manifest_yaml: str,
    ) -> httpx.Response:
        return self._request(
            method="PATCH",
            runtime=runtime,
            path=self._resource_path(manifest_kind, namespace, name),
            params={"dryRun": "All", "fieldManager": "cywell-copilot"},
            headers={"Content-Type": "application/apply-patch+yaml"},
            content=manifest_yaml,
        )

    @staticmethod
    def _parse_manifest_yaml(manifest_yaml: str) -> dict:
        try:
            manifest = yaml.safe_load(manifest_yaml) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"manifest_yaml is invalid: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("manifest_yaml must decode to a mapping object.")
        return manifest

    @staticmethod
    def _resource_path(manifest_kind: str, namespace: str, name: str) -> str:
        kind = OcpActionPreviewService._normalize_kind(manifest_kind)
        if kind == "deployments":
            return f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}"
        if kind == "services":
            return f"/api/v1/namespaces/{namespace}/services/{name}"
        if kind == "routes":
            return f"/apis/route.openshift.io/v1/namespaces/{namespace}/routes/{name}"
        raise ValueError(f"yaml_apply does not support kind={manifest_kind!r}")

    @staticmethod
    def _normalize_kind(manifest_kind: str) -> str:
        normalized = str(manifest_kind or "").strip().casefold()
        singular_to_plural = {
            "deployment": "deployments",
            "service": "services",
            "route": "routes",
        }
        return singular_to_plural.get(normalized, normalized)

    @staticmethod
    def _build_unified_diff(before: dict, after: dict) -> str:
        before_text = json.dumps(before or {}, ensure_ascii=False, indent=2, sort_keys=True).splitlines()
        after_text = json.dumps(after or {}, ensure_ascii=False, indent=2, sort_keys=True).splitlines()
        return "\n".join(
            difflib.unified_diff(
                before_text,
                after_text,
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            for key in ("message", "error", "detail"):
                value = payload.get(key)
                if str(value or "").strip():
                    return str(value)
        text = response.text.strip()
        return text or f"dry-run rejected with HTTP {response.status_code}"

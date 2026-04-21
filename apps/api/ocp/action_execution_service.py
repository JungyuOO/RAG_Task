from __future__ import annotations

from datetime import datetime, timezone

import httpx
import yaml

from apps.api.storage.action_execution_repository import InMemoryActionExecutionRepository
from apps.api.schemas.actions import (
    OcpActionAuditEventType,
    OcpActionExecuteRequest,
    OcpActionExecutionListResponse,
    OcpActionExecutionRecord,
    OcpActionExecutionStatus,
)
from apps.api.ocp.action_audit_service import OcpActionAuditService
from apps.api.ocp.action_request_service import OcpActionRequestService
from apps.api.ocp.auth import OcpConnectionBroker
from apps.api.ocp.manifest_sanitizer import dump_manifest_yaml, sanitize_manifest_for_apply


class _ActionPreflightError(ValueError):
    def __init__(self, message: str, *, checks: list[str]) -> None:
        super().__init__(message)
        self.checks = checks


class _ActionExecutionError(ValueError):
    def __init__(self, code: str, message: str, *, checks: list[str]) -> None:
        super().__init__(message)
        self.code = code
        self.checks = checks


class OcpActionExecutionService:
    """Safe execute flow for approved action requests.

    Execution uses real apply or read-only requests where possible:
    - scale_deployment -> PATCH scale subresource (real apply)
    - rollout_restart -> PATCH deployment annotation (real apply)
    - yaml_apply -> PATCH resource via server-side apply (real apply)
    - log_bundle -> GET pod log preview (read-only)
    """

    def __init__(
        self,
        *,
        request_service: OcpActionRequestService,
        broker: OcpConnectionBroker,
        repository: InMemoryActionExecutionRepository | None = None,
        audit_service: OcpActionAuditService | None = None,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.request_service = request_service
        self.broker = broker
        self.repository = repository or InMemoryActionExecutionRepository()
        self.audit_service = audit_service or OcpActionAuditService()
        self.transport = transport
        self.timeout = timeout

    def execute(self, request_id: str, request: OcpActionExecuteRequest) -> OcpActionExecutionRecord:
        action_request = self.request_service.get(request_id)
        if action_request is None:
            raise LookupError(f"Unknown action request: {request_id}")
        if action_request.status != "approved":
            raise ValueError("Only approved action requests can be executed.")
        if not action_request.preview.allowed:
            raise ValueError("Blocked actions cannot be executed.")
        actor_roles = self._normalize_roles(request.actor_roles)
        if not actor_roles:
            raise ValueError("actor_roles is required for action execution.")
        if actor_roles and not actor_roles.intersection(action_request.preview.executor_roles):
            raise ValueError(
                f"Actor roles {sorted(actor_roles)} do not satisfy executor roles {action_request.preview.executor_roles}."
            )

        preview = action_request.preview
        profile = self.broker.get_profile(preview.connection_id)
        if profile is None:
            raise LookupError(f"Unknown connection_id={preview.connection_id}")
        runtime = self.broker.build_runtime_config(profile)
        if runtime.get("exchange_required"):
            raise ValueError("This connection requires token exchange and cannot run safe dry-run execution yet.")

        preflight_checks: list[str] = []
        try:
            if preview.action_type == "scale_deployment":
                preflight_checks = self._preflight_scale(
                    runtime,
                    preview.namespace,
                    preview.resource_name,
                    preview.preview_command,
                    break_glass=preview.break_glass,
                    break_glass_ticket=preview.break_glass_ticket,
                )
                output_lines = self._execute_scale(runtime, preview.namespace, preview.resource_name, preview.preview_command)
                if preview.break_glass:
                    output_lines.insert(0, f"Break-glass override acknowledged for ticket {preview.break_glass_ticket or '-'}")
                record = self.repository.create(
                    request_id=request_id,
                    status=OcpActionExecutionStatus.SUCCEEDED,
                    execution_mode="real",
                    simulated=False,
                    preview=preview,
                    summary=f"Approved action request {request_id} applied a scale operation.",
                    preflight_checks=preflight_checks,
                    output_lines=output_lines,
                )
                self._audit_success(
                    record,
                    actor_id=request.actor_id,
                    actor_roles=request.actor_roles,
                    execution_note=request.execution_note,
                )
                return record
            if preview.action_type == "rollout_restart":
                preflight_checks = self._preflight_rollout_restart(
                    runtime,
                    preview.namespace,
                    preview.resource_name,
                    preview.preview_command,
                    break_glass=preview.break_glass,
                    break_glass_ticket=preview.break_glass_ticket,
                )
                output_lines = self._execute_rollout_restart(runtime, preview.namespace, preview.resource_name, preview.preview_command)
                if preview.break_glass:
                    output_lines.insert(0, f"Break-glass override acknowledged for ticket {preview.break_glass_ticket or '-'}")
                record = self.repository.create(
                    request_id=request_id,
                    status=OcpActionExecutionStatus.SUCCEEDED,
                    execution_mode="real",
                    simulated=False,
                    preview=preview,
                    summary=f"Approved action request {request_id} applied a rollout restart operation.",
                    preflight_checks=preflight_checks,
                    output_lines=output_lines,
                )
                self._audit_success(
                    record,
                    actor_id=request.actor_id,
                    actor_roles=request.actor_roles,
                    execution_note=request.execution_note,
                )
                return record
            if preview.action_type == "log_bundle":
                preflight_checks = self._preflight_log_bundle(runtime, preview.namespace, preview.resource_name, preview.preview_command)
                output_lines = self._execute_log_bundle_read(runtime, preview.namespace, preview.resource_name, preview.preview_command)
                record = self.repository.create(
                    request_id=request_id,
                    status=OcpActionExecutionStatus.SUCCEEDED,
                    execution_mode="read_only",
                    simulated=False,
                    preview=preview,
                    summary=f"Approved action request {request_id} fetched a read-only log preview.",
                    preflight_checks=preflight_checks,
                    output_lines=output_lines,
                )
                self._audit_success(
                    record,
                    actor_id=request.actor_id,
                    actor_roles=request.actor_roles,
                    execution_note=request.execution_note,
                )
                return record
            if preview.action_type == "yaml_apply":
                preflight_checks = [
                    f"Preflight: target {preview.resource_name} in namespace {preview.namespace} prepared for server-side apply.",
                    "Preflight: manifest payload supplied from the approved request record.",
                ]
                output_lines = self._execute_yaml_apply(
                    runtime,
                    preview=preview,
                    manifest_yaml=action_request.manifest_yaml,
                    force=request.force,
                )
                record = self.repository.create(
                    request_id=request_id,
                    status=OcpActionExecutionStatus.SUCCEEDED,
                    execution_mode="real",
                    simulated=False,
                    preview=preview,
                    summary=f"Approved action request {request_id} applied YAML via server-side apply.",
                    preflight_checks=preflight_checks,
                    output_lines=output_lines,
                )
                self._audit_success(
                    record,
                    actor_id=request.actor_id,
                    actor_roles=request.actor_roles,
                    execution_note=request.execution_note,
                )
                return record
        except Exception as exc:
            failed_preflight = list(getattr(exc, "checks", preflight_checks))
            record = self.repository.create(
                request_id=request_id,
                status=OcpActionExecutionStatus.FAILED,
                execution_mode=self._execution_mode_for_action(preview.action_type),
                simulated=False,
                preview=preview,
                summary=f"Approved action request {request_id} failed during safe execution.",
                preflight_checks=failed_preflight,
                output_lines=[f"Requested command: {preview.preview_command}"],
                error=str(exc),
            )
            self.audit_service.log(
                event_type=OcpActionAuditEventType.EXECUTION_FAILED,
                actor_id=request.actor_id,
                request_id=request_id,
                execution_id=record.execution_id,
                action_type=preview.action_type,
                namespace=preview.namespace,
                resource_name=preview.resource_name,
                risk_level=preview.risk_level,
                decision_note=request.execution_note,
                details={
                    "error": str(exc),
                    "error_code": getattr(exc, "code", ""),
                    "execution_mode": record.execution_mode,
                    "preflight_checks": failed_preflight,
                    "actor_roles": request.actor_roles,
                    "break_glass": preview.break_glass,
                    "break_glass_reason": preview.break_glass_reason,
                    "break_glass_ticket": preview.break_glass_ticket,
                    "force": request.force,
                },
            )
            return record

        record = self.repository.create(
            request_id=request_id,
            status=OcpActionExecutionStatus.SUCCEEDED,
            execution_mode="simulated",
            simulated=True,
            preview=preview,
            summary=f"Approved action request {request_id} executed in simulated mode.",
            preflight_checks=[],
            output_lines=[
                "Safe execute mode: no cluster mutation has been performed.",
                f"Would run: {preview.preview_command}",
                f"Risk level: {preview.risk_level}",
            ],
        )
        self._audit_success(
            record,
            actor_id=request.actor_id,
            actor_roles=request.actor_roles,
            execution_note=request.execution_note,
        )
        return record

    def list_recent(self, limit: int = 20) -> OcpActionExecutionListResponse:
        return self.repository.list_recent(limit=limit)

    def _audit_success(
        self,
        record: OcpActionExecutionRecord,
        *,
        actor_id: str,
        actor_roles: list[str],
        execution_note: str,
    ) -> None:
        self.audit_service.log(
            event_type=OcpActionAuditEventType.EXECUTION_SUCCEEDED,
            actor_id=actor_id,
            request_id=record.request_id,
            execution_id=record.execution_id,
            action_type=record.preview.action_type,
            namespace=record.preview.namespace,
            resource_name=record.preview.resource_name,
            risk_level=record.preview.risk_level,
            decision_note=execution_note,
            details={
                "execution_mode": record.execution_mode,
                "summary": record.summary,
                "preflight_checks": record.preflight_checks,
                "actor_roles": actor_roles,
                "break_glass": record.preview.break_glass,
                "break_glass_reason": record.preview.break_glass_reason,
                "break_glass_ticket": record.preview.break_glass_ticket,
            },
        )

    @staticmethod
    def _normalize_roles(actor_roles: list[str]) -> set[str]:
        return {
            str(role or "").strip().casefold()
            for role in actor_roles
            if str(role or "").strip()
        }

    @staticmethod
    def _execution_mode_for_action(action_type: str) -> str:
        if action_type == "log_bundle":
            return "read_only"
        if action_type in {"scale_deployment", "rollout_restart", "yaml_apply"}:
            return "real"
        return "simulated"

    def _request(
        self,
        *,
        method: str,
        runtime: dict,
        path: str,
        params: dict | None = None,
        headers: dict | None = None,
        json_body: dict | None = None,
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
            response = client.request(
                method=method,
                url=f"{runtime['base_url']}{path}",
                params=params,
                headers=merged_headers,
                json=json_body,
                content=content,
            )
        response.raise_for_status()
        return response

    def _execute_scale(self, runtime: dict, namespace: str, resource_name: str, preview_command: str) -> list[str]:
        response = self._request(
            method="PATCH",
            runtime=runtime,
            path=f"/apis/apps/v1/namespaces/{namespace}/deployments/{resource_name}/scale",
            headers={"Content-Type": "application/merge-patch+json"},
            json_body={"spec": {"replicas": self._extract_replicas(preview_command)}},
        )
        payload = response.json() if response.content else {}
        replicas = ((payload.get("spec") or {}).get("replicas")) if isinstance(payload, dict) else None
        return [
            f"API PATCH succeeded: HTTP {response.status_code}",
            f"Requested command: {preview_command}",
            f"Server-confirmed replicas: {replicas}",
        ]

    def _execute_rollout_restart(self, runtime: dict, namespace: str, resource_name: str, preview_command: str) -> list[str]:
        restarted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        response = self._request(
            method="PATCH",
            runtime=runtime,
            path=f"/apis/apps/v1/namespaces/{namespace}/deployments/{resource_name}",
            headers={"Content-Type": "application/merge-patch+json"},
            json_body={
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": restarted_at,
                            }
                        }
                    }
                }
            },
        )
        return [
            f"API PATCH succeeded: HTTP {response.status_code}",
            f"Requested command: {preview_command}",
            f"Restart annotation set to: {restarted_at}",
        ]

    def _execute_log_bundle_read(self, runtime: dict, namespace: str, resource_name: str, preview_command: str) -> list[str]:
        response = self._request(
            method="GET",
            runtime=runtime,
            path=f"/api/v1/namespaces/{namespace}/pods/{resource_name}/log",
            params={"tailLines": 20},
        )
        text = response.text.strip()
        lines = [line for line in text.splitlines() if line.strip()]
        return [
            f"Read-only log fetch succeeded: HTTP {response.status_code}",
            f"Requested command: {preview_command}",
            *lines[:10],
        ]

    def _execute_yaml_apply(
        self,
        runtime: dict,
        *,
        preview,
        manifest_yaml: str,
        force: bool,
    ) -> list[str]:
        if not manifest_yaml.strip():
            raise _ActionExecutionError(
                "missing_manifest_yaml",
                "yaml_apply execution requires manifest_yaml on the approved request.",
                checks=["Execution blocked: approved request does not carry manifest_yaml."],
            )
        manifest = yaml.safe_load(manifest_yaml) or {}
        manifest = sanitize_manifest_for_apply(manifest)
        sanitized_manifest_yaml = dump_manifest_yaml(manifest)
        path = self._resource_path_from_manifest(
            manifest_kind=str(manifest.get("kind") or ""),
            namespace=preview.namespace,
            name=preview.resource_name,
        )
        params = {"fieldManager": "cywell-copilot"}
        if force:
            params["force"] = "true"
        try:
            response = self._request(
                method="PATCH",
                runtime=runtime,
                path=path,
                params=params,
                headers={"Content-Type": "application/apply-patch+yaml"},
                content=sanitized_manifest_yaml,
            )
        except httpx.HTTPStatusError as exc:
            checks = [f"Execution PATCH failed with HTTP {exc.response.status_code}."]
            if exc.response.status_code == 409:
                raise _ActionExecutionError(
                    "field_ownership_conflict",
                    "field ownership conflict detected during server-side apply",
                    checks=checks,
                ) from exc
            raise
        return [
            f"API PATCH succeeded: HTTP {response.status_code}",
            f"Requested command: {preview.preview_command}",
            f"Force apply: {'yes' if force else 'no'}",
        ]

    @staticmethod
    def _resource_path_from_manifest(*, manifest_kind: str, namespace: str, name: str) -> str:
        normalized = str(manifest_kind or "").strip().casefold()
        if normalized == "deployment":
            return f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}"
        if normalized == "service":
            return f"/api/v1/namespaces/{namespace}/services/{name}"
        if normalized == "route":
            return f"/apis/route.openshift.io/v1/namespaces/{namespace}/routes/{name}"
        raise ValueError(f"Unsupported yaml_apply manifest kind: {manifest_kind!r}")

    def _preflight_scale(
        self,
        runtime: dict,
        namespace: str,
        resource_name: str,
        preview_command: str,
        *,
        break_glass: bool,
        break_glass_ticket: str,
    ) -> list[str]:
        deployment = self._fetch_deployment(runtime, namespace, resource_name)
        spec = deployment.get("spec", {}) or {}
        health_checks, health_error = self._collect_deployment_health_checks(deployment, resource_name=resource_name)
        current_replicas = int(spec.get("replicas") or 0)
        requested_replicas = self._extract_replicas(preview_command)
        delta = abs(requested_replicas - current_replicas)
        checks = [
            f"Preflight: target deployment/{resource_name} exists in namespace {namespace}.",
            f"Preflight: target current replicas={current_replicas}.",
            f"Preflight: guardrail requested replicas={requested_replicas} (delta={delta}).",
        ]
        checks.extend(health_checks)
        pdb_checks, pdb_block_message = self._collect_pdb_checks(
            runtime,
            namespace,
            deployment,
            block_on_zero_disruptions=requested_replicas < current_replicas,
        )
        checks.extend(pdb_checks)
        if delta > 5:
            if break_glass:
                checks.append(
                    f"Preflight: override break-glass ticket={break_glass_ticket or '-'} bypassed replica delta guardrail."
                )
            else:
                raise _ActionPreflightError(
                    "Replica delta above 5 is blocked by execution preflight.",
                    checks=checks,
                )
        if health_error and requested_replicas <= current_replicas:
            if break_glass:
                checks.append(
                    f"Preflight: override break-glass ticket={break_glass_ticket or '-'} bypassed health gate: {health_error}"
                )
            else:
                raise _ActionPreflightError(health_error, checks=checks)
        if pdb_block_message:
            if break_glass:
                checks.append(
                    f"Preflight: override break-glass ticket={break_glass_ticket or '-'} bypassed PDB gate: {pdb_block_message}"
                )
            else:
                raise _ActionPreflightError(pdb_block_message, checks=checks)
        return checks

    def _preflight_rollout_restart(
        self,
        runtime: dict,
        namespace: str,
        resource_name: str,
        preview_command: str,
        *,
        break_glass: bool,
        break_glass_ticket: str,
    ) -> list[str]:
        deployment = self._fetch_deployment(runtime, namespace, resource_name)
        spec = deployment.get("spec", {}) or {}
        status = deployment.get("status", {}) or {}
        health_checks, health_error = self._collect_deployment_health_checks(deployment, resource_name=resource_name)
        paused = bool(spec.get("paused") or False)
        ready_replicas = int(status.get("readyReplicas") or 0)
        checks = [
            f"Preflight: target deployment/{resource_name} exists in namespace {namespace}.",
            f"Preflight: guardrail paused={'yes' if paused else 'no'}.",
            f"Preflight: health ready replicas={ready_replicas}.",
            f"Preflight: command requested={preview_command}.",
        ]
        checks.extend(health_checks)
        if paused:
            raise _ActionPreflightError(
                "Paused deployments are blocked for rollout_restart dry-run execution.",
                checks=checks,
            )
        if health_error:
            if break_glass:
                checks.append(
                    f"Preflight: override break-glass ticket={break_glass_ticket or '-'} bypassed health gate: {health_error}"
                )
            else:
                raise _ActionPreflightError(health_error, checks=checks)
        pdb_checks, pdb_block_message = self._collect_pdb_checks(
            runtime,
            namespace,
            deployment,
            block_on_zero_disruptions=True,
        )
        checks.extend(pdb_checks)
        if pdb_block_message:
            if break_glass:
                checks.append(
                    f"Preflight: override break-glass ticket={break_glass_ticket or '-'} bypassed PDB gate: {pdb_block_message}"
                )
            else:
                raise _ActionPreflightError(pdb_block_message, checks=checks)
        return checks

    def _preflight_log_bundle(self, runtime: dict, namespace: str, resource_name: str, _: str) -> list[str]:
        pod = self._fetch_pod(runtime, namespace, resource_name)
        spec = pod.get("spec", {}) or {}
        status = pod.get("status", {}) or {}
        containers = spec.get("containers") or []
        phase = str(status.get("phase") or "")
        checks = [
            f"Preflight: target pod/{resource_name} exists in namespace {namespace}.",
            f"Preflight: health pod phase={phase or 'unknown'}.",
            f"Preflight: target container count={len(containers)}.",
        ]
        return checks

    def _fetch_deployment(self, runtime: dict, namespace: str, resource_name: str) -> dict:
        response = self._request(
            method="GET",
            runtime=runtime,
            path=f"/apis/apps/v1/namespaces/{namespace}/deployments/{resource_name}",
        )
        return response.json() if response.content else {}

    def _fetch_pod(self, runtime: dict, namespace: str, resource_name: str) -> dict:
        response = self._request(
            method="GET",
            runtime=runtime,
            path=f"/api/v1/namespaces/{namespace}/pods/{resource_name}",
        )
        return response.json() if response.content else {}

    def _collect_deployment_health_checks(self, deployment: dict, *, resource_name: str) -> tuple[list[str], str | None]:
        spec = deployment.get("spec", {}) or {}
        status = deployment.get("status", {}) or {}
        metadata = deployment.get("metadata", {}) or {}
        desired = int(spec.get("replicas") or 0)
        ready = int(status.get("readyReplicas") or 0)
        available = int(status.get("availableReplicas") or 0)
        unavailable = int(status.get("unavailableReplicas") or 0)
        updated = int(status.get("updatedReplicas") or 0)
        generation = int(metadata.get("generation") or 0)
        observed_generation = int(status.get("observedGeneration") or 0)
        checks = [
            f"Preflight: health rollout desired={desired}, updated={updated}, ready={ready}, available={available}, unavailable={unavailable}.",
            f"Preflight: health observedGeneration={observed_generation}, generation={generation}.",
        ]
        conditions = {str(item.get('type') or ''): item for item in (status.get("conditions") or []) if isinstance(item, dict)}
        available_condition = conditions.get("Available") or {}
        progressing_condition = conditions.get("Progressing") or {}
        if available_condition:
            checks.append(
                "Preflight: health Available condition="
                f"{available_condition.get('status') or 'Unknown'}"
                + (
                    f" ({available_condition.get('reason')})"
                    if available_condition.get("reason")
                    else ""
                )
            )
        if progressing_condition:
            checks.append(
                "Preflight: health Progressing condition="
                f"{progressing_condition.get('status') or 'Unknown'}"
                + (
                    f" ({progressing_condition.get('reason')})"
                    if progressing_condition.get("reason")
                    else ""
                )
            )

        if observed_generation and generation and observed_generation < generation:
            return (
                checks,
                f"Deployment {resource_name} has not observed the latest generation yet; rollout preflight is blocked.",
            )
        if str(progressing_condition.get("reason") or "") == "ProgressDeadlineExceeded":
            return (
                checks,
                f"Deployment {resource_name} is already in a progressing timeout state; rollout preflight is blocked.",
            )
        if str(available_condition.get("status") or "").casefold() == "false" or unavailable > 0:
            return (
                checks,
                f"Deployment {resource_name} is not currently healthy enough for this action; rollout preflight is blocked.",
            )
        return checks, None

    def _collect_pdb_checks(
        self,
        runtime: dict,
        namespace: str,
        deployment: dict,
        *,
        block_on_zero_disruptions: bool,
    ) -> tuple[list[str], str | None]:
        selector_labels = (
            ((deployment.get("spec") or {}).get("selector") or {}).get("matchLabels")
            or (((deployment.get("spec") or {}).get("template") or {}).get("metadata") or {}).get("labels")
            or {}
        )
        if not selector_labels:
            return ["Preflight: pdb no selector labels available for matching."], None

        try:
            response = self._request(
                method="GET",
                runtime=runtime,
                path=f"/apis/policy/v1/namespaces/{namespace}/poddisruptionbudgets",
            )
        except Exception:
            return ["Preflight: pdb lookup unavailable."], None

        payload = response.json() if response.content else {}
        items = payload.get("items", []) if isinstance(payload, dict) else []
        matches: list[str] = []
        zero_disruption_names: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_spec = item.get("spec", {}) or {}
            item_status = item.get("status", {}) or {}
            pdb_selector = (item_spec.get("selector") or {}).get("matchLabels") or {}
            if not pdb_selector:
                continue
            if not all(selector_labels.get(key) == value for key, value in pdb_selector.items()):
                continue
            name = str(((item.get("metadata") or {}).get("name")) or "pdb")
            disruptions_allowed = int(item_status.get("disruptionsAllowed") or 0)
            matches.append(f"{name}(disruptionsAllowed={disruptions_allowed})")
            if disruptions_allowed <= 0:
                zero_disruption_names.append(name)

        if not matches:
            return ["Preflight: pdb no matching budgets found."], None

        checks = [f"Preflight: pdb matching budgets: {', '.join(matches)}."]
        if block_on_zero_disruptions and zero_disruption_names:
            return (
                checks,
                "Matching pod disruption budgets currently allow zero disruptions: "
                + ", ".join(zero_disruption_names)
                + ".",
            )
        return checks, None

    @staticmethod
    def _extract_replicas(preview_command: str) -> int:
        marker = "--replicas="
        if marker in preview_command:
            value = preview_command.split(marker, 1)[1].split()[0]
            try:
                return int(value)
            except ValueError:
                return 1
        return 1





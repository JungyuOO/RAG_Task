from __future__ import annotations

from apps.api.api.schemas.actions import OcpActionPreviewRequest, OcpActionPreviewResponse, OcpActionType


class OcpActionPolicyService:
    PROTECTED_NAMESPACE_PREFIXES = ("openshift-", "kube-")
    PROTECTED_NAMESPACES = {"default", "kube-system", "kube-public"}
    PRODUCTION_NAMESPACE_MARKERS = ("prod", "production", "live")
    YAML_APPLY_ALLOWED_KINDS = {"deployments", "services", "routes"}
    VIEWER_ROLES = {"viewer", "operator", "admin"}
    WRITE_REQUESTER_ROLES = {"operator", "admin"}
    WRITE_APPROVER_ROLES = {"operator", "admin"}
    WRITE_EXECUTOR_ROLES = {"admin"}
    PRODUCTION_WRITE_ROLES = {"admin"}

    def apply(
        self,
        request: OcpActionPreviewRequest,
        preview: OcpActionPreviewResponse,
        *,
        connection_metadata: dict | None = None,
    ) -> OcpActionPreviewResponse:
        policy_checks: list[str] = []
        blocked_reasons: list[str] = []
        validation_messages = list(preview.validation_messages)
        actor_roles = self._normalize_roles(request.actor_roles)
        connection_roles = self._normalize_roles((connection_metadata or {}).get("resolved_roles", []))
        permission_hints = {
            str(key): bool(value)
            for key, value in dict((connection_metadata or {}).get("permission_hints") or {}).items()
        }

        is_yaml_apply = request.action_type == OcpActionType.YAML_APPLY
        is_write_action = request.action_type in {
            OcpActionType.SCALE_DEPLOYMENT,
            OcpActionType.ROLLOUT_RESTART,
            OcpActionType.YAML_APPLY,
        }
        break_glass_requested = bool(request.break_glass)

        protected_namespace = preview.namespace in self.PROTECTED_NAMESPACES or preview.namespace.startswith(
            self.PROTECTED_NAMESPACE_PREFIXES
        )
        if protected_namespace and is_write_action:
            blocked_reasons.append("Protected system namespaces are blocked for write actions.")

        if is_write_action and not request.reason.strip():
            blocked_reasons.append("A non-empty reason is required for write actions.")

        if request.action_type == OcpActionType.SCALE_DEPLOYMENT:
            policy_checks.append("Replica change guardrail enabled.")
            policy_checks.append("Execution preflight validates deployment existence and blocks replica deltas above 5.")
            if request.replicas > 10 and not break_glass_requested:
                blocked_reasons.append("Replica target above 10 is blocked in the current policy.")
            if request.replicas > 10 and break_glass_requested:
                validation_messages.append("Break-glass can override the replica target >10 preview guardrail during audited dry-run.")
            if request.replicas < 0:
                blocked_reasons.append("Replica target must be zero or positive.")
            if permission_hints and not permission_hints.get("can_patch_deployments", False):
                blocked_reasons.append("Connected identity does not have deployment patch permission in this namespace.")
            policy_checks.append("Scale changes use single approval to support rapid capacity adjustments.")

        if request.action_type == OcpActionType.ROLLOUT_RESTART:
            policy_checks.append("Restart preflight requires explicit deployment target.")
            policy_checks.append("Execution preflight blocks paused deployments before dry-run restart.")
            if permission_hints and not permission_hints.get("can_patch_deployments", False):
                blocked_reasons.append("Connected identity does not have deployment patch permission in this namespace.")

        if request.action_type == OcpActionType.LOG_BUNDLE:
            policy_checks.append("Log preview remains read-only.")
            policy_checks.append("Execution preflight verifies pod existence before log fetch.")
            if not request.reason.strip():
                validation_messages.append("Reason is recommended for log export/audit visibility.")
            if permission_hints and not permission_hints.get("can_get_pod_logs", False):
                blocked_reasons.append("Connected identity does not have pod log access in this namespace.")

        if is_yaml_apply:
            kind = str((request.metadata or {}).get("kind") or "").lower()
            if kind not in self.YAML_APPLY_ALLOWED_KINDS:
                blocked_reasons.append(
                    f"yaml_apply does not allow kind={kind!r}. Allowed: {sorted(self.YAML_APPLY_ALLOWED_KINDS)}."
                )
            policy_checks.append("yaml_apply uses server-side dryRun validation instead of heavy preflight.")

        required_approvals = 1
        approval_strategy = "single_approval"
        requester_roles = sorted(self.VIEWER_ROLES)
        approver_roles = sorted({"operator", "admin"})
        executor_roles = sorted(self.VIEWER_ROLES)
        approval_rules: list[str] = []
        if is_write_action and request.action_type != OcpActionType.SCALE_DEPLOYMENT:
            required_approvals = 2
            approval_strategy = "dual_approval"
            policy_checks.append("Write actions require dual approval before execution.")
            requester_roles = sorted(self.WRITE_REQUESTER_ROLES)
            approver_roles = sorted(self.WRITE_APPROVER_ROLES)
            executor_roles = sorted(self.WRITE_EXECUTOR_ROLES)
            approval_rules.append("one_admin_approval_required")
        if is_yaml_apply:
            required_approvals = 1
            approval_strategy = "single_approval"
            requester_roles = sorted(self.VIEWER_ROLES)
            approver_roles = sorted(self.WRITE_APPROVER_ROLES)
            executor_roles = sorted(self.WRITE_APPROVER_ROLES)
            approval_rules = []
            policy_checks.append(f"yaml_apply kind allowlist = {sorted(self.YAML_APPLY_ALLOWED_KINDS)}.")
        elif request.action_type == OcpActionType.SCALE_DEPLOYMENT:
            required_approvals = 1
            approval_strategy = "single_approval"
            requester_roles = sorted(self.WRITE_REQUESTER_ROLES)
            approver_roles = sorted(self.WRITE_APPROVER_ROLES)
            executor_roles = sorted(self.WRITE_APPROVER_ROLES)
            approval_rules = []

        if break_glass_requested:
            if not is_write_action:
                blocked_reasons.append("Break-glass is only supported for write actions.")
            required_approvals = max(required_approvals, 2)
            approval_strategy = "break_glass_dual_approval"
            requester_roles = ["admin"]
            approver_roles = ["admin"]
            executor_roles = ["admin"]
            approval_rules = ["break_glass_admin_only", "break_glass_ticket_required", "admin_executor_required"]
            policy_checks.append("Break-glass keeps execution in admin-only, fully audited dry-run mode.")
            if "admin" not in actor_roles:
                blocked_reasons.append("Break-glass requests require an admin actor role.")
            if len(request.break_glass_reason.strip()) < 20:
                blocked_reasons.append("Break-glass requires an emergency justification of at least 20 characters.")
            if not request.break_glass_ticket.strip():
                blocked_reasons.append("Break-glass requires a ticket or incident reference.")

        production_like_namespace = any(
            marker in preview.namespace.lower()
            for marker in self.PRODUCTION_NAMESPACE_MARKERS
        )
        if production_like_namespace and is_write_action:
            policy_checks.append("Production-like namespaces require extended reason text and dual approval.")
            policy_checks.append("Production-like namespaces require admin-like RBAC evidence from the connected identity.")
            requester_roles = sorted(self.PRODUCTION_WRITE_ROLES)
            approver_roles = sorted(self.PRODUCTION_WRITE_ROLES)
            executor_roles = sorted(self.PRODUCTION_WRITE_ROLES)
            approval_rules = ["admin_only_approvers", "admin_executor_required", *approval_rules]
            if len(request.reason.strip()) < 10 and not break_glass_requested:
                blocked_reasons.append("Production-like namespace writes require a reason of at least 10 characters.")
            if len(request.reason.strip()) < 10 and break_glass_requested:
                validation_messages.append("Break-glass emergency justification is being used in place of the standard production reason-length hint.")
            if permission_hints and not (
                permission_hints.get("has_cluster_admin_like_access", False)
                or permission_hints.get("has_namespace_admin_like_access", False)
                or permission_hints.get("can_manage_clusterroles", False)
                or permission_hints.get("can_manage_cluster_rolebindings", False)
                or permission_hints.get("can_bind_cluster_roles", False)
                or permission_hints.get("can_escalate_cluster_roles", False)
                or permission_hints.get("can_create_projects", False)
            ):
                blocked_reasons.append(
                    "Connected identity does not expose admin-like RBAC evidence required for production-like namespace writes."
                )

        if connection_roles:
            policy_checks.append(f"Connection-backed roles detected: {', '.join(sorted(connection_roles))}.")
            if actor_roles and not actor_roles.issubset(connection_roles):
                blocked_reasons.append(
                    f"Actor roles {sorted(actor_roles)} exceed connection-backed roles {sorted(connection_roles)}."
                )

        if actor_roles and not actor_roles.intersection(requester_roles):
            blocked_reasons.append(
                f"Actor roles {sorted(actor_roles)} do not satisfy requester roles {requester_roles}."
            )

        if blocked_reasons:
            validation_messages.extend(blocked_reasons)
            return preview.model_copy(
                update={
                    "allowed": False,
                    "risk_level": "high",
                    "break_glass": request.break_glass,
                    "break_glass_reason": request.break_glass_reason,
                    "break_glass_ticket": request.break_glass_ticket,
                    "required_approvals": required_approvals,
                    "approval_strategy": approval_strategy,
                    "requester_roles": requester_roles,
                    "approver_roles": approver_roles,
                    "executor_roles": executor_roles,
                    "approval_rules": approval_rules,
                    "policy_checks": policy_checks,
                    "blocked_reasons": blocked_reasons,
                    "validation_messages": validation_messages,
                    "next_step": "Resolve blocked policy items before creating an approval request.",
                }
            )

        return preview.model_copy(
            update={
                "required_approvals": required_approvals,
                "approval_strategy": approval_strategy,
                "break_glass": request.break_glass,
                "break_glass_reason": request.break_glass_reason,
                "break_glass_ticket": request.break_glass_ticket,
                "requester_roles": requester_roles,
                "approver_roles": approver_roles,
                "executor_roles": executor_roles,
                "approval_rules": approval_rules,
                "policy_checks": policy_checks,
                "blocked_reasons": [],
                "validation_messages": validation_messages,
            }
        )

    @staticmethod
    def _normalize_roles(actor_roles: list[str]) -> set[str]:
        return {
            str(role or "").strip().casefold()
            for role in actor_roles
            if str(role or "").strip()
        }


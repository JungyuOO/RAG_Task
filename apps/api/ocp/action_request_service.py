from __future__ import annotations

from apps.api.storage.action_request_repository import InMemoryActionRequestRepository
from apps.api.schemas.actions import (
    OcpActionAuditEventType,
    OcpActionRequestCreateRequest,
    OcpActionRequestDecisionRequest,
    OcpActionRequestListResponse,
    OcpActionRequestRecord,
    OcpActionPreviewRequest,
)
from apps.api.ocp.action_audit_service import OcpActionAuditService
from apps.api.ocp.action_preview_service import OcpActionPreviewService
from apps.api.ocp.auth import OcpConnectionBroker


class OcpActionRequestService:
    def __init__(
        self,
        *,
        preview_service: OcpActionPreviewService | None = None,
        repository: InMemoryActionRequestRepository | None = None,
        audit_service: OcpActionAuditService | None = None,
    ) -> None:
        self.preview_service = preview_service or OcpActionPreviewService()
        self.repository = repository or InMemoryActionRequestRepository()
        self.audit_service = audit_service or OcpActionAuditService()

    def create(self, request: OcpActionRequestCreateRequest, broker: OcpConnectionBroker) -> OcpActionRequestRecord:
        actor_roles = self._normalize_roles(request.actor_roles)
        if not actor_roles:
            raise ValueError("actor_roles is required for approval-aware action requests.")
        preview = self.preview_service.build_preview(
            OcpActionPreviewRequest(
                connection_id=request.connection_id,
                actor_id=request.actor_id,
                actor_roles=request.actor_roles,
                action_type=request.action_type,
                namespace=request.namespace,
                resource_name=request.resource_name,
                replicas=request.replicas,
                reason=request.reason,
                break_glass=request.break_glass,
                break_glass_reason=request.break_glass_reason,
                break_glass_ticket=request.break_glass_ticket,
                manifest_yaml=request.manifest_yaml,
                resource_version=request.resource_version,
            ),
            broker,
        )
        if not preview.allowed:
            raise ValueError("Action is blocked by policy and cannot be submitted for approval.")
        record = self.repository.create(
            preview=preview,
            reason=request.reason,
            requested_by=request.actor_id,
            requested_roles=request.actor_roles,
            required_approvals=preview.required_approvals,
            manifest_yaml=request.manifest_yaml,
            resource_version=request.resource_version,
        )
        self.audit_service.log(
            event_type=OcpActionAuditEventType.REQUEST_CREATED,
            actor_id=request.actor_id,
            request_id=record.request_id,
            execution_id="",
            action_type=record.preview.action_type,
            namespace=record.preview.namespace,
            resource_name=record.preview.resource_name,
            risk_level=record.preview.risk_level,
            details={
                "summary": record.preview.summary,
                "required_approvals": record.required_approvals,
                "approval_strategy": record.preview.approval_strategy,
                "requester_roles": record.preview.requester_roles,
                "requested_roles": record.requested_roles,
                "approver_roles": record.preview.approver_roles,
                "executor_roles": record.preview.executor_roles,
                "break_glass": record.preview.break_glass,
                "break_glass_reason": record.preview.break_glass_reason,
                "break_glass_ticket": record.preview.break_glass_ticket,
            },
        )
        return record

    def approve(self, request_id: str, request: OcpActionRequestDecisionRequest) -> OcpActionRequestRecord:
        if not self._normalize_roles(request.actor_roles):
            raise ValueError("actor_roles is required for approval decisions.")
        current = self.repository.get(request_id)
        if current is None:
            raise LookupError(f"Unknown action request: {request_id}")
        actor_roles = self._normalize_roles(request.actor_roles)
        if actor_roles and not actor_roles.intersection(current.preview.approver_roles):
            raise ValueError(
                f"Actor roles {sorted(actor_roles)} do not satisfy approver roles {current.preview.approver_roles}."
            )
        if "admin_only_approvers" in current.preview.approval_rules and "admin" not in actor_roles:
            raise ValueError("This request requires admin approvers.")
        if (
            "one_admin_approval_required" in current.preview.approval_rules
            and current.approval_count + 1 >= current.required_approvals
            and not self._has_admin_approval(current, actor_roles)
        ):
            raise ValueError("At least one admin approval is required before this request can become approved.")

        record = self.repository.approve(
            request_id,
            actor_id=request.actor_id,
            actor_roles=request.actor_roles,
            decision_note=request.decision_note,
        )
        self.audit_service.log(
            event_type=OcpActionAuditEventType.REQUEST_APPROVED,
            actor_id=request.actor_id,
            request_id=record.request_id,
            execution_id="",
            action_type=record.preview.action_type,
            namespace=record.preview.namespace,
            resource_name=record.preview.resource_name,
            risk_level=record.preview.risk_level,
            decision_note=request.decision_note,
            details={
                "approval_count": record.approval_count,
                "required_approvals": record.required_approvals,
                "approver_ids": record.approver_ids,
                "approver_role_map": record.approver_role_map,
                "actor_roles": request.actor_roles,
                "break_glass": record.preview.break_glass,
                "break_glass_ticket": record.preview.break_glass_ticket,
            },
        )
        return record

    def reject(self, request_id: str, request: OcpActionRequestDecisionRequest) -> OcpActionRequestRecord:
        actor_roles = self._normalize_roles(request.actor_roles)
        if not actor_roles:
            raise ValueError("actor_roles is required for approval decisions.")
        current = self.repository.get(request_id)
        if current is None:
            raise LookupError(f"Unknown action request: {request_id}")
        if actor_roles and not actor_roles.intersection(current.preview.approver_roles):
            raise ValueError(
                f"Actor roles {sorted(actor_roles)} do not satisfy approver roles {current.preview.approver_roles}."
            )
        record = self.repository.reject(request_id, decision_note=request.decision_note)
        self.audit_service.log(
            event_type=OcpActionAuditEventType.REQUEST_REJECTED,
            actor_id=request.actor_id,
            request_id=record.request_id,
            execution_id="",
            action_type=record.preview.action_type,
            namespace=record.preview.namespace,
            resource_name=record.preview.resource_name,
            risk_level=record.preview.risk_level,
            decision_note=request.decision_note,
            details={
                "actor_roles": request.actor_roles,
                "break_glass": record.preview.break_glass,
                "break_glass_ticket": record.preview.break_glass_ticket,
            },
        )
        return record

    def get(self, request_id: str) -> OcpActionRequestRecord | None:
        return self.repository.get(request_id)

    def list_recent(self, limit: int = 20) -> OcpActionRequestListResponse:
        return self.repository.list_recent(limit=limit)

    @staticmethod
    def _normalize_roles(actor_roles: list[str]) -> set[str]:
        return {
            str(role or "").strip().casefold()
            for role in actor_roles
            if str(role or "").strip()
        }

    def _has_admin_approval(self, record: OcpActionRequestRecord, actor_roles: set[str]) -> bool:
        if "admin" in actor_roles:
            return True
        return any("admin" in self._normalize_roles(roles) for roles in record.approver_role_map.values())





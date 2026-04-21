from __future__ import annotations

from apps.api.storage.action_audit_repository import InMemoryActionAuditRepository
from apps.api.schemas.actions import OcpActionAuditEventType, OcpActionAuditListResponse, OcpActionAuditRecord, OcpActionType


class OcpActionAuditService:
    def __init__(self, *, repository: InMemoryActionAuditRepository | None = None) -> None:
        self.repository = repository or InMemoryActionAuditRepository()

    def log(
        self,
        *,
        event_type: OcpActionAuditEventType,
        actor_id: str,
        request_id: str,
        execution_id: str,
        action_type: OcpActionType,
        namespace: str,
        resource_name: str,
        risk_level: str,
        decision_note: str = "",
        details: dict | None = None,
    ) -> OcpActionAuditRecord:
        return self.repository.create(
            event_type=event_type,
            actor_id=actor_id,
            request_id=request_id,
            execution_id=execution_id,
            action_type=action_type,
            namespace=namespace,
            resource_name=resource_name,
            risk_level=risk_level,
            decision_note=decision_note,
            details=details,
        )

    def list_recent(self, limit: int = 20) -> OcpActionAuditListResponse:
        return self.repository.list_recent(limit=limit)





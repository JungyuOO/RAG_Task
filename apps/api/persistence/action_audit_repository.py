from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from apps.api.persistence.json_persistence import load_json_file, save_json_file
from apps.api.api.schemas.actions import OcpActionAuditEventType, OcpActionAuditListResponse, OcpActionAuditRecord, OcpActionType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass(slots=True)
class _AuditModel:
    order: int
    event_id: str
    event_type: OcpActionAuditEventType
    actor_id: str
    request_id: str
    execution_id: str
    action_type: OcpActionType
    namespace: str
    resource_name: str
    risk_level: str
    decision_note: str
    details: dict
    created_at: datetime


class InMemoryActionAuditRepository:
    def __init__(self, *, storage_path: Path | None = None) -> None:
        self._lock = Lock()
        self._counter = 0
        self._items: dict[str, _AuditModel] = {}
        self.storage_path = storage_path
        self._load()

    def create(
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
        with self._lock:
            self._counter += 1
            now = _utc_now()
            model = _AuditModel(
                order=self._counter,
                event_id=f"audit-{uuid4().hex}",
                event_type=event_type,
                actor_id=actor_id,
                request_id=request_id,
                execution_id=execution_id,
                action_type=action_type,
                namespace=namespace,
                resource_name=resource_name,
                risk_level=risk_level,
                decision_note=decision_note,
                details=dict(details or {}),
                created_at=now,
            )
            self._items[model.event_id] = model
            self._save()
            return self._to_record(model)

    def list_recent(self, limit: int = 20) -> OcpActionAuditListResponse:
        with self._lock:
            models = sorted(self._items.values(), key=lambda item: (item.created_at, item.order), reverse=True)
            return OcpActionAuditListResponse(items=[self._to_record(item) for item in models[: max(limit, 1)]])

    def clear(self) -> None:
        with self._lock:
            self._counter = 0
            self._items.clear()
            self._save()

    @staticmethod
    def _to_record(model: _AuditModel) -> OcpActionAuditRecord:
        return OcpActionAuditRecord(
            event_id=model.event_id,
            event_type=model.event_type,
            actor_id=model.actor_id,
            request_id=model.request_id,
            execution_id=model.execution_id,
            action_type=model.action_type,
            namespace=model.namespace,
            resource_name=model.resource_name,
            risk_level=model.risk_level,
            decision_note=model.decision_note,
            details=model.details,
            created_at=model.created_at,
        )

    def _load(self) -> None:
        if self.storage_path is None:
            return
        payload = load_json_file(self.storage_path, default={"counter": 0, "items": []})
        self._counter = int(payload.get("counter") or 0)
        self._items = {}
        for item in payload.get("items", []):
            model = _AuditModel(
                order=int(item["order"]),
                event_id=item["event_id"],
                event_type=OcpActionAuditEventType(item["event_type"]),
                actor_id=item.get("actor_id", ""),
                request_id=item.get("request_id", ""),
                execution_id=item.get("execution_id", ""),
                action_type=OcpActionType(item["action_type"]),
                namespace=item.get("namespace", ""),
                resource_name=item.get("resource_name", ""),
                risk_level=item.get("risk_level", ""),
                decision_note=item.get("decision_note", ""),
                details=dict(item.get("details") or {}),
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            self._items[model.event_id] = model

    def _save(self) -> None:
        if self.storage_path is None:
            return
        save_json_file(
            self.storage_path,
            {
                "counter": self._counter,
                "items": [
                    {
                        "order": item.order,
                        "event_id": item.event_id,
                        "event_type": item.event_type.value,
                        "actor_id": item.actor_id,
                        "request_id": item.request_id,
                        "execution_id": item.execution_id,
                        "action_type": item.action_type.value,
                        "namespace": item.namespace,
                        "resource_name": item.resource_name,
                        "risk_level": item.risk_level,
                        "decision_note": item.decision_note,
                        "details": item.details,
                        "created_at": item.created_at.isoformat(),
                    }
                    for item in self._items.values()
                ],
            },
        )


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from apps.api.persistence.json_persistence import load_json_file, save_json_file
from apps.api.api.schemas.actions import (
    OcpActionRequestListResponse,
    OcpActionRequestRecord,
    OcpActionRequestStatus,
    OcpActionPreviewResponse,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass(slots=True)
class _ActionRequestModel:
    order: int
    request_id: str
    status: OcpActionRequestStatus
    preview: OcpActionPreviewResponse
    requested_by: str
    requested_roles: list[str]
    required_approvals: int
    approver_ids: list[str]
    approver_role_map: dict[str, list[str]]
    reason: str
    manifest_yaml: str
    resource_version: str | None
    decision_note: str
    created_at: datetime
    updated_at: datetime


class InMemoryActionRequestRepository:
    def __init__(self, *, storage_path: Path | None = None) -> None:
        self._lock = Lock()
        self._counter = 0
        self._items: dict[str, _ActionRequestModel] = {}
        self.storage_path = storage_path
        self._load()

    def create(
        self,
        *,
        preview: OcpActionPreviewResponse,
        reason: str,
        requested_by: str,
        requested_roles: list[str],
        required_approvals: int,
        manifest_yaml: str = "",
        resource_version: str | None = None,
    ) -> OcpActionRequestRecord:
        with self._lock:
            self._counter += 1
            now = _utc_now()
            model = _ActionRequestModel(
                order=self._counter,
                request_id=f"action-{uuid4().hex}",
                status=OcpActionRequestStatus.PENDING,
                preview=preview,
                requested_by=requested_by,
                requested_roles=list(requested_roles),
                required_approvals=max(int(required_approvals), 1),
                approver_ids=[],
                approver_role_map={},
                reason=reason,
                manifest_yaml=manifest_yaml,
                resource_version=resource_version,
                decision_note="",
                created_at=now,
                updated_at=now,
            )
            self._items[model.request_id] = model
            self._save()
            return self._to_record(model)

    def get(self, request_id: str) -> OcpActionRequestRecord | None:
        with self._lock:
            model = self._items.get(request_id)
            return self._to_record(model) if model is not None else None

    def approve(self, request_id: str, *, actor_id: str, actor_roles: list[str], decision_note: str = "") -> OcpActionRequestRecord:
        with self._lock:
            model = self._require(request_id)
            if model.status == OcpActionRequestStatus.REJECTED:
                raise ValueError("Rejected requests cannot be approved.")
            if actor_id in model.approver_ids:
                raise ValueError("The same actor cannot approve the same request twice.")
            if model.status == OcpActionRequestStatus.APPROVED:
                raise ValueError("This request is already fully approved.")
            model.approver_ids.append(actor_id)
            model.approver_role_map[actor_id] = list(actor_roles)
            model.status = (
                OcpActionRequestStatus.APPROVED
                if len(model.approver_ids) >= model.required_approvals
                else OcpActionRequestStatus.PENDING
            )
            model.decision_note = decision_note
            model.updated_at = _utc_now()
            self._save()
            return self._to_record(model)

    def reject(self, request_id: str, *, decision_note: str = "") -> OcpActionRequestRecord:
        with self._lock:
            model = self._require(request_id)
            model.status = OcpActionRequestStatus.REJECTED
            model.decision_note = decision_note
            model.updated_at = _utc_now()
            self._save()
            return self._to_record(model)

    def list_recent(self, limit: int = 20) -> OcpActionRequestListResponse:
        with self._lock:
            models = sorted(
                self._items.values(),
                key=lambda item: (item.updated_at, item.created_at, item.order),
                reverse=True,
            )
            return OcpActionRequestListResponse(items=[self._to_record(model) for model in models[: max(limit, 1)]])

    def clear(self) -> None:
        with self._lock:
            self._counter = 0
            self._items.clear()
            self._save()

    def _require(self, request_id: str) -> _ActionRequestModel:
        model = self._items.get(request_id)
        if model is None:
            raise LookupError(f"Unknown action request: {request_id}")
        return model

    def _load(self) -> None:
        if self.storage_path is None:
            return
        payload = load_json_file(self.storage_path, default={"counter": 0, "items": []})
        self._counter = int(payload.get("counter") or 0)
        self._items = {}
        for item in payload.get("items", []):
            model = _ActionRequestModel(
                order=int(item["order"]),
                request_id=item["request_id"],
                status=OcpActionRequestStatus(item["status"]),
                preview=OcpActionPreviewResponse.model_validate(item["preview"]),
                requested_by=item.get("requested_by", ""),
                requested_roles=list(item.get("requested_roles") or []),
                required_approvals=max(int(item.get("required_approvals") or 1), 1),
                approver_ids=list(item.get("approver_ids") or []),
                approver_role_map={str(key): list(value) for key, value in dict(item.get("approver_role_map") or {}).items()},
                reason=item.get("reason", ""),
                manifest_yaml=item.get("manifest_yaml", ""),
                resource_version=item.get("resource_version"),
                decision_note=item.get("decision_note", ""),
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )
            self._items[model.request_id] = model

    def _save(self) -> None:
        if self.storage_path is None:
            return
        save_json_file(
            self.storage_path,
            {
                "counter": self._counter,
                "items": [
                    {
                        "order": model.order,
                        "request_id": model.request_id,
                        "status": model.status.value,
                        "preview": model.preview.model_dump(mode="json"),
                        "requested_by": model.requested_by,
                        "requested_roles": model.requested_roles,
                        "required_approvals": model.required_approvals,
                        "approver_ids": model.approver_ids,
                        "approver_role_map": model.approver_role_map,
                        "reason": model.reason,
                        "manifest_yaml": model.manifest_yaml,
                        "resource_version": model.resource_version,
                        "decision_note": model.decision_note,
                        "created_at": model.created_at.isoformat(),
                        "updated_at": model.updated_at.isoformat(),
                    }
                    for model in self._items.values()
                ],
            },
        )

    @staticmethod
    def _to_record(model: _ActionRequestModel) -> OcpActionRequestRecord:
        return OcpActionRequestRecord(
            request_id=model.request_id,
            status=model.status,
            preview=model.preview,
            requested_by=model.requested_by,
            requested_roles=model.requested_roles,
            required_approvals=model.required_approvals,
            approval_count=len(model.approver_ids),
            approver_ids=model.approver_ids,
            approver_role_map=model.approver_role_map,
            reason=model.reason,
            manifest_yaml=model.manifest_yaml,
            resource_version=model.resource_version,
            decision_note=model.decision_note,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


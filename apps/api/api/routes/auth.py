from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.api.schemas.auth import (
    OcpConnectionLeaseRefreshRequest,
    OcpConnectionRequest,
    OcpConnectionStatusResponse,
    OcpConnectionTestRequest,
    OcpConnectionTestResult,
    OcpDisconnectRequest,
    OcpLeaseSchedulerStatusResponse,
)
from apps.api.runtime import connection_broker, connection_lease_scheduler, connection_verifier

router = APIRouter(prefix="/ocp", tags=["ocp-auth"])
_broker = connection_broker
_verifier = connection_verifier


@router.post("/connect", response_model=OcpConnectionStatusResponse)
async def connect_ocp(request: OcpConnectionRequest) -> OcpConnectionStatusResponse:
    profile = _broker.create_profile(request)
    return OcpConnectionStatusResponse(connected=True, connection=profile, message="Connection profile created.")


@router.get("/status/{connection_id}", response_model=OcpConnectionStatusResponse)
async def get_ocp_connection_status(connection_id: str) -> OcpConnectionStatusResponse:
    profile = _broker.get_profile(connection_id)
    if profile is None:
        return OcpConnectionStatusResponse(connected=False, connection=None, message="Connection profile not found.")
    return OcpConnectionStatusResponse(connected=True, connection=profile, message="Connection profile loaded.")


@router.post("/test", response_model=OcpConnectionTestResult)
async def test_ocp_connection(request: OcpConnectionTestRequest) -> OcpConnectionTestResult:
    profile = _broker.get_profile(request.connection_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Connection profile not found.")
    runtime = _broker.build_runtime_config(profile)
    result = await _verifier.verify(profile, runtime, _broker)
    if result.success:
        updated_profile = profile.model_copy(
            update={
                "username_hint": result.resolved_user or profile.username_hint,
                "metadata": {
                    **profile.metadata,
                    "resolved_groups": result.resolved_groups,
                    "resolved_roles": result.resolved_roles,
                    "identity_source": result.identity_source,
                    "permission_hints": result.permission_hints,
                    "rbac_evidence": result.rbac_evidence,
                    "rbac_rules_incomplete": result.rbac_rules_incomplete,
                    "rbac_evaluation_error": result.rbac_evaluation_error,
                    "secret_backend": result.secret_backend,
                    "secret_version": result.secret_version,
                    "secret_created_at": result.secret_created_at,
                    "secret_lease_renewable": result.secret_lease_renewable,
                    "secret_lease_ttl_seconds": result.secret_lease_ttl_seconds,
                    "secret_lease_expires_at": result.secret_lease_expires_at,
                    "secret_rotation_supported": result.secret_rotation_supported,
                    "secret_auto_renew_applied": result.secret_auto_renew_applied,
                    "secret_auto_renew_threshold_seconds": result.secret_auto_renew_threshold_seconds,
                    "secret_renew_message": result.secret_renew_message,
                },
            }
        )
        _broker.profile_store.put(updated_profile)
    return result


@router.post("/lease/refresh", response_model=OcpConnectionTestResult)
async def refresh_ocp_connection_lease(request: OcpConnectionLeaseRefreshRequest) -> OcpConnectionTestResult:
    profile = _broker.get_profile(request.connection_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Connection profile not found.")
    secret_status = _broker.describe_secret(profile, refresh=True, renew=True)
    result = _broker.build_success_result(
        profile,
        resolved_user=profile.username_hint,
        resolved_groups=list(profile.metadata.get("resolved_groups", [])),
        resolved_roles=list(profile.metadata.get("resolved_roles", [])),
        identity_source=str(profile.metadata.get("identity_source") or ""),
        permission_hints=dict(profile.metadata.get("permission_hints") or {}),
        rbac_evidence=list(profile.metadata.get("rbac_evidence", [])),
        secret_status=secret_status,
        resolved_namespace=profile.default_namespace,
        message="Secret lease metadata refreshed.",
    )
    updated_profile = profile.model_copy(
        update={
            "metadata": {
                **profile.metadata,
                "secret_backend": result.secret_backend,
                "secret_version": result.secret_version,
                "secret_created_at": result.secret_created_at,
                "secret_lease_renewable": result.secret_lease_renewable,
                "secret_lease_ttl_seconds": result.secret_lease_ttl_seconds,
                "secret_lease_expires_at": result.secret_lease_expires_at,
                "secret_rotation_supported": result.secret_rotation_supported,
                "secret_auto_renew_applied": result.secret_auto_renew_applied,
                "secret_auto_renew_threshold_seconds": result.secret_auto_renew_threshold_seconds,
                "secret_renew_message": result.secret_renew_message,
            }
        }
    )
    _broker.profile_store.put(updated_profile)
    return result


@router.get("/lease/status", response_model=OcpLeaseSchedulerStatusResponse)
async def get_ocp_connection_lease_status() -> OcpLeaseSchedulerStatusResponse:
    return connection_lease_scheduler.status()


@router.post("/disconnect", response_model=OcpConnectionStatusResponse)
async def disconnect_ocp(request: OcpDisconnectRequest) -> OcpConnectionStatusResponse:
    profile = _broker.disconnect_by_id(request.connection_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Connection profile not found.")
    return OcpConnectionStatusResponse(connected=False, connection=None, message=f"Disconnected from {profile.cluster_url}.")

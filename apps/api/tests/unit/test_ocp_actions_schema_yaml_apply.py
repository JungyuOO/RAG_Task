from __future__ import annotations

from apps.api.schemas.ocp_action_execute import OcpActionExecuteRequest
from apps.api.schemas.ocp_actions import (
    OcpActionPreviewRequest,
    OcpActionPreviewResponse,
    OcpActionType,
)


def test_yaml_apply_action_type_exists() -> None:
    assert OcpActionType.YAML_APPLY.value == "yaml_apply"


def test_preview_request_accepts_manifest_yaml() -> None:
    req = OcpActionPreviewRequest(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="default",
        resource_name="web",
        manifest_yaml="apiVersion: apps/v1\nkind: Deployment\n",
        resource_version="42",
    )
    assert req.manifest_yaml.startswith("apiVersion")
    assert req.resource_version == "42"


def test_preview_response_carries_dry_run_fields() -> None:
    resp = OcpActionPreviewResponse(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="default",
        resource_name="web",
        allowed=True,
        risk_level="medium",
        summary="",
        diff_unified="--- a\n+++ b\n",
        dry_run_status="ok",
        dry_run_messages=[],
    )
    assert resp.dry_run_status == "ok"
    assert resp.diff_unified.startswith("---")


def test_execute_request_force_defaults_false() -> None:
    req = OcpActionExecuteRequest(actor_roles=["admin"])
    assert req.force is False


def test_execute_request_accepts_force_true() -> None:
    req = OcpActionExecuteRequest(actor_roles=["admin"], force=True)
    assert req.force is True


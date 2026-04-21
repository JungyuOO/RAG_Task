from __future__ import annotations

from apps.api.schemas.ocp_actions import (
    OcpActionPreviewRequest,
    OcpActionPreviewResponse,
    OcpActionType,
)
from apps.api.ocp.action_policy_service import OcpActionPolicyService


def _base_preview(**overrides) -> OcpActionPreviewResponse:
    defaults = dict(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="demo",
        resource_name="web",
        allowed=True,
        risk_level="medium",
        summary="",
    )
    defaults.update(overrides)
    return OcpActionPreviewResponse(**defaults)


def _base_request(**overrides) -> OcpActionPreviewRequest:
    defaults = dict(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="demo",
        resource_name="web",
        reason="edit deployment",
        metadata={"kind": "deployments"},
        manifest_yaml="apiVersion: apps/v1\nkind: Deployment\n",
    )
    defaults.update(overrides)
    return OcpActionPreviewRequest(**defaults)


def test_yaml_apply_uses_single_approval() -> None:
    svc = OcpActionPolicyService()
    result = svc.apply(_base_request(actor_roles=["admin"]), _base_preview())
    assert result.allowed is True
    assert result.approval_strategy == "single_approval"
    assert result.required_approvals == 1


def test_yaml_apply_rejects_disallowed_kind() -> None:
    svc = OcpActionPolicyService()
    req = _base_request(metadata={"kind": "pods"})
    result = svc.apply(req, _base_preview())
    assert result.allowed is False
    assert any("pods" in reason.lower() for reason in result.blocked_reasons)


def test_yaml_apply_requires_reason() -> None:
    svc = OcpActionPolicyService()
    req = _base_request(reason="")
    result = svc.apply(req, _base_preview())
    assert result.allowed is False
    assert any("reason" in reason.lower() for reason in result.blocked_reasons)


def test_yaml_apply_blocks_protected_namespace() -> None:
    svc = OcpActionPolicyService()
    result = svc.apply(
        _base_request(namespace="openshift-config"),
        _base_preview(namespace="openshift-config"),
    )
    assert result.allowed is False



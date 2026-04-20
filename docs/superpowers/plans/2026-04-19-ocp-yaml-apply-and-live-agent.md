# OCP YAML Apply & Live Agent Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resources 탭의 YAML 편집→실제 OCP 반영 경로를 뚫고, live chat을 LLM tool-use로 재설계해 "내 pod yaml 뭐야?" / "공식문서 vs 내 yaml 차이"류 질의를 처리한다.

**Architecture:** 3개 트랙 = 3개 PR. (1) backend `yaml_apply` action 추가 + 기존 scale/restart를 dry-run에서 실반영으로 전환 (Server-Side Apply), (2) Resources 탭 Monaco editor + preview 모달 + apply 체인, (3) `live_chat_service`를 `LiveAgent` (Anthropic tool-use)로 교체 + mixed lane에 synthesis 프롬프트 추가.

**Tech Stack:** FastAPI, Pydantic v2, httpx(MockTransport 테스트), PyYAML, React 18, Monaco editor (`@monaco-editor/react`), Anthropic Claude tool-use.

**Spec reference:** `docs/superpowers/specs/2026-04-19-ocp-yaml-apply-and-live-agent-design.md`

## Implementation decisions (spec 보완)

세 가지는 브레인스토밍 후 확정:

1. **`yaml_apply`는 `single_approval` 정책으로 특수 분기.** 기존 scale/restart의 `dual_approval`은 유지(실반영 전환으로 위험도 상승). yaml_apply는 단일 사용자 UI를 위해 `single_approval` 화이트리스트에 추가.
2. **`yaml_apply`는 PDB/health preflight 생략.** SSA dryRun이 admission webhook까지 검증하므로 중복 가드를 최소화.
3. **`manifest_yaml` / `resource_version`은 `OcpActionPreviewRequest`의 top-level 필드로 추가.** `metadata: dict` 안에 숨기지 않음.

## PR 분할

- **PR1** = Phase 1 (Track A backend). 단독 revert 가능.
- **PR2** = Phase 2 (Track C frontend). PR1 머지 후.
- **PR3** = Phase 3 (Track B live agent). 독립 PR.

각 Phase 끝에 "Manual smoke + PR 생성" 체크포인트가 있다.

---

## Phase 0 · Baseline

### Task 0.1: 기존 테스트 초록 확인

**Files:** (검증만)

- [ ] **Step 1: 전체 유닛 테스트 실행**

Run: `python -m pytest apps/api/tests/unit -x -q`
Expected: 전부 PASS (회귀 기준선 확보). 실패가 있다면 기존 이슈이므로 플랜 착수 전에 fix 커밋 먼저.

- [ ] **Step 2: 현재 브랜치 스냅샷 저장**

Run: `git log -1 --oneline > /tmp/phase0-baseline.txt && cat /tmp/phase0-baseline.txt`

---

## Phase 1 · Track A — Backend YAML write path (PR1)

### Task 1.1: 스키마 확장 (`yaml_apply` action type + preview/execute 필드)

**Files:**
- Modify: `apps/api/api/schemas/ocp_actions.py`
- Modify: `apps/api/api/schemas/ocp_action_execute.py`
- Test: `apps/api/tests/unit/test_ocp_actions_schema_yaml_apply.py` (create)

- [ ] **Step 1: 테스트 먼저 작성**

Create `apps/api/tests/unit/test_ocp_actions_schema_yaml_apply.py`:
```python
from apps.api.api.schemas.actions import (
    OcpActionExecuteRequest,
    OcpActionPreviewRequest,
    OcpActionPreviewResponse,
    OcpActionType,
)


def test_yaml_apply_action_type_exists():
    assert OcpActionType.YAML_APPLY.value == "yaml_apply"


def test_preview_request_accepts_manifest_yaml():
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


def test_preview_response_carries_dry_run_fields():
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


def test_execute_request_force_defaults_false():
    req = OcpActionExecuteRequest(actor_roles=["admin"])
    assert req.force is False


def test_execute_request_accepts_force_true():
    req = OcpActionExecuteRequest(actor_roles=["admin"], force=True)
    assert req.force is True
```

- [ ] **Step 2: 실행 → 전부 FAIL 확인**

Run: `python -m pytest apps/api/tests/unit/test_ocp_actions_schema_yaml_apply.py -v`
Expected: FAIL (필드/enum 값 없음).

- [ ] **Step 3: `OcpActionType`에 `YAML_APPLY` 추가**

Edit `apps/api/api/schemas/ocp_actions.py`:
```python
class OcpActionType(str, Enum):
    SCALE_DEPLOYMENT = "scale_deployment"
    ROLLOUT_RESTART = "rollout_restart"
    LOG_BUNDLE = "log_bundle"
    YAML_APPLY = "yaml_apply"
```

- [ ] **Step 4: PreviewRequest/Response 필드 추가**

Same file — `OcpActionPreviewRequest`에 top-level 필드:
```python
    manifest_yaml: str = ""
    resource_version: str | None = None
```

`OcpActionPreviewResponse`에:
```python
    from typing import Literal
    ...
    diff_unified: str = ""
    dry_run_status: Literal["ok", "rejected", "skipped"] = "skipped"
    dry_run_messages: list[str] = Field(default_factory=list)
```
(기존 imports에 `Literal` 추가.)

- [ ] **Step 5: ExecuteRequest에 `force` 추가**

Edit `apps/api/api/schemas/ocp_action_execute.py`:
```python
class OcpActionExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    actor_id: str = "ui"
    actor_roles: list[str] = Field(default_factory=list)
    execution_note: str = ""
    force: bool = False
```

- [ ] **Step 6: 재실행 → PASS 확인**

Run: `python -m pytest apps/api/tests/unit/test_ocp_actions_schema_yaml_apply.py -v`
Expected: 5 passed.

- [ ] **Step 7: 기존 스키마 테스트 회귀 확인**

Run: `python -m pytest apps/api/tests/unit -k "action" -q`
Expected: 전부 PASS (기존 동작 변함 없음).

- [ ] **Step 8: 커밋**

```bash
git add apps/api/api/schemas/ocp_actions.py apps/api/api/schemas/ocp_action_execute.py apps/api/tests/unit/test_ocp_actions_schema_yaml_apply.py
git commit -m "feat(ocp-actions): add yaml_apply action type and preview/execute fields"
```

### Task 1.2: Policy 분기 (`yaml_apply`는 single_approval + ALLOWED_KINDS)

**Files:**
- Modify: `apps/api/integrations/ocp/action_policy_service.py`
- Test: `apps/api/tests/unit/test_ocp_action_policy_yaml_apply.py` (create)

**Design note:** `yaml_apply`는 write action이지만 기존 scale/restart의 `dual_approval`과 다르게 `single_approval`로 별도 분기. ALLOWED_KINDS={"deployments","services","routes"} 외 kind면 `blocked_reasons`에 추가.

- [ ] **Step 1: 테스트 작성**

Create `apps/api/tests/unit/test_ocp_action_policy_yaml_apply.py`:
```python
from apps.api.api.schemas.actions import (
    OcpActionPreviewRequest,
    OcpActionPreviewResponse,
    OcpActionType,
)
from apps.api.integrations.ocp.action_policy_service import OcpActionPolicyService


def _base_preview(**overrides):
    defaults = dict(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="default",
        resource_name="web",
        allowed=True,
        risk_level="medium",
        summary="",
    )
    defaults.update(overrides)
    return OcpActionPreviewResponse(**defaults)


def _base_request(**overrides):
    defaults = dict(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="default",
        resource_name="web",
        reason="edit deployment",
        metadata={"kind": "deployments"},
        manifest_yaml="apiVersion: apps/v1\nkind: Deployment\n",
    )
    defaults.update(overrides)
    return OcpActionPreviewRequest(**defaults)


def test_yaml_apply_uses_single_approval():
    svc = OcpActionPolicyService()
    result = svc.apply(_base_request(actor_roles=["admin"]), _base_preview())
    assert result.allowed is True
    assert result.approval_strategy == "single_approval"
    assert result.required_approvals == 1


def test_yaml_apply_rejects_disallowed_kind():
    svc = OcpActionPolicyService()
    req = _base_request(metadata={"kind": "pods"})
    result = svc.apply(req, _base_preview())
    assert result.allowed is False
    assert any("pods" in reason.lower() for reason in result.blocked_reasons)


def test_yaml_apply_requires_reason():
    svc = OcpActionPolicyService()
    req = _base_request(reason="")
    result = svc.apply(req, _base_preview())
    assert result.allowed is False
    assert any("reason" in reason.lower() for reason in result.blocked_reasons)


def test_yaml_apply_blocks_protected_namespace():
    svc = OcpActionPolicyService()
    result = svc.apply(
        _base_request(namespace="openshift-config"),
        _base_preview(namespace="openshift-config"),
    )
    assert result.allowed is False
```

- [ ] **Step 2: FAIL 확인**

Run: `python -m pytest apps/api/tests/unit/test_ocp_action_policy_yaml_apply.py -v`
Expected: 4 FAIL.

- [ ] **Step 3: 정책에 `yaml_apply` 분기 추가**

Edit `apps/api/integrations/ocp/action_policy_service.py`:

1. 클래스 상단 상수 추가:
```python
    YAML_APPLY_ALLOWED_KINDS = {"deployments", "services", "routes"}
```

2. `is_write_action` 계산 직후 별도 플래그:
```python
        is_yaml_apply = request.action_type == OcpActionType.YAML_APPLY
        is_write_action = request.action_type in {
            OcpActionType.SCALE_DEPLOYMENT,
            OcpActionType.ROLLOUT_RESTART,
            OcpActionType.YAML_APPLY,
        }
```
(기존 `is_write_action` 정의를 이 형태로 교체 — `YAML_APPLY`도 write지만 approval 경로가 다르므로 별도 플래그.)

3. `protected_namespace`·`reason` 검증 기존 블록은 그대로 (write action 공통).

4. 기존 `if is_write_action:` 블록에서 `required_approvals = 2 / approval_strategy = "dual_approval"` 설정이 일어나는데, yaml_apply는 그 뒤에서 덮어쓴다. 기존 블록 하단에 추가:
```python
        if is_yaml_apply:
            kind = str((request.metadata or {}).get("kind") or "").lower()
            if kind not in self.YAML_APPLY_ALLOWED_KINDS:
                blocked_reasons.append(
                    f"yaml_apply does not allow kind={kind!r}. Allowed: {sorted(self.YAML_APPLY_ALLOWED_KINDS)}."
                )
            policy_checks.append("yaml_apply uses server-side dryRun validation instead of heavy preflight.")
            required_approvals = 1
            approval_strategy = "single_approval"
            approval_rules = []
            requester_roles = sorted(self.VIEWER_ROLES)
            approver_roles = sorted(self.WRITE_APPROVER_ROLES)
            executor_roles = sorted(self.WRITE_APPROVER_ROLES)
```

5. 기존 `SCALE_DEPLOYMENT`/`ROLLOUT_RESTART`/`LOG_BUNDLE` 분기 뒤에 yaml_apply 분기 추가 (policy_checks용):
```python
        if request.action_type == OcpActionType.YAML_APPLY:
            policy_checks.append(f"yaml_apply kind allowlist = {sorted(self.YAML_APPLY_ALLOWED_KINDS)}.")
```

- [ ] **Step 4: 재실행 → PASS 확인**

Run: `python -m pytest apps/api/tests/unit/test_ocp_action_policy_yaml_apply.py -v`
Expected: 4 passed.

- [ ] **Step 5: 기존 policy 테스트 회귀 확인**

Run: `python -m pytest apps/api/tests/unit -k "policy or preview" -q`
Expected: 전부 PASS.

- [ ] **Step 6: 커밋**

```bash
git add apps/api/integrations/ocp/action_policy_service.py apps/api/tests/unit/test_ocp_action_policy_yaml_apply.py
git commit -m "feat(ocp-actions): add yaml_apply policy branch (single_approval, kind allowlist)"
```

### Task 1.3: 기존 execute의 `_dry_run` 제거 (scale/restart 실반영 전환)

**Files:**
- Modify: `apps/api/integrations/ocp/action_execution_service.py`
- Modify: `apps/api/tests/unit/test_ocp_action_execution_service.py`

**Design note:** 이 Task는 **사용자 요청에 따라 기존 scale/restart도 실반영으로 전환**. dryRun 제거 후에도 기존 테스트가 transport를 mock하므로 "query param에 `dryRun` 없음"만 assertion을 바꾸면 된다.

- [ ] **Step 1: 기존 테스트 읽기**

Read `apps/api/tests/unit/test_ocp_action_execution_service.py` 에서 dryRun 관련 assertion 전부 grep:
```bash
grep -n "dryRun\|dry_run\|Dry-run" apps/api/tests/unit/test_ocp_action_execution_service.py
```

Assertion 위치 목록을 메모 — 다음 Step에서 한꺼번에 업데이트한다.

- [ ] **Step 2: 메서드 rename + dryRun 제거**

Edit `apps/api/integrations/ocp/action_execution_service.py`:

1. `_execute_scale_dry_run` → `_execute_scale`:
```python
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
```
— `params={"dryRun": "All"}` 인자 제거, 로그 문구 변경.

2. `_execute_rollout_restart_dry_run` → `_execute_rollout_restart`: 동일하게 `params={"dryRun": "All"}` 제거 + 로그 문구 `Dry-run API PATCH succeeded` → `API PATCH succeeded`.

3. `execute()` 내부 호출처 업데이트: `self._execute_scale_dry_run(...)` → `self._execute_scale(...)`, `self._execute_rollout_restart_dry_run(...)` → `self._execute_rollout_restart(...)`.

4. `execute()` 내부 `execution_mode="dry_run"` 인자가 있는 경우 → `execution_mode="real"`로 변경. 해당 값은 `OcpActionExecutionRecord`에 내려간다. `summary` 문구의 "dry-ran" 단어도 실동작에 맞게 변경 (예: `"Approved action request {request_id} applied a scale operation."`).

5. 클래스 docstring(`"""Safe execute flow..."""`) 업데이트: "scale_deployment -> PATCH scale subresource (real apply)", "rollout_restart -> PATCH deployment annotation (real apply)", "log_bundle -> GET pod log (read-only)".

- [ ] **Step 3: 기존 테스트 assertion 업데이트**

Edit `apps/api/tests/unit/test_ocp_action_execution_service.py`:
- `params={"dryRun": "All"}` 를 검증하는 assertion → 해당 라인 제거 (dryRun 없어야 함).
- `"Dry-run API PATCH succeeded"` 문자열 검증 → `"API PATCH succeeded"`.
- `execution_mode="dry_run"` 검증 → `execution_mode="real"`.
- `summary`에 "dry-ran" 기대하는 부분 → 새 문구로.

- [ ] **Step 4: 실행**

Run: `python -m pytest apps/api/tests/unit/test_ocp_action_execution_service.py -v`
Expected: 전부 PASS.

- [ ] **Step 5: 회귀 확인**

Run: `python -m pytest apps/api/tests/unit -k "action" -q`
Expected: 전부 PASS.

- [ ] **Step 6: 커밋**

```bash
git add apps/api/integrations/ocp/action_execution_service.py apps/api/tests/unit/test_ocp_action_execution_service.py
git commit -m "refactor(ocp-actions): drop dryRun from scale/rollout_restart execute (real apply)"
```

### Task 1.4: Preview service — `yaml_apply` 분기

**Files:**
- Modify: `apps/api/integrations/ocp/action_preview_service.py`
- Test: `apps/api/tests/unit/test_ocp_action_preview_service.py` (extend)

**Design note:** preview는 (1) manifest YAML parse, (2) 현재 리소스 GET, (3) SSA dryRun PATCH, (4) unified diff 계산, (5) dryRun 4xx → `dry_run_status="rejected"` body 반환. preview service가 broker를 받아 자기가 httpx로 API server 호출하도록 확장 (기존 service는 순수 로직이었지만 dryRun 호출 필요).

- [ ] **Step 1: yaml_apply preview 테스트 작성**

Append to `apps/api/tests/unit/test_ocp_action_preview_service.py`:
```python
import httpx
import pytest

from apps.api.integrations.ocp.action_preview_service import OcpActionPreviewService


def _fake_manifest():
    return (
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        "  name: web\n"
        "  namespace: default\n"
        "spec:\n"
        "  replicas: 3\n"
    )


def _make_broker_with_runtime(transport: httpx.MockTransport):
    # 기존 테스트에서 broker 모킹 유틸이 있으면 그걸 재사용. 없으면 아래 패턴:
    from apps.api.integrations.ocp.auth import OcpConnectionBroker
    broker = OcpConnectionBroker(...)  # 기존 conftest.py 헬퍼 참고
    # runtime_config는 {'base_url': 'https://api.cluster', 'token': 't', 'verify_ssl': False}
    # broker.build_runtime_config가 이걸 리턴하도록 monkeypatch
    return broker


def test_yaml_apply_preview_ok(monkeypatch):
    request_body_holder = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request_body_holder[request.method + " " + request.url.path] = request
        if request.method == "GET":
            return httpx.Response(200, json={
                "apiVersion": "apps/v1", "kind": "Deployment",
                "metadata": {"name": "web", "namespace": "default", "resourceVersion": "10"},
                "spec": {"replicas": 2},
            })
        # PATCH (dryRun)
        assert request.url.params.get("dryRun") == "All"
        assert "fieldManager=cywell-copilot" in str(request.url)
        return httpx.Response(200, json={
            "apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": {"name": "web", "namespace": "default", "resourceVersion": "10"},
            "spec": {"replicas": 3},
        })

    transport = httpx.MockTransport(handler)
    # preview service에 transport 주입 (생성자 추가 필요)
    service = OcpActionPreviewService(transport=transport)
    broker = _make_broker_with_runtime(transport)
    from apps.api.api.schemas.actions import OcpActionPreviewRequest, OcpActionType
    req = OcpActionPreviewRequest(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="default",
        resource_name="web",
        reason="bump replicas",
        metadata={"kind": "deployments"},
        manifest_yaml=_fake_manifest(),
    )
    resp = service.build_preview(req, broker)
    assert resp.dry_run_status == "ok"
    assert "replicas" in resp.diff_unified


def test_yaml_apply_preview_rejected(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={
                "apiVersion": "apps/v1", "kind": "Deployment",
                "metadata": {"name": "web", "namespace": "default"}, "spec": {"replicas": 2},
            })
        return httpx.Response(422, json={"message": "invalid manifest: spec.replicas must be >= 0"})

    transport = httpx.MockTransport(handler)
    service = OcpActionPreviewService(transport=transport)
    broker = _make_broker_with_runtime(transport)
    from apps.api.api.schemas.actions import OcpActionPreviewRequest, OcpActionType
    req = OcpActionPreviewRequest(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="default",
        resource_name="web",
        reason="bad edit",
        metadata={"kind": "deployments"},
        manifest_yaml=_fake_manifest(),
    )
    resp = service.build_preview(req, broker)
    assert resp.dry_run_status == "rejected"
    assert any("replicas must be" in m for m in resp.dry_run_messages)


def test_yaml_apply_preview_rejects_namespace_mismatch():
    service = OcpActionPreviewService()
    broker = _make_broker_with_runtime(None)
    from apps.api.api.schemas.actions import OcpActionPreviewRequest, OcpActionType
    manifest = _fake_manifest().replace("namespace: default", "namespace: other")
    req = OcpActionPreviewRequest(
        connection_id="c1",
        action_type=OcpActionType.YAML_APPLY,
        namespace="default",
        resource_name="web",
        reason="mismatched ns",
        metadata={"kind": "deployments"},
        manifest_yaml=manifest,
    )
    with pytest.raises(ValueError, match="namespace"):
        service.build_preview(req, broker)
```

(conftest에 broker 헬퍼가 이미 있으면 그걸 우선 재사용. 없으면 `apps/api/tests/conftest.py`를 보고 `OcpConnectionBroker`를 profile 하나 가진 형태로 생성하는 fixture를 추가.)

- [ ] **Step 2: FAIL 확인**

Run: `python -m pytest apps/api/tests/unit/test_ocp_action_preview_service.py -v -k yaml_apply`
Expected: FAIL.

- [ ] **Step 3: Preview service 확장**

Edit `apps/api/integrations/ocp/action_preview_service.py`:

1. `__init__`에 httpx transport/timeout 추가 (execute service와 동형):
```python
import httpx
...
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
```

2. `build_preview` 하단 `else: raise ValueError(...)` 앞에 `yaml_apply` 분기 추가:
```python
        elif request.action_type == OcpActionType.YAML_APPLY:
            summary, preview_command, risk_level, diff_unified, dry_run_status, dry_run_messages = (
                self._prepare_yaml_apply_preview(request, runtime=self.broker_runtime(broker, request))
            )
            # 주: OcpActionPreviewResponse 생성 시 이 필드들을 포함
```

실제로는 `build_preview`의 `OcpActionPreviewResponse(...)` 생성자에 `diff_unified=..., dry_run_status=..., dry_run_messages=...`를 전달하므로, 위 헬퍼에서 튜플을 받아 최종 `preview = OcpActionPreviewResponse(...)` 호출 시 분기 결과 변수를 쓴다. 변수 기본값 (scale/restart 경로용)은 `""`, `"skipped"`, `[]`로 초기화해두면 된다.

3. 새 프라이빗 헬퍼 `_prepare_yaml_apply_preview`:
```python
    import difflib
    import yaml

    def _prepare_yaml_apply_preview(
        self, request: OcpActionPreviewRequest, *, runtime: dict
    ) -> tuple[str, str, str, str, str, list[str]]:
        if not request.manifest_yaml.strip():
            raise ValueError("manifest_yaml is required for yaml_apply")
        try:
            parsed = yaml.safe_load(request.manifest_yaml)
        except yaml.YAMLError as exc:
            raise ValueError(f"manifest_yaml is not valid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("manifest_yaml must be a single Kubernetes object")
        kind = str((parsed.get("kind") or "")).lower() + "s"  # Deployment -> deployments
        # 더 안전하게: request.metadata['kind']와 맞는지 비교 + 허용 kind는 policy가 최종 판정
        parsed_name = str((parsed.get("metadata") or {}).get("name") or "")
        parsed_ns = str((parsed.get("metadata") or {}).get("namespace") or "")
        if parsed_name != request.resource_name:
            raise ValueError(f"manifest name {parsed_name!r} does not match request resource_name {request.resource_name!r}")
        if parsed_ns and parsed_ns != request.namespace:
            raise ValueError(f"manifest namespace {parsed_ns!r} does not match request namespace {request.namespace!r}")

        api_path = self._resource_api_path(kind, request.namespace, request.resource_name)
        baseline = self._fetch_baseline(runtime, api_path)
        baseline_yaml = yaml.safe_dump(baseline, sort_keys=False) if baseline else ""

        try:
            dry_run_response = self._ssa_patch(runtime, api_path, request.manifest_yaml, dry_run=True, force=False)
        except httpx.HTTPStatusError as exc:
            messages = self._extract_api_errors(exc.response)
            return (
                f"{request.namespace} namespace의 {kind}/{request.resource_name} YAML 변경 검증 실패.",
                f"oc apply -f - -n {request.namespace}",
                "high",
                "",
                "rejected",
                messages,
            )

        applied_yaml = yaml.safe_dump(dry_run_response, sort_keys=False)
        diff = "".join(
            difflib.unified_diff(
                baseline_yaml.splitlines(keepends=True),
                applied_yaml.splitlines(keepends=True),
                fromfile="current",
                tofile="proposed",
                n=3,
            )
        )
        summary = f"{request.namespace} namespace의 {kind}/{request.resource_name} 에 Server-Side Apply를 수행합니다."
        return (summary, f"oc apply -f - -n {request.namespace}", "medium", diff, "ok", [])

    @staticmethod
    def _resource_api_path(kind: str, namespace: str, name: str) -> str:
        api = {
            "deployments": ("apis/apps/v1", f"namespaces/{namespace}/deployments"),
            "services": ("api/v1", f"namespaces/{namespace}/services"),
            "routes": ("apis/route.openshift.io/v1", f"namespaces/{namespace}/routes"),
        }.get(kind)
        if api is None:
            raise ValueError(f"Unsupported yaml_apply kind={kind!r}")
        prefix, tail = api
        return f"/{prefix}/{tail}/{name}"

    def _fetch_baseline(self, runtime: dict, api_path: str) -> dict | None:
        try:
            response = self._request(method="GET", runtime=runtime, path=api_path)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return response.json() if response.content else None

    def _ssa_patch(self, runtime: dict, api_path: str, manifest_yaml: str, *, dry_run: bool, force: bool) -> dict:
        params: dict[str, str] = {"fieldManager": "cywell-copilot"}
        if dry_run:
            params["dryRun"] = "All"
        if force:
            params["force"] = "true"
        response = self._request(
            method="PATCH",
            runtime=runtime,
            path=api_path,
            params=params,
            headers={"Content-Type": "application/apply-patch+yaml"},
            body=manifest_yaml,
        )
        return response.json() if response.content else {}

    @staticmethod
    def _extract_api_errors(response: httpx.Response) -> list[str]:
        try:
            body = response.json()
        except Exception:
            return [response.text[:500] or f"HTTP {response.status_code}"]
        message = str(body.get("message") or "")
        details = body.get("details") or {}
        causes = [str(c.get("message")) for c in (details.get("causes") or []) if c.get("message")]
        out = [f"HTTP {response.status_code}: {message}".strip(": ")]
        out.extend(causes)
        return out
```

4. 같은 파일에 `_request` 헬퍼 추가 (execute service의 것과 유사하지만 `body: str | None` 지원):
```python
    def _request(
        self,
        *,
        method: str,
        runtime: dict,
        path: str,
        params: dict | None = None,
        headers: dict | None = None,
        json_body: dict | None = None,
        body: str | None = None,
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
                content=body.encode("utf-8") if body is not None else None,
            )
        response.raise_for_status()
        return response

    def broker_runtime(self, broker: OcpConnectionBroker, request: OcpActionPreviewRequest) -> dict:
        profile = broker.get_profile(request.connection_id)
        if profile is None:
            raise LookupError(f"Unknown connection_id={request.connection_id}")
        runtime = broker.build_runtime_config(profile)
        if runtime.get("exchange_required"):
            raise ValueError("This connection requires token exchange.")
        return runtime
```

5. `build_preview` 상단의 `profile = broker.get_profile(...)` 블록은 유지 (diff_unified 경로 밖에서도 namespace 검증 필요).

- [ ] **Step 4: PyYAML 의존성 확인**

Run: `grep -E "^pyyaml|^PyYAML" requirements.txt`
Expected: 이미 있음. 없으면 `PyYAML>=6.0` 추가하고 다음 커밋에 같이.

- [ ] **Step 5: 재실행 → PASS**

Run: `python -m pytest apps/api/tests/unit/test_ocp_action_preview_service.py -v`
Expected: 모든 테스트 PASS (기존 + 새로 추가한 yaml_apply 3건).

- [ ] **Step 6: 커밋**

```bash
git add apps/api/integrations/ocp/action_preview_service.py apps/api/tests/unit/test_ocp_action_preview_service.py requirements.txt
git commit -m "feat(ocp-actions): yaml_apply preview with SSA dryRun + unified diff"
```

### Task 1.5: Execution service — `yaml_apply` 실반영 분기 + 409 force 처리

**Files:**
- Modify: `apps/api/integrations/ocp/action_execution_service.py`
- Test: `apps/api/tests/unit/test_ocp_action_execution_service.py` (extend)

- [ ] **Step 1: 테스트 추가**

Append to `apps/api/tests/unit/test_ocp_action_execution_service.py`:
```python
def test_yaml_apply_execute_calls_ssa_without_dry_run(...):
    # preview 단계에서 생성된 action_request (action_type=yaml_apply) 를 준비.
    # transport handler가 PATCH 수신 시:
    #   - params에 dryRun 없음
    #   - params["fieldManager"] == "cywell-copilot"
    #   - Content-Type: application/apply-patch+yaml
    #   - body == manifest_yaml (bytes)
    # 200 응답으로 업데이트된 manifest 반환.
    # record.status == succeeded, record.execution_mode == "real" 확인.

def test_yaml_apply_execute_conflict_without_force_raises(...):
    # handler가 409 반환 (field ownership conflict).
    # execute가 ValueError("field_ownership_conflict") 로 raise 되고 record는 FAILED로 저장됨.

def test_yaml_apply_execute_conflict_with_force_succeeds(...):
    # execute_request.force=True일 때 handler가 params["force"]=="true" 확인 후 200 반환.
```

(네 번째 케이스 — audit 기록 확인은 Task 1.6 에 둔다.)

- [ ] **Step 2: FAIL 확인**

Run: `python -m pytest apps/api/tests/unit/test_ocp_action_execution_service.py -v -k yaml_apply`
Expected: FAIL.

- [ ] **Step 3: `execute()`에 `yaml_apply` 분기 추가**

Edit `apps/api/integrations/ocp/action_execution_service.py`:

`preview.action_type == "scale_deployment"`/`"rollout_restart"`/`"log_bundle"` 세 분기 뒤에 새 분기:
```python
            elif preview.action_type == OcpActionType.YAML_APPLY:
                output_lines = self._execute_yaml_apply(
                    runtime,
                    preview,
                    force=bool(request.force),
                )
                record = self.repository.create(
                    request_id=request_id,
                    status=OcpActionExecutionStatus.SUCCEEDED,
                    execution_mode="real",
                    simulated=False,
                    preview=preview,
                    summary=f"Approved action request {request_id} applied a YAML patch via SSA.",
                    preflight_checks=[],
                    output_lines=output_lines,
                )
                self._audit_success(
                    audit_type=OcpActionAuditEventType.EXECUTION_SUCCEEDED,
                    action_request=action_request,
                    record=record,
                    actor_roles=actor_roles,
                    extras={"diff_unified": preview.diff_unified},
                )
```
(Preview 단계에서 저장된 `diff_unified`를 audit에 스냅샷으로 추가. `_audit_success`가 extras 인자를 받도록 확장 — Task 1.6에서 구현.)

새 헬퍼:
```python
    def _execute_yaml_apply(self, runtime: dict, preview, *, force: bool) -> list[str]:
        # preview에는 manifest_yaml이 없음 — action_request.original_request 에서 꺼내야 함.
        # action_request는 create 시점에 원 request를 보존하므로 여기서 접근 가능.
        # 필요시 OcpActionRequestRecord에 manifest_yaml을 노출하거나, preview.metadata에 담아 오는 방식.
        # 구현 단계에서 action_request schema 확인 후 가장 깔끔한 경로 선택.
        manifest_yaml = self._manifest_from_request(preview)
        kind = self._kind_from_request(preview)
        api_path = self._yaml_apply_path(kind, preview.namespace, preview.resource_name)
        params = {"fieldManager": "cywell-copilot"}
        if force:
            params["force"] = "true"
        try:
            response = self._request(
                method="PATCH",
                runtime=runtime,
                path=api_path,
                params=params,
                headers={"Content-Type": "application/apply-patch+yaml"},
                body=manifest_yaml,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise ValueError("field_ownership_conflict") from exc
            raise
        return [
            f"SSA apply succeeded: HTTP {response.status_code}",
            f"fieldManager=cywell-copilot force={force}",
        ]
```

`_request`는 execute service 기존 것. `json_body` 대신 `content=body.encode()`를 지원하도록 확장 (preview service와 동일 수정).

`_manifest_from_request` / `_kind_from_request` 구현: `action_request.original_request`에서 `manifest_yaml`, `metadata["kind"]` 꺼냄. request schema의 실제 필드명은 `OcpActionRequestCreateRequest` 스펙 확인 후 맞춘다.

- [ ] **Step 4: 실행 → PASS**

Run: `python -m pytest apps/api/tests/unit/test_ocp_action_execution_service.py -v`
Expected: 기존 + 신규 전부 PASS.

- [ ] **Step 5: 커밋**

```bash
git add apps/api/integrations/ocp/action_execution_service.py apps/api/tests/unit/test_ocp_action_execution_service.py
git commit -m "feat(ocp-actions): yaml_apply execute with SSA real apply + force retry on 409"
```

### Task 1.6: Audit — diff_unified 스냅샷 + yaml_apply 이벤트

**Files:**
- Modify: `apps/api/integrations/ocp/action_audit_service.py`
- Modify: `apps/api/api/schemas/ocp_action_audit.py` (필요시 `extras`/`diff_unified` 필드)
- Test: `apps/api/tests/unit/test_ocp_action_audit_service.py` (extend)

- [ ] **Step 1: audit schema 확인**

Read `apps/api/api/schemas/ocp_action_audit.py`. `OcpActionAuditRecord`에 `extras: dict[str, Any]` 혹은 유사 확장 필드가 있는지 확인. 없으면 추가.

- [ ] **Step 2: 테스트 추가**

Append to `apps/api/tests/unit/test_ocp_action_audit_service.py`:
```python
def test_yaml_apply_audit_captures_diff():
    # execute_success 경로에 yaml_apply action_request + extras={"diff_unified": "..."}를 넘기면
    # audit record에 diff_unified가 보존됨을 확인.
```

- [ ] **Step 3: FAIL → 구현 → PASS**

`_audit_success` (또는 동등 메서드)가 `extras: dict | None = None` 인자를 받고, audit record에 병합.

- [ ] **Step 4: 커밋**

```bash
git add apps/api/integrations/ocp/action_audit_service.py apps/api/api/schemas/ocp_action_audit.py apps/api/tests/unit/test_ocp_action_audit_service.py
git commit -m "feat(ocp-actions): persist diff_unified snapshot in yaml_apply audit record"
```

### Task 1.7: Routes smoke — 새 필드가 FastAPI schema로 노출되는지 확인

**Files:** (검증만)
- `apps/api/api/routes/actions.py` (변경 불필요, 스키마 상속)

- [ ] **Step 1: OpenAPI 덤프**

Run: API 서버 기동 후
```bash
curl -s http://localhost:8000/openapi.json | python -m json.tool | grep -E "yaml_apply|manifest_yaml|diff_unified|dry_run_status|force"
```
Expected: 해당 식별자들이 schemas 섹션에 보임.

- [ ] **Step 2: 커밋 — 변경 없음 (스모크만)**

### Task 1.8: Phase 1 E2E manual smoke + PR1 생성

- [ ] **Step 1: 실 클러스터 대신 local mock으로 한 번 더 검증**

httpx MockTransport가 이미 유닛 테스트에서 검증했지만, 라우트 레이어까지 통합 확인을 위해 실제 FastAPI TestClient로:
```python
# apps/api/tests/unit/test_ocp_action_routes.py 에 1건 추가
def test_yaml_apply_preview_route(client, patched_broker_with_mock_transport):
    payload = {
      "connection_id": "c1",
      "action_type": "yaml_apply",
      "namespace": "default",
      "resource_name": "web",
      "reason": "bump replicas",
      "metadata": {"kind": "deployments"},
      "manifest_yaml": "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\n  namespace: default\nspec:\n  replicas: 3\n",
    }
    resp = client.post("/actions/preview", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run_status"] == "ok"
```

- [ ] **Step 2: 전체 유닛 스위트 그린 확인**

Run: `python -m pytest apps/api/tests/unit -q`
Expected: 0 failed.

- [ ] **Step 3: PR1 push + 생성**

```bash
git push -u origin dev-ver2
gh pr create --title "feat(ocp-actions): yaml_apply action + scale/restart real apply" --body "$(cat <<'EOF'
## Summary
- `yaml_apply` action type + SSA preview (dryRun + unified diff) + execute (real apply + force on 409).
- Existing `scale_deployment`/`rollout_restart` switched from dryRun to real apply.
- Audit captures diff_unified snapshot on yaml_apply.

## Test plan
- [ ] `pytest apps/api/tests/unit -q` green
- [ ] Manual via `/actions/preview` + `/actions/requests` + approve + execute against a test cluster
EOF
)"
```

**Phase 1 완료 체크포인트:** PR1 링크 확인 후 Phase 2 착수.

---

## Phase 2 · Track C — Frontend Resources tab editor (PR2)

### Task 2.1: Monaco 의존성 결정 + 설치

**Files:**
- Modify: `apps/web/package.json`, `apps/web/package-lock.json`

- [ ] **Step 1: 기존 에디터 라이브러리 확인**

Run: `grep -E "monaco|codemirror|prism" apps/web/package.json`
Expected: Monaco/CodeMirror 모두 없으면 Monaco 선택. CodeMirror가 이미 있으면 CodeMirror로 통일 — 이 경우 Task 2.3 코드도 CodeMirror로 스왑.

- [ ] **Step 2: 설치**

Run (선택지에 따라 하나만):
```bash
cd apps/web && npm install @monaco-editor/react monaco-editor
# 또는 CodeMirror:
# cd apps/web && npm install @uiw/react-codemirror @codemirror/lang-yaml @codemirror/merge
```

- [ ] **Step 3: 빌드 스모크**

Run: `cd apps/web && npm run build`
Expected: 성공.

- [ ] **Step 4: 커밋**

```bash
git add apps/web/package.json apps/web/package-lock.json
git commit -m "chore(web): add Monaco editor for YAML apply UI"
```

### Task 2.2: `useYamlApply` hook + API 래퍼

**Files:**
- Create: `apps/web/src/lib/api/yamlApplyApi.ts`
- Create: `apps/web/src/features/actions/useYamlApply.ts`

- [ ] **Step 1: API 래퍼**

Create `apps/web/src/lib/api/yamlApplyApi.ts`:
```ts
import type { OcpActionPreviewResponse, OcpActionRequestRecord, OcpActionExecutionRecord } from "../../entities/ocp/types";

export type YamlApplyKind = "deployments" | "services" | "routes";

export interface YamlApplyPreviewArgs {
  connectionId: string;
  namespace: string;
  kind: YamlApplyKind;
  resourceName: string;
  manifestYaml: string;
  resourceVersion?: string | null;
  reason: string;
}

export async function previewYamlApply(args: YamlApplyPreviewArgs): Promise<OcpActionPreviewResponse> {
  const body = {
    connection_id: args.connectionId,
    action_type: "yaml_apply",
    namespace: args.namespace,
    resource_name: args.resourceName,
    reason: args.reason,
    metadata: { kind: args.kind },
    manifest_yaml: args.manifestYaml,
    resource_version: args.resourceVersion ?? null,
    actor_id: "ui",
    actor_roles: ["admin"],
  };
  const resp = await fetch("/actions/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`preview failed: ${resp.status}`);
  return resp.json();
}

export interface ApplyYamlArgs extends YamlApplyPreviewArgs { force?: boolean }

export async function applyYamlAction(args: ApplyYamlArgs): Promise<OcpActionExecutionRecord> {
  const preview = await previewYamlApply(args);
  if (preview.dry_run_status === "rejected") {
    throw Object.assign(new Error("dry_run_rejected"), { preview });
  }

  const createBody = {
    connection_id: args.connectionId,
    action_type: "yaml_apply",
    namespace: args.namespace,
    resource_name: args.resourceName,
    reason: args.reason,
    metadata: { kind: args.kind },
    manifest_yaml: args.manifestYaml,
    resource_version: args.resourceVersion ?? null,
    actor_id: "ui",
    actor_roles: ["admin"],
  };
  const createResp = await fetch("/actions/requests", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(createBody),
  });
  if (!createResp.ok) throw new Error(`request create failed: ${createResp.status}`);
  const record: OcpActionRequestRecord = await createResp.json();

  const approveBody = { actor_id: "ui", actor_roles: ["admin"], decision_note: "auto-approve (single-user)" };
  const approveResp = await fetch(`/actions/requests/${record.id}/approve`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(approveBody),
  });
  if (!approveResp.ok) throw new Error(`approve failed: ${approveResp.status}`);

  const executeBody = { actor_id: "ui", actor_roles: ["admin"], force: Boolean(args.force) };
  const executeResp = await fetch(`/actions/requests/${record.id}/execute`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(executeBody),
  });
  if (!executeResp.ok) {
    const errBody = await executeResp.json().catch(() => ({}));
    throw Object.assign(new Error(`execute failed: ${executeResp.status}`), { status: executeResp.status, detail: errBody.detail ?? "" });
  }
  return executeResp.json();
}
```

- [ ] **Step 2: React hook**

Create `apps/web/src/features/actions/useYamlApply.ts`:
```ts
import { useState } from "react";
import { previewYamlApply, applyYamlAction, type YamlApplyPreviewArgs, type ApplyYamlArgs } from "../../lib/api/yamlApplyApi";
import type { OcpActionPreviewResponse, OcpActionExecutionRecord } from "../../entities/ocp/types";

export function useYamlApply() {
  const [preview, setPreview] = useState<OcpActionPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [applyLoading, setApplyLoading] = useState(false);
  const [error, setError] = useState("");
  const [pendingForceRetry, setPendingForceRetry] = useState(false);

  async function runPreview(args: YamlApplyPreviewArgs) {
    setPreviewLoading(true); setError("");
    try {
      const resp = await previewYamlApply(args);
      setPreview(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : "preview error");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function runApply(args: ApplyYamlArgs): Promise<OcpActionExecutionRecord | null> {
    setApplyLoading(true); setError(""); setPendingForceRetry(false);
    try {
      return await applyYamlAction(args);
    } catch (e: any) {
      const msg = String(e?.detail || e?.message || "");
      if (msg.includes("field_ownership_conflict")) {
        setPendingForceRetry(true);
        setError("Field ownership conflict. Force apply?");
      } else if (e?.status === 403) {
        setError("Your token lacks write permission.");
      } else {
        setError(msg || "apply error");
      }
      return null;
    } finally {
      setApplyLoading(false);
    }
  }

  function reset() { setPreview(null); setError(""); setPendingForceRetry(false); }

  return { preview, previewLoading, applyLoading, error, pendingForceRetry, runPreview, runApply, reset };
}
```

- [ ] **Step 3: TypeScript 컴파일 확인**

Run: `cd apps/web && npx tsc --noEmit`
Expected: 에러 없음. 에러 있으면 `OcpActionPreviewResponse` / `OcpActionExecutionRecord` 타입 정의(`entities/ocp/types.ts`)를 백엔드 스키마에 맞게 확장.

- [ ] **Step 4: 커밋**

```bash
git add apps/web/src/lib/api/yamlApplyApi.ts apps/web/src/features/actions/useYamlApply.ts apps/web/src/entities/ocp/types.ts
git commit -m "feat(web): yamlApplyApi + useYamlApply hook for preview/apply chain"
```

### Task 2.3: Edit 토글 + Monaco editor 통합

**Files:**
- Modify: `apps/web/src/pages/resources/ResourcesPage.tsx`

- [ ] **Step 1: 편집 가능 kind 판별 상수**

ResourcesPage.tsx 상단에 추가:
```ts
const EDITABLE_KINDS = new Set<ResourceKind>(["deployments", "services", "routes"]);
```

- [ ] **Step 2: Edit 상태 + editor 렌더**

컴포넌트 내부에:
```tsx
import Editor from "@monaco-editor/react";
import { useYamlApply } from "../../features/actions/useYamlApply";

// inside component
const [editMode, setEditMode] = useState(false);
const [draftYaml, setDraftYaml] = useState("");
const yamlApply = useYamlApply();

function enterEditMode() {
  if (!resourceDetail) return;
  setDraftYaml(resourceDetail.manifestYaml);
  setEditMode(true);
  yamlApply.reset();
}

function exitEditMode() {
  if (draftYaml && resourceDetail && draftYaml !== resourceDetail.manifestYaml) {
    if (!window.confirm("Discard unsaved YAML changes?")) return;
  }
  setEditMode(false);
  setDraftYaml("");
  yamlApply.reset();
}
```

YAML Manifest 카드 헤더 영역(기존 SurfaceCard 파일의 "YAML Manifest" 카드)에 Edit 토글:
```tsx
{EDITABLE_KINDS.has(resource) && resourceDetail && !editMode && (
  <button onClick={enterEditMode}>Edit</button>
)}
{editMode && (
  <>
    <button onClick={exitEditMode}>Cancel</button>
    <button
      onClick={() => yamlApply.runPreview({
        connectionId: controller.profile!.connectionId,
        namespace, kind: resource as YamlApplyKind, resourceName: selectedName,
        manifestYaml: draftYaml,
        resourceVersion: resourceDetail?.resourceVersion ?? null,
        reason: "ui edit",
      })}
      disabled={yamlApply.previewLoading || draftYaml === resourceDetail?.manifestYaml}
    >
      Preview
    </button>
  </>
)}
```

본문 `<pre><code>{manifestYaml}</code></pre>` 부분을:
```tsx
{editMode ? (
  <Editor
    height="520px"
    defaultLanguage="yaml"
    value={draftYaml}
    onChange={(v) => setDraftYaml(v ?? "")}
    options={{ minimap: { enabled: false }, automaticLayout: true }}
  />
) : (
  <pre className="info-code"><code>{resourceDetail?.manifestYaml ?? ""}</code></pre>
)}
```

- [ ] **Step 3: 브라우저 스모크**

Run: `cd apps/web && npm run dev`
→ 브라우저에서 Resources 탭 → deployments 선택 → "Edit" 버튼 보여야 함. pods/events에서는 버튼 안 보여야 함.

- [ ] **Step 4: 커밋**

```bash
git add apps/web/src/pages/resources/ResourcesPage.tsx
git commit -m "feat(web): Edit toggle + Monaco YAML editor in Resources tab"
```

### Task 2.4: Preview 모달 (diff + dryRun 배지)

**Files:**
- Create: `apps/web/src/features/actions/YamlApplyPreviewModal.tsx`
- Modify: `apps/web/src/pages/resources/ResourcesPage.tsx`

- [ ] **Step 1: 모달 컴포넌트**

Create `apps/web/src/features/actions/YamlApplyPreviewModal.tsx`:
```tsx
import { DiffEditor } from "@monaco-editor/react";
import type { OcpActionPreviewResponse } from "../../entities/ocp/types";

interface Props {
  open: boolean;
  preview: OcpActionPreviewResponse | null;
  originalYaml: string;
  draftYaml: string;
  applyLoading: boolean;
  pendingForceRetry: boolean;
  onCancel: () => void;
  onApply: (force: boolean) => void;
}

export function YamlApplyPreviewModal(props: Props) {
  if (!props.open || !props.preview) return null;
  const rejected = props.preview.dry_run_status === "rejected";
  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <header>
          <h3>YAML Apply Preview</h3>
          <span className={rejected ? "badge-red" : "badge-green"}>
            {rejected ? "Rejected by server" : "dryRun OK"}
          </span>
        </header>
        {rejected && (
          <ul className="error-list">
            {props.preview.dry_run_messages.map((m) => <li key={m}>{m}</li>)}
          </ul>
        )}
        <DiffEditor
          height="480px"
          language="yaml"
          original={props.originalYaml}
          modified={props.draftYaml}
          options={{ renderSideBySide: true, readOnly: true }}
        />
        <footer>
          <button onClick={props.onCancel} disabled={props.applyLoading}>Cancel</button>
          <button
            onClick={() => props.onApply(false)}
            disabled={rejected || props.applyLoading || props.pendingForceRetry}
          >
            {props.applyLoading ? "Applying…" : "Apply"}
          </button>
          {props.pendingForceRetry && (
            <button onClick={() => props.onApply(true)} disabled={props.applyLoading}>
              Force Apply
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 페이지에 모달 wiring**

Edit ResourcesPage.tsx — preview 결과가 있으면 모달 렌더:
```tsx
{yamlApply.preview && (
  <YamlApplyPreviewModal
    open={Boolean(yamlApply.preview)}
    preview={yamlApply.preview}
    originalYaml={resourceDetail?.manifestYaml ?? ""}
    draftYaml={draftYaml}
    applyLoading={yamlApply.applyLoading}
    pendingForceRetry={yamlApply.pendingForceRetry}
    onCancel={() => yamlApply.reset()}
    onApply={async (force) => {
      const result = await yamlApply.runApply({
        connectionId: controller.profile!.connectionId,
        namespace, kind: resource as YamlApplyKind, resourceName: selectedName,
        manifestYaml: draftYaml,
        resourceVersion: resourceDetail?.resourceVersion ?? null,
        reason: "ui edit",
        force,
      });
      if (result) {
        yamlApply.reset();
        setEditMode(false);
        setDraftYaml("");
        // refetch detail
        const refreshed = await getOcpResourceDetail(controller.profile!.connectionId, resource, namespace, selectedName);
        setResourceDetail(refreshed);
      }
    }}
  />
)}
```

(모달 CSS는 기존 `shared/components` 스타일 패턴 참조 — 프로젝트에 modal-overlay 관련 유틸이 이미 있는지 확인 후 맞춤.)

- [ ] **Step 3: 커밋**

```bash
git add apps/web/src/features/actions/YamlApplyPreviewModal.tsx apps/web/src/pages/resources/ResourcesPage.tsx
git commit -m "feat(web): YAML apply preview modal with diff view and force retry"
```

### Task 2.5: 에러 배너 + unsaved guard

**Files:**
- Modify: `apps/web/src/pages/resources/ResourcesPage.tsx`

- [ ] **Step 1: error/forceRetry 상태 UI 반영**

에러 배너:
```tsx
{yamlApply.error && (
  <StatusNotice tone="danger">{yamlApply.error}</StatusNotice>
)}
```

`beforeunload` 가드:
```tsx
useEffect(() => {
  if (!editMode) return;
  const handler = (e: BeforeUnloadEvent) => {
    if (resourceDetail && draftYaml !== resourceDetail.manifestYaml) {
      e.preventDefault();
      e.returnValue = "";
    }
  };
  window.addEventListener("beforeunload", handler);
  return () => window.removeEventListener("beforeunload", handler);
}, [editMode, draftYaml, resourceDetail]);
```

- [ ] **Step 2: 커밋**

```bash
git add apps/web/src/pages/resources/ResourcesPage.tsx
git commit -m "feat(web): unsaved YAML guard + error banner in Resources tab"
```

### Task 2.6: Phase 2 manual E2E + PR2

- [ ] **Step 1: 수동 체크리스트**

실 클러스터 or kind 클러스터 연결 후:
1. Resources 탭 → deployments → Edit 버튼 보임. pods/events에서는 안 보임. ✔
2. Deployment YAML 열기 → replicas 숫자만 바꿔서 Preview → 모달 diff에 해당 라인만 변경 표시. ✔
3. Apply → 토스트 → 재조회 후 manifestYaml에 변경 반영. `oc get deploy/x -o yaml` 로 서버 상태 재확인. ✔
4. 고의로 잘못된 YAML (필수 field 제거) → Preview → rejected 배지 + 에러 메시지. Apply 버튼 disabled. ✔
5. 동일 리소스를 외부에서 동시 수정 → 본인이 Apply → 409 → Force Apply 버튼 노출 → 클릭 시 덮어쓰기 성공. ✔

- [ ] **Step 2: PR2 push + 생성**

```bash
git push
gh pr create --title "feat(web): YAML edit & apply UI in Resources tab" --body "..."
```

**Phase 2 완료 체크포인트:** PR2 링크 확인 후 Phase 3 착수.

---

## Phase 3 · Track B — LiveAgent (LLM tool-use) + mixed synthesis (PR3)

### Task 3.1: Tool schema 정의 + ConnectedOcpService 래핑

**Files:**
- Create: `apps/api/rag/query/live_agent_tools.py`
- Test: `apps/api/tests/unit/test_live_agent_tools.py`

- [ ] **Step 1: 테스트**

`test_live_agent_tools.py`:
```python
import pytest
from unittest.mock import AsyncMock
from apps.api.rag.query.live_agent_tools import LiveAgentTools, TOOL_SCHEMAS


def test_tool_schemas_cover_four_tools():
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert names == {"list_namespaces", "list_resources", "get_resource_yaml", "get_overview"}


@pytest.mark.asyncio
async def test_dispatch_get_resource_yaml_calls_live_service():
    live = AsyncMock()
    live.resolve_resource_detail_by_name.return_value = type("X", (), {"manifestYaml": "apiVersion: v1\n", "resourceVersion": "7"})()
    tools = LiveAgentTools(live_service=live)
    result = await tools.dispatch("get_resource_yaml", {"kind": "pods", "namespace": "default", "name": "web-1"}, connection_id="c1", broker=object())
    assert result["manifest_yaml"].startswith("apiVersion")
    assert result["resource_version"] == "7"
```

- [ ] **Step 2: 구현**

Create `apps/api/rag/query/live_agent_tools.py`:
```python
from __future__ import annotations
from typing import Any
from apps.api.integrations.ocp.auth import OcpConnectionBroker
from apps.api.integrations.ocp.live_service import ConnectedOcpService

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_namespaces",
        "description": "List namespaces in the connected OpenShift cluster.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_resources",
        "description": "List resources of a given kind in a namespace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["pods", "deployments", "services", "routes", "events"]},
                "namespace": {"type": "string"},
            },
            "required": ["kind", "namespace"],
        },
    },
    {
        "name": "get_resource_yaml",
        "description": "Fetch the full YAML manifest of a specific resource.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["pods", "deployments", "services", "routes", "events"]},
                "namespace": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["kind", "namespace", "name"],
        },
    },
    {
        "name": "get_overview",
        "description": "Return cluster overview: resource counts, default namespace, namespace count.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


class LiveAgentTools:
    def __init__(self, *, live_service: ConnectedOcpService) -> None:
        self.live_service = live_service

    async def dispatch(
        self, name: str, args: dict, *, connection_id: str, broker: OcpConnectionBroker, namespace_default: str = ""
    ) -> dict:
        if name == "list_namespaces":
            resp = await self.live_service.list_namespaces(connection_id, broker)
            return {"namespaces": [item.name for item in resp.items]}
        if name == "list_resources":
            resp = await self.live_service.list_resources(
                connection_id, resource=args["kind"], namespace=args["namespace"], broker=broker
            )
            return {"items": [{"name": it.name, "namespace": it.namespace, "status_summary": it.status_summary} for it in resp.items]}
        if name == "get_resource_yaml":
            detail = await self.live_service.resolve_resource_detail_by_name(
                connection_id, resource=args["kind"], namespace=args["namespace"], name=args["name"], broker=broker
            )
            return {"manifest_yaml": detail.manifestYaml, "resource_version": getattr(detail, "resourceVersion", "")}
        if name == "get_overview":
            overview = await self.live_service.get_overview(connection_id, broker)
            return {
                "resource_counts": overview.resource_counts,
                "default_namespace": overview.default_namespace,
                "namespace_count": overview.namespace_count,
            }
        raise ValueError(f"unknown tool: {name}")
```

- [ ] **Step 3: PASS + 커밋**

Run: `python -m pytest apps/api/tests/unit/test_live_agent_tools.py -v`
커밋: `git commit -m "feat(rag): LiveAgent tool schemas + dispatcher wrapping ConnectedOcpService"`

### Task 3.2: LiveAgent 클래스 + 루프

**Files:**
- Create: `apps/api/rag/query/live_agent.py`
- Test: `apps/api/tests/unit/test_live_agent.py`

- [ ] **Step 1: 테스트 (4 시나리오)**

`test_live_agent.py`:
```python
# 시나리오:
# (a) LLM이 tool_use 없이 바로 text 응답 → final_text 그대로.
# (b) LLM이 get_overview tool_use → dispatcher 결과를 tool_result로 반환 → 2nd LLM 호출에서 text.
# (c) MAX_TOOL_ITERATIONS (3) 초과 → fallback prompt로 한 번 더 호출.
# (d) dispatcher가 예외 throw → tool_result에 에러 내용을 넣고 LLM이 graceful 답변.

# Anthropic 클라이언트 호출을 AsyncMock으로 치환. side_effect 시퀀스로 각 turn의 응답 지정.
```

- [ ] **Step 2: 구현**

`live_agent.py`:
```python
from __future__ import annotations
import asyncio, time, logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from apps.api.rag.query.live_agent_tools import LiveAgentTools, TOOL_SCHEMAS
from apps.api.integrations.ocp.auth import OcpConnectionBroker

logger = logging.getLogger(__name__)
MAX_TOOL_ITERATIONS = 3
AGENT_TIMEOUT_SECONDS = 20.0

SYSTEM_PROMPT = (
    "You are a live cluster assistant connected to an OpenShift cluster via tools. "
    "Use tools to inspect real resources. Never invent resource names. "
    "If the user's namespace or target is ambiguous, call list_resources first. "
    "If a tool returns an empty list, say so explicitly — do not hallucinate. "
    "Respond in the user's language (Korean or English)."
)


@dataclass
class LiveAgentResult:
    answer: str
    tools_used: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    fallback_used: bool = False


class LiveAgent:
    def __init__(self, *, llm_client, tools: LiveAgentTools) -> None:
        self.llm_client = llm_client
        self.tools = tools

    async def run(
        self, *, message: str, connection_id: str, namespace: str,
        broker: OcpConnectionBroker,
        progress: Callable[[str, str], Awaitable[None] | None] | None = None,
    ) -> LiveAgentResult:
        started = time.perf_counter()
        messages = [{"role": "user", "content": message + (f"\n\n(Default namespace: {namespace})" if namespace else "")}]
        tools_used: list[dict[str, Any]] = []
        try:
            for iteration in range(MAX_TOOL_ITERATIONS):
                elapsed = time.perf_counter() - started
                remaining = AGENT_TIMEOUT_SECONDS - elapsed
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                resp = await asyncio.wait_for(
                    self.llm_client.messages_create(
                        system=SYSTEM_PROMPT, messages=messages, tools=TOOL_SCHEMAS
                    ),
                    timeout=remaining,
                )
                tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                if not tool_uses:
                    answer = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
                    return LiveAgentResult(answer=answer, tools_used=tools_used, iterations=iteration + 1)
                messages.append({"role": "assistant", "content": resp.content})
                tool_results_content = []
                for tu in tool_uses:
                    await self._emit_progress(progress, tu.name)
                    try:
                        result = await self.tools.dispatch(tu.name, tu.input, connection_id=connection_id, broker=broker)
                    except Exception as exc:
                        result = {"error": str(exc)}
                    tools_used.append({"name": tu.name, "input": tu.input, "result_summary": _summarize(result)})
                    tool_results_content.append({"type": "tool_result", "tool_use_id": tu.id, "content": _result_to_text(result)})
                messages.append({"role": "user", "content": tool_results_content})
            return await self._fallback(messages, tools_used, MAX_TOOL_ITERATIONS)
        except asyncio.TimeoutError:
            logger.warning("[live_agent] timeout after %.2fs", time.perf_counter() - started)
            return await self._fallback(messages, tools_used, MAX_TOOL_ITERATIONS, reason="timeout")

    async def _fallback(self, messages, tools_used, iterations, reason: str = "max_iterations") -> LiveAgentResult:
        messages.append({"role": "user", "content": f"Please produce a final answer now based on the gathered tool results. (reason={reason})"})
        resp = await self.llm_client.messages_create(system=SYSTEM_PROMPT, messages=messages, tools=[])
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
        return LiveAgentResult(answer=text or "응답을 생성하지 못했습니다.", tools_used=tools_used, iterations=iterations, fallback_used=True)

    @staticmethod
    async def _emit_progress(progress, tool_name):
        if progress is None: return
        label = {
            "list_namespaces": "fetch_namespaces",
            "list_resources": "fetch_live_resources",
            "get_resource_yaml": "fetch_resource_manifest",
            "get_overview": "fetch_cluster_overview",
        }.get(tool_name, tool_name)
        out = progress(label, tool_name)
        if asyncio.iscoroutine(out): await out


def _result_to_text(result: dict) -> str:
    import json
    return json.dumps(result, ensure_ascii=False)[:8000]


def _summarize(result: dict) -> str:
    import json
    s = json.dumps(result, ensure_ascii=False)
    return (s[:200] + "…") if len(s) > 200 else s
```

(주: `llm_client.messages_create` 는 이 리포의 Anthropic client 래퍼 호출명에 맞게 변경 — `apps/api/rag/generation/llm_client.py` 확인.)

- [ ] **Step 3: 실행 + 커밋**

Run: `python -m pytest apps/api/tests/unit/test_live_agent.py -v`
커밋: `git commit -m "feat(rag): LiveAgent with tool-use loop, timeout, and fallback prompt"`

### Task 3.3: `live_chat_service` 교체

**Files:**
- Modify: `apps/api/integrations/ocp/live_chat_service.py`
- Modify: `apps/api/runtime.py` (DI 조립)
- Modify: `apps/api/tests/unit/test_live_ocp_chat_routes.py` (재작성)

- [ ] **Step 1: `LiveOcpChatService.answer`를 `LiveAgent.run`으로 delegate**

`live_chat_service.py`에서 `_detect_intent`, `_format_*`, `_QueryIntent` 전부 제거. `LiveOcpChatService.__init__`에 `live_agent: LiveAgent` 받아서 `answer`가 agent를 호출하고 결과를 `OcpLiveChatResponse`로 매핑:
```python
class LiveOcpChatService:
    def __init__(self, *, live_agent: LiveAgent, live_service: ConnectedOcpService) -> None:
        self.live_agent = live_agent
        self.live_service = live_service

    async def answer(self, *, connection_id, message, namespace, broker, progress=None) -> OcpLiveChatResponse:
        result = await self.live_agent.run(
            message=message, connection_id=connection_id, namespace=namespace,
            broker=broker, progress=progress,
        )
        # items: agent가 실제로 들여다본 리소스 기반. MVP는 get_resource_yaml / list_resources 결과에서 최대 8개.
        items = self._extract_items(result.tools_used)
        return OcpLiveChatResponse(
            mode=self._infer_mode(result.tools_used),
            namespace=namespace,
            answer=result.answer,
            items=items,
        )
```

`_extract_items` / `_infer_mode`은 `tools_used` 리스트를 훑어서 기존 mode 값(`overview|list|manifest|general`) 중 하나로 결정.

- [ ] **Step 2: DI**

`apps/api/runtime.py`에서 `live_chat_service` 생성 위치 찾아서 `LiveAgent`·`LiveAgentTools`를 조립:
```python
from apps.api.rag.query.live_agent_tools import LiveAgentTools
from apps.api.rag.query.live_agent import LiveAgent

live_agent = LiveAgent(llm_client=llm_client, tools=LiveAgentTools(live_service=connected_ocp_service))
live_ocp_chat_service = LiveOcpChatService(live_agent=live_agent, live_service=connected_ocp_service)
```

- [ ] **Step 3: 기존 route 테스트 재작성**

`test_live_ocp_chat_routes.py`: `LiveAgent`를 mock하고, `/chat/live` 요청이 `LiveAgent.run`을 정확한 인자로 호출하는지 + response shape 그대로인지 확인.

- [ ] **Step 4: 실행 + 커밋**

Run: `python -m pytest apps/api/tests/unit -k "live_chat or live_ocp" -v`
커밋: `git commit -m "refactor(rag): replace rule-based live chat with LiveAgent tool-use"`

### Task 3.4: Mixed lane synthesis 프롬프트

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py` (`_build_mixed_response`)
- Test: `apps/api/tests/unit/test_unified_copilot_service.py` (extend)

- [ ] **Step 1: 테스트**

`test_unified_copilot_service.py`에 시나리오 추가 — mock LLM에 주어지는 synthesis prompt가 doc chunks + live YAML 둘 다 포함함을 assertion:
```python
async def test_mixed_lane_synthesis_includes_both_doc_and_live_yaml(...):
    # QueryRouter가 'mixed' lane 결정.
    # doc lane이 2개 citation을 가진 답변을 반환, live lane이 get_resource_yaml tool 결과로 manifest를 수집.
    # _build_mixed_response 내부에서 LLM.synthesize가 호출되며, prompt에 `[1]`, `[2]`, ```yaml```, manifest 일부가 포함돼야 함.
```

- [ ] **Step 2: 구현**

기존 `_build_mixed_response`에서 단순 concat 경로를 교체. 새 synthesis prompt:
```python
SYNTHESIS_SYSTEM = (
    "You are comparing official OpenShift documentation with the user's actual cluster state. "
    "Reference docs with [N] citations (preserve existing numbering from the doc answer). "
    "Reference live YAML with ```yaml``` code blocks. "
    "If there is a difference between docs and actual YAML, state it explicitly. "
    "Respond in the user's language."
)

async def _build_mixed_response(self, *, message, doc_response, live_response, live_available, answer_delta):
    if not live_available or live_response is None:
        return doc_response  # degrade gracefully
    user_prompt = (
        f"User question: {message}\n\n"
        f"=== Documentation answer (with existing citations) ===\n{doc_response.answer}\n\n"
        f"=== Live cluster data ===\n{live_response.answer}\n\n"
        "Synthesize a comparison answer."
    )
    synth = await self.llm_client.messages_create(
        system=SYNTHESIS_SYSTEM, messages=[{"role": "user", "content": user_prompt}], tools=[]
    )
    text = "".join(getattr(b, "text", "") for b in synth.content if getattr(b, "type", None) == "text")
    await self._emit_answer_text(text, answer_delta)
    return CopilotChatResponse(
        lane="mixed", mode="mixed", fallback_used=False, preview_ready=False,
        answer=text, sources=doc_response.sources + live_response.sources,
    )
```

- [ ] **Step 3: 실행 + 커밋**

Run: `python -m pytest apps/api/tests/unit/test_unified_copilot_service.py -v`
커밋: `git commit -m "feat(rag): mixed lane synthesis prompt combining doc citations and live YAML"`

### Task 3.5: Phase 3 manual E2E + PR3

- [ ] **Step 1: 수동 체크리스트**

클러스터 연결 후 챗 탭에서:
1. "default namespace의 web pod yaml 보여줘" → live lane → `get_resource_yaml` 호출 → YAML 코드블록 답변. ✔
2. "deployment 목록 알려줘" → `list_resources` 호출 → 목록 답변. ✔
3. "공식문서의 Deployment 권장 구성과 내 web deployment yaml의 차이점 알려줘" → mixed lane → doc citation `[1]` + live ```yaml``` 블록 + 비교 문장. ✔
4. "없는 리소스 alpha-99 pod yaml" → `list_resources` 또는 `get_resource_yaml`이 빈 응답 → "검색된 리소스 없음" 명시적 답변. 환각 없음. ✔

- [ ] **Step 2: PR3**

```bash
git push
gh pr create --title "feat(rag): LiveAgent tool-use + mixed lane synthesis" --body "..."
```

---

## Self-review (플랜 작성 후 자가 점검 기록)

- **Spec coverage:**
  - 3.1 Schema → Task 1.1 ✔
  - 3.2 Policy → Task 1.2 ✔
  - 3.3 Preview → Task 1.4 ✔
  - 3.4 Execute (real + 409 force + rename) → Tasks 1.3, 1.5 ✔
  - 3.5 Audit → Task 1.6 ✔
  - 3.6 Routes → Task 1.7 ✔ (schema wiring 자동)
  - 4.1 LiveAgent 컴포넌트 → Task 3.2 ✔
  - 4.2 Tools → Task 3.1 ✔
  - 4.3 Loop + timeout → Task 3.2 ✔
  - 4.4 System prompt → Task 3.2 ✔
  - 4.5 live_chat_service 교체 → Task 3.3 ✔
  - 4.6 Mixed synthesis → Task 3.4 ✔
  - 5.1~5.7 Frontend → Tasks 2.1~2.6 ✔
  - 6 Testing → 각 Task 내부 + 수동 체크리스트 ✔
  - 7 Sequencing (3 PR) → Phase 경계 ✔
  - 8 Observability → Task 3.2(live_agent 로그), Task 1.5(execute 로그)에서 로깅 포함. 추가 stage emit 상수는 Phase 3 내부에서 자연 발생.
  - 9 Out of scope → 스펙 그대로 (plan에 새 추가 없음).
- **Placeholder scan:** "..." 은 HEREDOC/명령에서 gh body 자리만 — 문맥상 수동 작성이므로 남겨둠. 그 외 TBD/TODO 없음.
- **Type consistency:** `yaml_apply` action 값, `YAML_APPLY_ALLOWED_KINDS`, `field_ownership_conflict` 에러 이름, `cywell-copilot` fieldManager 문자열 모두 모든 Task에서 일치. `force` 필드명 일치.
- **Open items:** Task 1.5 Step 3의 `_manifest_from_request` — `OcpActionRequestRecord` 실제 schema에 따라 경로가 달라질 수 있음. 구현자가 해당 파일을 확인 후 top-level 필드 또는 `metadata["manifest_yaml"]` 중 실제 저장 형태를 선택. 이는 구현 시점에만 결정 가능한 localization이라 플랜 범위 밖.

# OCP YAML Apply & Live Agent Redesign — Design

- Date: 2026-04-19
- Branch: dev-ver2
- Author: JungyuOO (w/ Claude brainstorming)
- Status: Design approved, plan pending

## 1. Problem

현재 시스템 상태:

- `apps/api/api/routes/ocp.py`의 `resource-detail`은 **GET only** — Resources 탭에서 YAML을 보여주기만 하고 편집·apply 경로가 없음. 프론트(`apps/web/src/pages/resources/ResourcesPage.tsx:162`)도 `<pre><code>{manifestYaml}</code></pre>` 뷰어에 그침.
- `apps/api/integrations/ocp/action_execution_service.py`의 action 파이프라인은 `scale_deployment / rollout_restart / log_bundle_read` 3종을 preview → request → approve → execute로 태우지만 execute가 전부 `dryRun=All`로 호출되므로 **실제 cluster mutation은 발생하지 않음**.
- `apps/api/integrations/ocp/live_chat_service.py`는 rule-based intent(`list / overview / manifest_detail`)라 "우리 `foo-app` pod yaml 뭐야?"는 manifest_detail 룰에 걸릴 때만 동작하고, "공식문서와 내 yaml 차이"류 복합 질의는 전혀 처리 불가.

목표:

1. 사용자가 Resources 탭에서 YAML을 편집하고 **실제 OCP 서버에 반영**할 수 있어야 한다.
2. 챗봇이 연결된 클러스터의 실제 YAML을 가져와서 답변할 수 있어야 한다 ("내 `X` pod yaml 보여줘").
3. 챗봇이 공식 문서와 실제 클러스터 YAML을 비교해 답변할 수 있어야 한다 ("공식문서랑 내 yaml 차이 알려줘").

## 2. Approach

3개의 병렬 트랙으로 구성한다.

- **Track A · YAML write path (backend)** — `OcpActionType`에 `yaml_apply` 추가. 기존 preview → request → approve → execute 파이프라인을 그대로 재사용하고 execute 단계에서 Server-Side Apply(SSA)로 실제 반영. 기존 `scale_deployment / rollout_restart`도 execute에서 `dryRun=All`을 제거해 실반영으로 전환. preview 단계는 `dryRun=All`을 유지해 API server의 스키마/admission webhook 검증을 선행.
- **Track B · Live agent LLM tool-use (backend)** — `live_chat_service.py`의 rule-based 로직을 `LiveAgent` LLM tool-use로 교체. 도구 4종(`list_namespaces`, `list_resources`, `get_resource_yaml`, `get_overview`)은 기존 `ConnectedOcpService`를 wrapping. mixed lane은 agent의 tool 결과 + doc retrieval chunks를 한 프롬프트로 합쳐 LLM이 synthesis.
- **Track C · Resources 탭 편집 UX (frontend)** — `deployments / services / routes`에 한해 Edit 모드. Monaco editor로 draft 편집, Preview 모달에서 diff + dryRun status 표시, Apply 시 request→approve→execute 체인 자동 실행.

### 2.1 공통 전제

- 단일 사용자(dev) 기준. approve는 UI에서 "Confirm apply" 한 번 누르는 걸로 자동 처리. audit record는 남김.
- RBAC은 연결된 OCP 프로필 토큰 권한 그대로 의존. write 권한 없으면 API server 403 → UI 배너.
- Apply 메커니즘은 SSA (`PATCH` + `Content-Type: application/apply-patch+yaml` + `fieldManager=cywell-copilot`). 필드 소유권 conflict(409)는 UI에서 명시적 "force apply" 재시도.

## 3. Track A — Backend YAML write path

### 3.1 Schema (`apps/api/api/schemas/actions.py`)

- `OcpActionType`에 `yaml_apply` 추가.
- `OcpActionPreviewRequest.payload` 확장: `manifest_yaml: str`, `resource_version: str | None`.
- `OcpActionPreviewResponse` 확장: `diff_unified: str`, `dry_run_status: Literal["ok", "rejected"]`, `dry_run_messages: list[str]`.
- `OcpActionExecuteRequest`에 `force: bool = False` 추가.

### 3.2 Policy (`apps/api/integrations/ocp/action_policy_service.py`)

- `yaml_apply`를 허용 action 목록에 추가.
- 상수 `ALLOWED_KINDS = {"deployments", "services", "routes"}` 정의. 그 외 kind 요청 시 `ValueError` → route에서 400.
- manifest YAML에서 파싱한 kind/metadata.namespace/name이 request의 namespace와 불일치하면 400.

### 3.3 Preview 확장 (`apps/api/integrations/ocp/action_preview_service.py`)

`yaml_apply` 분기:

1. manifest YAML 파싱 (PyYAML). apiVersion/kind/metadata.{namespace,name} 검증.
2. 현재 리소스 GET으로 baseline 조회.
3. SSA dryRun PATCH 호출: `PATCH /apis/.../{name}?dryRun=All&fieldManager=cywell-copilot`, `Content-Type: application/apply-patch+yaml`, body=manifest YAML.
4. dryRun 응답 manifest vs baseline → `difflib.unified_diff` → `diff_unified`.
5. dryRun이 4xx 반환하면 HTTP 실패로 올리지 않고 `dry_run_status="rejected"` + `dry_run_messages=[에러 메시지]`로 body에 담아 반환. 사용자가 왜 실패인지 UI에서 볼 수 있어야 함.

### 3.4 Execute 확장 (`apps/api/integrations/ocp/action_execution_service.py`)

- `yaml_apply` 분기: preview와 동일 SSA PATCH를 `dryRun` 파라미터 없이 호출. `OcpActionExecutionRecord.details`에 응답 manifest 저장.
- 기존 함수 rename: `_execute_scale_dry_run` → `_execute_scale`, `_execute_rollout_restart_dry_run` → `_execute_rollout_restart`. 내부에서 `params={"dryRun": "All"}` 제거. 로그 문구 `"Dry-run API PATCH succeeded"` → `"API PATCH succeeded"`.
- 409 응답 → `ActionExecutionError("field_ownership_conflict", checks=[...])` raise. force=True일 때만 SSA query에 `&force=true` 첨부.

### 3.5 Audit

- `action_audit_service` 구조 그대로 사용. audit 엔트리에 `action_type=yaml_apply`, `resource={kind,namespace,name}`, `diff_unified` 스냅샷 포함.

### 3.6 Routes

- 신규 엔드포인트 없음. 기존 `/actions/preview`, `/actions/requests`, `/actions/requests/{id}/approve`, `/actions/requests/{id}/execute` 그대로 재사용.

## 4. Track B — Live agent LLM tool-use

### 4.1 신규 컴포넌트

`apps/api/rag/query/live_agent.py` (LLM client가 이미 rag/query에 있으므로 의존성 방향상 이쪽에 둔다).

```python
class LiveAgent:
    def __init__(self, llm_client, live_service: ConnectedOcpService): ...
    async def run(self, *, message: str, connection_id: str, namespace: str,
                  broker: OcpConnectionBroker, progress=None) -> LiveAgentResult: ...
```

### 4.2 Tools

- `list_namespaces()` → `{namespaces: [name, ...]}`
- `list_resources(kind: Literal["pods","deployments","services","routes","events"], namespace: str)` → `{items: [{name, namespace, status_summary}, ...]}`
- `get_resource_yaml(kind, namespace, name)` → `{manifest_yaml: str, resource_version: str}`
- `get_overview()` → `{resource_counts, default_namespace, namespace_count}`

모두 기존 `ConnectedOcpService` 메서드를 thin wrapping — 네트워크 로직 중복 없음.

### 4.3 Agent loop

- `MAX_TOOL_ITERATIONS = 3` (무한 루프 가드).
- 각 iteration: LLM 호출 → `tool_use` 블록 있으면 디스패치 → `tool_result`를 메시지에 append → 재호출. tool_use 없으면 종료.
- 전체 agent 실행 20초 타임아웃 → 초과 시 현재까지 수집된 tool 결과만 넣고 fallback prompt로 한 번 더 호출해 텍스트 응답 생성.
- 각 tool 호출/응답을 **stage progress**로 emit (`fetch_resource_manifest`, `fetch_live_resources`, `fetch_cluster_overview` 등).

### 4.4 System prompt 뼈대

- "You are a live cluster assistant. Use tools to inspect the connected OpenShift cluster. Never invent resource names. If the user's namespace or resource is ambiguous, call `list_resources` first."
- Tool 응답이 빈 리스트면 "검색된 리소스 없음"으로 명시적으로 답변 — 환각 금지.
- 질문 언어(한/영) 일관성 유지.

### 4.5 `LiveOcpChatService.answer` 교체

- 기존 `_detect_intent / _format_overview_answer / _format_list_answer` 및 `_QueryIntent` dataclass 제거.
- `LiveAgent.run(...)` 호출 → `OcpLiveChatResponse`로 매핑. `items`는 agent가 실제로 호출한 tool 결과 기반으로 "사용자에게 보여줄 리소스"만 추출.

### 4.6 Mixed lane 통합 (`apps/api/rag/generation/unified_copilot_service.py`)

- `QueryRouter`의 lane 결정은 그대로(`live/mixed/doc/needs_connection`). LiveAgent는 live lane 내부 구현 변경이다.
- `_build_mixed_response`는 단순 concat을 버리고 "공식문서 청크 N개 + live YAML/리소스 M개"를 한 synthesis prompt로 LLM에 한 번 더 호출. **+1 LLM turn latency(~1–2초) 수용**. 이게 "공식문서 vs 내 yaml 차이" 답변의 실체.
- doc lane citation `[N]`은 보존. live YAML은 ```yaml``` 인라인 코드블록으로 삽입.

## 5. Track C — Frontend Resources tab editor

### 5.1 진입점 (`apps/web/src/pages/resources/ResourcesPage.tsx`)

- YAML Manifest 카드 헤더에 **"Edit" 토글 버튼**. `resource ∈ {deployments, services, routes}`일 때만 렌더. `pods / events`는 렌더하지 않는다.
- Edit 모드 진입 시 `resourceDetail.manifestYaml`을 draft로 복사 → editor로 교체.

### 5.2 Editor

- 기본 **Monaco** (`@monaco-editor/react`). YAML 하이라이트 + diff view(`DiffEditor`) 재사용.
- `package.json`에 Monaco가 없으면 의존성 추가. CodeMirror가 이미 들어와 있으면 CodeMirror로 통일(구현 단계에서 확정).
- Fallback: plain `<textarea>` + prism CSS (저사양 환경용).

### 5.3 State 추가

- `editMode: boolean`
- `draftYaml: string`
- `preview: OcpActionPreviewResponse | null`
- `previewLoading, applyLoading, applyError`
- `pendingForceRetry: boolean`

### 5.4 플로우

1. Edit 클릭 → `editMode=true`, `draftYaml = resourceDetail.manifestYaml`.
2. 수정 → `draftYaml` 업데이트, "Preview" 버튼 활성화.
3. Preview 클릭 → `POST /actions/preview` `{action_type: "yaml_apply", kind, namespace, manifest_yaml, resource_version}`.
4. 응답 → **모달** 오픈:
   - 상단 status badge: `dry_run_status === "ok"` → 초록, `rejected` → 빨강 + `dry_run_messages` 리스트, Apply 버튼 disabled.
   - 본문: `diff_unified`를 Monaco DiffEditor (original=현재, modified=draft).
   - 하단: Cancel / Apply.
5. Apply 클릭 → `applyYamlAction(...)` 단일 래퍼가 (a) `POST /actions/requests` → (b) `POST /actions/requests/{id}/approve` → (c) `POST /actions/requests/{id}/execute` 연쇄 실행.
6. Execute 성공 → 토스트 "Applied. Refreshing…" → `getOcpResourceDetail` 재조회 → view 모드 복귀.
7. Execute 실패:
   - `field_ownership_conflict` (409) → 모달에 "This resource has conflicting field ownership. Force apply?" 버튼 → `{force: true}`로 execute 재호출.
   - 403 → "Your token lacks write permission. Check OCP connection RBAC." 배너. draft 유지.
   - 422 / 기타 → 에러 배너. draft 유지.

### 5.5 API 모듈

- `apps/web/src/lib/api/actionPreviewApi.ts` 확장 또는 신규 `yamlApplyApi.ts`. 내보낼 함수: `previewYamlApply`, `applyYamlAction`.
- `apps/web/src/features/actions/` 밑에 `useYamlApply()` hook 신설. 상태 + 네트워크 orchestration 격리.

### 5.6 UX 가드

- Edit 모드에서 페이지 이탈/탭 전환 시 `beforeunload` confirm ("You have unsaved YAML changes. Discard?").
- Draft 변경 시 preview state 초기화 → 재Preview 강제.
- Apply 진행 중 모달 닫기 차단.

### 5.7 Chat 탭 프론트 변경

없음. 트랙 B는 서버 응답 스키마 유지이므로 기존 ChatPage가 그대로 렌더. mixed lane 답변에 ```yaml``` 인라인 블록이 들어오니 `MarkdownArticle`가 코드블록 처리하는지 확인만 하면 된다.

## 6. Testing

### 6.1 Backend

- `tests/integrations/ocp/test_action_execution_yaml_apply.py` — httpx MockTransport로 SSA PATCH 호출 verify (Content-Type, fieldManager, dryRun 유무, force query).
- `tests/integrations/ocp/test_action_preview_yaml_apply.py` — diff 생성, dryRun rejected 응답 body 검증.
- `tests/integrations/ocp/test_action_policy_yaml_apply.py` — 허용 외 kind 400, manifest-request namespace 불일치 400.
- `tests/integrations/ocp/test_live_agent.py` — mock LLM tool-use 시나리오 4종: (a) overview only, (b) list_resources → get_resource_yaml 연쇄, (c) MAX_TOOL_ITERATIONS 초과 fallback, (d) tool 에러 graceful degrade.
- `tests/rag/generation/test_unified_copilot_mixed_synthesis.py` — mixed lane에서 live YAML + doc chunks가 synthesis prompt에 같이 들어가는지 assertion.
- 기존 scale/restart 테스트 회귀: `_dry_run` 제거 리팩토링 이후 rename 테스트 동반 수정.

### 6.2 Frontend

- vitest 스냅샷 최소만. E2E 자동화는 스코프 밖.
- 수동 checklist:
  1. Edit 토글 deployments/services/routes에만 보임
  2. 유효 YAML preview → diff 표시
  3. 고의로 깨진 YAML → rejected 배지
  4. 실제 replicas 변경 → apply → 서버 재반영 확인
  5. 동시에 다른 클라이언트가 수정 → 409 → force 재시도 성공

## 7. Sequencing

PR 3개로 분리:

1. **PR1 — Track A (backend yaml_apply + scale/restart 실반영 전환)**: backend-only, 테스트 포함.
2. **PR2 — Track C (Resources 편집 UI)**: PR1 머지 후. E2E 수동 테스트는 이 PR에서.
3. **PR3 — Track B (Live agent LLM tool-use + mixed synthesis)**: 독립이지만 도입 크기가 커서 분리. PR1/PR2와 병렬 가능.

각 PR은 독립 revert 가능해야 한다.

## 8. Observability

- Stage emit 추가: `yaml_apply_preview`, `yaml_apply_dry_run`, `yaml_apply_execute`, `yaml_apply_conflict`. 기존 progress stream 구조 재사용.
- 로그 포맷:
  - `[yaml_apply] kind=%s ns=%s name=%s fieldManager=%s elapsed=%.2fs`
  - `[yaml_apply] field_ownership_conflict ... force=%s`
  - `[live_agent] iterations=%d tools=%s elapsed=%.2fs`
  - `[live_agent.tool] name=%s args_summary=%s elapsed=%.2fs`

## 9. Out of scope

- configmaps / secrets 편집 (다음 스펙)
- pods exec / rsh / logs tail UI (다음 스펙)
- 멀티유저 approval workflow (단일 사용자 가정 유지)
- YAML 편집 undo 스택
- Helm / Kustomize 인식

## 10. Open items (구현 단계 판단)

- Monaco vs 기존 코드 에디터 의존성 통일 — `apps/web/package.json` 확인 후 결정.
- `OcpActionExecuteRequest.force` 필드 네이밍이 정책 서비스의 다른 플래그와 충돌 없는지 구현 시 확인.
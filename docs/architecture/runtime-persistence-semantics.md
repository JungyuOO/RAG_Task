# Runtime Persistence Semantics

작성일: 2026-04-15

## 목적

새 앱의 runtime persistence를 `development`와 `production`에서 같은 의미로 취급하지 않기 위한 운영 기준이다.
현재 구현은 SQLite-backed runtime metadata를 유지하되, 운영 모드에 따라 legacy migration과 secret backend 기본값을 다르게 해석한다.

## 현재 구현 기준

런타임 초기화는 `apps/api/runtime.py`에서 직접 경로를 만들지 않고 `apps/api/runtime_persistence.py`의 profile을 통해 결정한다.

지원 env:

- `RAG_TASK_RUNTIME_PERSISTENCE_MODE=development|production`
- `RAG_TASK_RUNTIME_STATE_DIR`
- `RAG_TASK_RUNTIME_DB_PATH`
- `RAG_TASK_RUNTIME_SECRET_PATH`
- `RAG_TASK_RUNTIME_SECRET_REFS_PATH`
- `RAG_TASK_RUNTIME_ENABLE_LEGACY_MIGRATION`
- `RAG_TASK_RUNTIME_REQUIRE_MANAGED_SECRETS`

### Development semantics

- 기본 mode
- state root 기본값: `data/runtime_state`
- legacy JSON -> SQLite migration 기본 허용
- secret backend는 local protected file / DPAPI / env-key 중 현재 환경에서 가능한 기본값 허용
- 목표: 로컬 개발, 단일 프로세스, 빠른 실험, 이전 JSON state에서의 무중단 전환

### Production semantics

- `RAG_TASK_RUNTIME_PERSISTENCE_MODE=production`
- legacy JSON migration 기본 비활성화
- managed secret backend 필수
  - 허용: `vault_http`, `vault_hashicorp`, `env_key`
  - `RAG_TASK_SECRET_MASTER_KEY`만 있으면 `env_key`를 managed default로 사용
- 목표: 예측 가능한 startup, 명시적 secret posture, legacy file 의존 차단

## 현재 SQLite를 유지해도 되는 범위

아래 조건이면 SQLite runtime store를 계속 control-plane metadata store로 유지한다.

- 단일 앱 프로세스 또는 매우 제한된 동시성
- action execution과 batch indexing이 요청 수보다 많지 않음
- 실행 lane이 주로 dry-run / read-only 중심
- 재시도, 작업 claim, dead-letter queue가 아직 필요 없음
- 감사 로그와 상태 조회가 동일 프로세스에서 충분히 처리 가능

현재 코드베이스는 이 범위에 들어간다.

## Queue/Worker-grade persistence로 넘어가야 하는 조건

아래 중 2개 이상이 동시에 나타나면 queue/worker lane으로 전환한다.

- long-running batch indexing이 여러 개 동시에 실행됨
- action execution이 dry-run을 넘어 실제 mutation/remote executor로 커짐
- 프로세스 재시작 후 “실행 중” job claim/lease 복구가 필요함
- worker를 여러 프로세스나 별도 컨테이너로 분리해야 함
- retry backoff, dead-letter, poison job 격리가 필요함
- SQLite lock contention 또는 단일 프로세스 병목이 관측됨

## 권장 전환 방식

전환 시에도 SQLite를 버리지 않는다. 역할을 분리한다.

- SQLite:
  - connection profile metadata
  - action request / approval / audit record
  - batch job summary / execution summary / UI polling source
- Queue / worker backend:
  - durable work enqueue
  - worker claim / heartbeat / lease
  - retry / dead-letter / cancellation
  - long-running execution output stream

즉, SQLite는 control plane metadata store로 유지하고, queue는 execution plane으로 추가한다.

## 다음 구현 우선순위

1. runtime profile을 status/debug surface에 노출
2. long-running lane에 job lease/claim 개념 추가 여부 재평가
3. 실제 mutation execution lane이 생길 때 queue backend를 먼저 도입

## 비범위

- 이번 단계에서는 queue broker 자체를 도입하지 않는다.
- 이번 단계에서는 worker process 분리나 remote executor는 구현하지 않는다.


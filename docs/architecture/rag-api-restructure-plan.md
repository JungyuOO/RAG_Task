# RAG API Restructure Plan

## Purpose

`apps/api`를 보통의 RAG 시스템 구조에 가깝게 재편하되, 현재 `apps/web`와의 연결을 깨지 않고 단계적으로 옮기는 계획을 정리한다.

이 문서는 기능 추가 문서가 아니라 구조 정리 문서다.

## Current Issues

- `apps/api` 루트에 runtime, logging, settings 성격 파일이 흩어져 있다.
- `routes/v1/*` 구조가 한 단계 과하다.
- `services/*`가 사실상 feature 단위인데 `chat`, `retrieval`, `ingestion`, `ocp`로만 나뉘어 있어 책임이 섞여 보인다.
- `repositories/*`가 persistence 책임을 가지는데 이름이 너무 generic하다.
- 빈 폴더가 남아 있다.
  - `observability/`
  - `settings/`
  - `workers/`
  - `services/memory/`
- 프론트가 의존하는 API surface는 안정화되어야 하므로 내부 구조만 먼저 바꿔야 한다.

## Non-Goals

- API path 변경
- 프론트 요청/응답 포맷 변경
- 도메인 로직 재설계
- live OCP 기능 축소

## Target Shape

```text
apps/api/
  main.py
  app_factory.py

  core/
    config.py
    logging.py
    runtime.py
    runtime_persistence.py
    text_utils.py
    llm_settings.py
    pgvector_settings.py

  api/
    routes/
      actions.py
      auth.py
      chat.py
      docs_preview.py
      indexing.py
      library.py
      ocp.py
    schemas/
      actions.py
      auth.py
      chat.py
      indexing.py
      library.py
      ocp.py

  rag/
    query/
      intent.py
      rewrite.py
      planner.py
      memory.py
    retrieval/
      sparse.py
      dense.py
      hybrid.py
      rerank.py
      embeddings.py
      filters.py
    generation/
      llm_client.py
      synthesis.py
      citation_grounding.py
      cache.py
    indexing/
      service.py
      batch.py
      jobs.py
      reset.py
      writer.py
      parser/
      normalize/
      enrich/
      chunk/

  integrations/
    ocp/
      auth.py
      live_query.py
      actions.py
    storage/
      pgvector.py
      sqlite.py
      artifacts.py
      secrets.py

  tests/
```

## Mapping

### Core

- `runtime.py` -> `core/runtime.py`
- `runtime_persistence.py` -> `core/runtime_persistence.py`
- `logging_setup.py` -> `core/logging.py`
- `llm_settings.py` -> `core/llm_settings.py`
- `pgvector_runtime_settings.py` -> `core/pgvector_settings.py`
- `text_utils.py` -> `core/text_utils.py`

### API

- `routes/v1/actions/routes.py` -> `api/routes/actions.py`
- `routes/v1/auth/routes.py` -> `api/routes/auth.py`
- `routes/v1/chat/routes.py` -> `api/routes/chat.py`
- `routes/v1/docs_preview/routes.py` -> `api/routes/docs_preview.py`
- `routes/v1/index/routes.py` + `routes/v1/index/batch_routes.py` -> `api/routes/indexing.py`
- `routes/v1/library/routes.py` -> `api/routes/library.py`
- `routes/v1/ocp/routes.py` -> `api/routes/ocp.py`

- `schemas/copilot_chat.py` -> `api/schemas/chat.py`
- `schemas/ocp_*` -> `api/schemas/ocp.py`
- `schemas/batch_*`, `schemas/indexing.py`, `schemas/index_admin.py`, `schemas/ingestion_parser.py` -> `api/schemas/indexing.py`
- `schemas/library_*` -> `api/schemas/library.py`
- `schemas/ocp_action*`, `schemas/ocp_actions.py` -> `api/schemas/actions.py`

### RAG Query

- `services/chat/intent_agent.py` -> `rag/query/intent.py`
- `services/chat/query_rewrite_agent.py` -> `rag/query/rewrite.py`
- `services/chat/chat_memory.py` -> `rag/query/memory.py`
- `unified_copilot_service.py` 안의 lane planning 일부 -> `rag/query/planner.py`

### RAG Retrieval

- `services/retrieval/document_retriever.py` -> `rag/retrieval/sparse.py`
- `services/retrieval/pgvector_bridge.py` -> `rag/retrieval/dense.py`
- `unified_copilot_service.py` 안의 merge 로직 -> `rag/retrieval/hybrid.py`
- `unified_copilot_service.py` 안의 rerank 로직 -> `rag/retrieval/rerank.py`
- `services/retrieval/embedding_clients.py` -> `rag/retrieval/embeddings.py`

### RAG Generation

- `services/chat/llm_client.py` -> `rag/generation/llm_client.py`
- `services/chat/citation_grounding.py` -> `rag/generation/citation_grounding.py`
- `services/chat/response_cache.py` -> `rag/generation/cache.py`
- `services/chat/unified_copilot_service.py` 안의 synthesis 로직 -> `rag/generation/synthesis.py`

### RAG Indexing

- `services/ingestion/index_service.py` -> `rag/indexing/service.py`
- `services/ingestion/batch_index_service.py` -> `rag/indexing/batch.py`
- `services/ingestion/batch_job_service.py` -> `rag/indexing/jobs.py`
- `services/ingestion/index_admin_service.py` -> `rag/indexing/reset.py`
- `services/ingestion/index_writer_bridge.py` -> `rag/indexing/writer.py`
- `services/ingestion/parsers/*` -> `rag/indexing/parser/*`
- `services/ingestion/normalize/*` -> `rag/indexing/normalize/*`
- `services/ingestion/enrich/*` -> `rag/indexing/enrich/*`
- `services/ingestion/chunking/*` -> `rag/indexing/chunk/*`

### Integrations

- `services/ocp/*` -> `integrations/ocp/*`
- `repositories/sqlite_runtime_repositories.py` -> `integrations/storage/sqlite.py`
- `repositories/normalized_artifact_repository.py` -> `integrations/storage/artifacts.py`
- `services/retrieval/pgvector_store.py` -> `integrations/storage/pgvector.py`
- `repositories/protected_secret_persistence.py`, vault helpers -> `integrations/storage/secrets.py`

## Frontend Contracts To Preserve

다음 프론트 연결은 절대 깨지면 안 된다.

- [copilotChatApi.ts](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/apps/web/src/lib/api/copilotChatApi.ts)
- [libraryApi.ts](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/apps/web/src/lib/api/libraryApi.ts)
- [ocpConnectionApi.ts](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/apps/web/src/features/auth/api/ocpConnectionApi.ts)

유지 대상:

- `/api/v1/chat/query`
- `/api/v1/chat/query/stream`
- `/api/v1/chat/live`
- `/api/v1/library/*`
- `/api/v1/index/*`
- `/api/v1/auth/*`
- `/api/v1/ocp/*`
- 요청/응답 JSON shape
- chat stream event type
  - `stage`
  - `answer_delta`
  - `result`
  - `error`

## Low-Risk First Pass

### Phase 1

- 빈 폴더 삭제
  - `observability/`
  - `settings/`
  - `workers/`
  - `services/memory/`
- `routes/v1`를 `api/routes`로 평탄화
- `schemas`를 `api/schemas`로 이동

### Phase 2

- `repositories`를 `integrations/storage`로 이동
- `runtime`, `logging`, `settings` 성격 파일을 `core`로 이동
- import path만 수정

### Phase 3

- `services/chat`, `services/retrieval`, `services/ingestion`를 `rag/*`로 이동
- `services/ocp`를 `integrations/ocp`로 이동
- `unified_copilot_service.py`를 더 작게 분해

### Phase 4

- 테스트 경로를 feature 기준으로 재배치
  - `tests/unit/chat`
  - `tests/unit/retrieval`
  - `tests/unit/indexing`
  - `tests/unit/ocp`

## High-Risk Areas

- `runtime.py`는 import fan-out가 커서 마지막에 옮기는 편이 안전하다.
- `unified_copilot_service.py`는 현재 chat/query/retrieval/generation이 섞여 있어 분리 시 회귀 위험이 높다.
- `routes/v1/index/*`는 `batch`와 `source` 인덱싱이 같이 묶여 있어 route 재배치 시 smoke test가 필요하다.
- `integrations/storage/sqlite.py`로 이동 시 배치 job과 OCP request/audit repository를 함께 옮겨야 한다.

## Verification

각 단계마다 아래를 확인한다.

- `apps/web` `npm run build`
- 핵심 Python unit tests
  - chat
  - retrieval
  - indexing
- `docker compose up -d --build app`
- `/healthz`
- chat smoke
- batch reindex smoke

## Done Definition

- `apps/api` 루트에는 진입점과 핵심 설정 파일만 남는다.
- `routes/v1` 계층이 사라진다.
- 빈 폴더가 제거된다.
- RAG 관련 코드는 `rag/` 아래로 모인다.
- OCP/DB/secret adapter는 `integrations/`로 분리된다.
- 프론트 API contract는 그대로 유지된다.

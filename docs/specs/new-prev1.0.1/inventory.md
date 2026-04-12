# RAG Task new-prev1.0.1 Inventory

작성일: 2026-04-10  
상태: Draft

관련 문서:

- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\spec.md)
- [plan.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\plan.md)

## 1. 목적

이 문서는 현재 리포지토리의 실제 참조 관계를 기준으로, `newv1.0.1` 재설계 전에 파일 단위 유지/삭제/통합 후보를 정리하기 위한 인벤토리 문서다.

기준:

- 현재 import/참조 관계
- 현재 런타임 연결 상태
- `newv1.0.1`의 4.21 단일 버전 전략
- 기능 중복 여부
- 책임 과밀 여부

## 2. 탑레벨 구조

현재 주요 디렉터리:

- `app/api`
- `app/llm`
- `app/rag`
- `app/session`
- `app/storage`
- `app/web`
- `scripts`
- `tests`

핵심 진단:

- `app/rag`에 retrieval, streaming, answer, indexing, memory glue 책임이 과도하게 몰려 있다.
- 버전 관련 기능은 API 라우트에 일부 남아 있으나, 실제 컨테이너에는 연결이 불완전하다.
- 새 버전에서는 `4.21 단일 버전` 전략이므로 버전 레이어는 대폭 축소될 가능성이 높다.

## 3. 유지 대상

### 3.1 앱 진입 / 설정

- `app/main.py`
- `app/app_factory.py`
- `app/config.py`
- `app/dependencies.py`

판단:

- 유지
- 단, `app_factory.py`는 startup/lifecycle 재설계 필요

### 3.2 API 계층

- `app/api/routes.py`
- `app/api/routes_chat.py`
- `app/api/routes_library.py`
- `app/api/routes_session.py`
- `app/api/routes_shared.py`
- `app/api/schemas.py`

판단:

- 유지
- 단, `routes_library.py`는 library status / chunk status / version 제거 여부를 기준으로 재정리 필요

### 3.3 LLM 계층

- `app/llm/base_agent.py`
- `app/llm/intent_agent.py`
- `app/llm/retrieval_agent.py`
- `app/llm/answer_rewrite_agent.py`
- `app/llm/__init__.py`

판단:

- 유지
- 단, 질문 분류 / query interpretation / answer rewrite 책임 재정리 필요

### 3.4 세션 계층

- `app/session/repository.py`
- `app/session/state.py`
- `app/session/store_sql.py`

판단:

- 유지
- `state.py`는 핵심 자산
- `store_sql.py`는 session persistence 구조 정리 대상이나 개념 유지

### 3.5 저장소 계층

- `app/storage/cache_repository.py`
- `app/storage/task_repository.py`
- `app/storage/vector_store.py`
- `app/storage/__init__.py`

판단:

- 유지
- 단, `vector_store.py`는 `pgvector` 기반으로 재설계 필요

### 3.6 프론트 계층

- `app/web/index.html`
- `app/web/js/*.js`

판단:

- 유지
- 전체 UI/디자인 재설계 필요

## 4. 삭제 후보

### 4.1 버전 저장소 / 버전 API

- `app/storage/version_store.py`
- `app/rag/version_manager.py`
- `app/api/routes_library.py` 내 version endpoints (`/api/library/versions*`)

근거:

- `routes_library.py`는 `container.version_store`를 기대하지만, 현재 `AppContainer`에는 `version_store`가 없다.
- 즉 이 경로는 현재 런타임에서 일관되게 연결되지 않은 상태다.
- `newv1.0.1`은 `4.21 단일 버전` 전략이므로 기능적 중요도도 낮다.

판단:

- 삭제 후보 1순위

### 4.2 다중 버전 관련 상태/분기

- `selected_versions`를 중심으로 한 다중 버전 분기
- version clarification 흐름
- 버전 혼합 retrieval 보조 경로 일부

근거:

- 새 스펙은 4.21 단일 버전 운영

판단:

- 삭제 후보
- 단, 제거는 `newv1.0.1 Phase 6`에서 일괄 수행

## 5. 통합 후보

### 5.1 Retrieval 계층 통합

파일:

- `app/rag/retrieval.py`
- `app/rag/retrieval_service.py`
- `app/rag/retrieval_state_builder.py`
- `app/rag/pipeline_scoring.py`

판단:

- 통합 후보
- retrieval planning / candidate filtering / rerank policy / context selection을 하나의 retrieval orchestration 계층으로 정리 가능

### 5.2 Answer 계층 통합

파일:

- `app/rag/answer.py`
- `app/rag/answer_format.py`
- `app/rag/answer_citation.py`
- `app/rag/answer_inline_citation.py`

판단:

- 통합 후보
- answer assembly / sanitize / citation / final payload를 하나의 answer layer로 정리 가능

### 5.3 Indexing 계층 통합

파일:

- `app/rag/index.py`
- `app/rag/indexing.py`
- `app/rag/cache.py`
- `app/storage/vector_store.py`

판단:

- 통합 후보
- ingest / embedding / persistence / cache invalidation을 명확히 분리해야 함

## 6. 분리 후보

### 6.1 Streaming orchestration

파일:

- `app/rag/pipeline_streaming.py`

현재 상태:

- turn classification
- transform lane
- retrieval lane
- answer emit
- cache
- clarification

이 한 파일에 몰려 있다.

판단:

- 분리 후보 1순위

### 6.2 Runtime support / pipeline core

파일:

- `app/rag/pipeline.py`
- `app/rag/pipeline_runtime_support.py`

판단:

- pipeline core / dependency builder / runtime helper를 다시 나눌 필요가 있음

## 7. 조건부 유지 후보

### 7.1 Ollama embedding 경로

파일:

- `app/rag/bge_embeddings.py`

현 상태:

- 현재 `.env`는 `EMBEDDING_BACKEND=tei`
- 런타임 주 경로는 TEI

판단:

- 조건부 유지 후보
- 완전 제거 전, 로컬 fallback 또는 개발용 유지 여부 결정 필요

### 7.2 기본 chunk helper

파일:

- `app/rag/chunking.py`
- `app/rag/chunking_markdown.py`
- `app/rag/chunking_markdown_support.py`

판단:

- 유지 가능성이 높음
- 단, HTML 전환 이후 chunk schema 기준 재정리 필요

## 8. 즉시 주의 대상

### 8.1 런타임 미연결 경로

- `version_store` 관련 API

### 8.2 역할 과밀 파일

- `pipeline_streaming.py`
- `retrieval_state_builder.py`
- `pipeline_scoring.py`
- `answer_format.py`

### 8.3 레거시 출력 / 깨진 문자열 가능성

- 일부 문서/문자열은 PowerShell 출력에서 깨져 보일 수 있음
- 저장 파일은 UTF-8 기준으로 재검토 필요

## 9. 라이브러리 메모

현재 핵심 의존성:

- `fastapi`
- `httpx`
- `numpy`
- `pydantic-settings`
- `pymupdf`
- `psycopg2-binary`
- `sentence-transformers`
- `uvicorn`

추가 검토 대상:

- `pgvector`

판단:

- `newv1.0.1`에서는 `pgvector` 채택 방향이 맞음

## 10. 다음 작업

다음 단계에서 해야 할 일:

1. 이 문서를 기준으로 `plan.md Phase 1`에 파일 단위 분류를 반영
2. `version_store` 제거 여부 확정
3. `pipeline_streaming.py` 분리 설계 초안 작성
4. `pgvector` 도입 범위 확정

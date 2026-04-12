# RAG Task new-prev1.0.1 Decisions

작성일: 2026-04-10  
상태: Draft

관련 문서:

- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\spec.md)
- [plan.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\plan.md)
- [inventory.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\inventory.md)
- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\newv1.0.1\spec.md)

## 1. 목적

이 문서는 `new-prev1.0.1` 단계에서 확정한 현재 코드베이스 분류 결과를 기록한다.

분류 범주:

- 유지 확정
- 삭제 확정
- 통합 대상
- 분리 대상
- 조건부 보류

이 문서는 실제 코드 삭제/병합 작업 전에 기준선을 고정하는 역할을 한다.

## 2. 유지 확정

### 2.1 앱 진입 / 설정

- `app/main.py`
- `app/app_factory.py`
- `app/config.py`
- `app/dependencies.py`

판단:

- 유지 확정
- 단, lifecycle / 설정 항목은 `newv1.0.1`에서 재구성

### 2.2 API 진입 계층

- `app/api/routes.py`
- `app/api/routes_chat.py`
- `app/api/routes_library.py`
- `app/api/routes_session.py`
- `app/api/routes_shared.py`
- `app/api/schemas.py`

판단:

- 유지 확정
- 단, route 수와 책임은 축소/재배치 가능

### 2.3 LLM 계층

- `app/llm/base_agent.py`
- `app/llm/intent_agent.py`
- `app/llm/retrieval_agent.py`
- `app/llm/answer_rewrite_agent.py`
- `app/llm/__init__.py`

판단:

- 유지 확정
- 단, 프롬프트와 역할 분리는 재정리

### 2.4 세션 계층

- `app/session/repository.py`
- `app/session/state.py`
- `app/session/store_sql.py`

판단:

- 유지 확정
- `state.py`는 핵심 자산

### 2.5 저장소 계층

- `app/storage/cache_repository.py`
- `app/storage/task_repository.py`
- `app/storage/vector_store.py`
- `app/storage/__init__.py`

판단:

- 유지 확정
- `vector_store.py`는 `pgvector` 기반 재구성 전제

### 2.6 프론트 계층

- `app/web/index.html`
- `app/web/js/chat.js`
- `app/web/js/library.js`
- `app/web/js/citation.js`
- `app/web/js/shared.js`
- `app/web/js/app.js`
- `app/web/js/status.js`
- `app/web/js/session.js`

판단:

- 유지 확정
- 단, UI/레이아웃/상태 표현/렌더링은 전면 재설계

## 3. 삭제 확정

### 3.1 다중 버전 저장소 / API

- `app/storage/version_store.py`
- `app/rag/version_manager.py`
- `app/api/routes_library.py` 내 버전 관련 endpoint

삭제 이유:

- 현재 런타임 컨테이너와 완전히 일관되게 연결되어 있지 않다.
- `newv1.0.1`은 `4.21 단일 버전` 전략이다.
- 유지 비용 대비 가치가 낮다.

### 3.2 다중 버전 분기 로직

- `selected_versions` 중심 분기
- version clarification 흐름
- 다중 버전 선택 UI

삭제 이유:

- 4.21 단일 버전 전략과 충돌

### 3.3 레거시 문서/결과

- 기존 `docs/specs/v*`
- 기존 `tests/results/v*`

삭제 이유:

- 현재 재설계 기준과 섞이면 판단을 흐린다.
- 단, 필요한 근거는 별도 보관 후 정리 가능

## 4. 통합 확정 대상

### 4.1 Retrieval 계층

통합 대상:

- `app/rag/retrieval.py`
- `app/rag/retrieval_service.py`
- `app/rag/retrieval_state_builder.py`
- `app/rag/pipeline_scoring.py`

목표:

- candidate generation
- filtering
- source anchoring
- rerank policy
- context selection

을 하나의 retrieval orchestration 계층으로 정리

### 4.2 Answer 계층

통합 대상:

- `app/rag/answer.py`
- `app/rag/answer_format.py`
- `app/rag/answer_citation.py`
- `app/rag/answer_inline_citation.py`

목표:

- answer assembly
- answer sanitize
- citation
- final render payload

을 하나의 answer layer로 정리

### 4.3 Indexing 계층

통합 대상:

- `app/rag/index.py`
- `app/rag/indexing.py`
- `app/rag/cache.py`
- `app/storage/vector_store.py`

목표:

- ingest
- embedding
- vector persistence
- cache invalidation

을 명확한 indexing stack으로 정리

## 5. 분리 확정 대상

### 5.1 Streaming orchestration

분리 대상:

- `app/rag/pipeline_streaming.py`

현재 문제:

- transform lane
- retrieval lane
- streaming emit
- cached answer path
- clarification path

가 한 파일에 몰려 있다.

목표:

- turn orchestration
- answer transformation
- retrieval execution
- final emit

을 분리

### 5.2 Pipeline core / runtime support

분리 대상:

- `app/rag/pipeline.py`
- `app/rag/pipeline_runtime_support.py`

목표:

- core pipeline assembly
- dependency wiring
- runtime helper

분리

## 6. 조건부 보류

### 6.1 Ollama embedding fallback

- `app/rag/bge_embeddings.py`

판단:

- 조건부 보류
- 운영 경로는 TEI 우선이지만, 개발/테스트 fallback로 남길지 `newv1.0.1 Phase 3`에서 최종 결정

### 6.2 기존 chunk helper

- `app/rag/chunking.py`
- `app/rag/chunking_markdown.py`
- `app/rag/chunking_markdown_support.py`

판단:

- 조건부 유지
- HTML 기반 문서화 이후 재사용 가능성 있음

## 7. 즉시 실행 메모

이 문서는 “확정 분류” 문서다.

다음 실행 순서는:

1. `plan.md Phase 1` 완료 처리
2. `newv1.0.1 plan`에 삭제 확정/통합 확정/분리 확정 내용 이관
3. 실제 코드 삭제는 해당 Phase에 맞춰 순차 수행

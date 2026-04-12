# RAG Task new-prev1.0.1 Handoff

작성일: 2026-04-10  
상태: Draft

관련 문서:

- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\spec.md)
- [plan.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\plan.md)
- [inventory.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\inventory.md)
- [decisions.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\decisions.md)
- [dependencies.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\dependencies.md)
- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\newv1.0.1\spec.md)
- [plan.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\newv1.0.1\plan.md)

## 1. 목적

이 문서는 `new-prev1.0.1` 단계의 분석 결과를 `newv1.0.1` 본작업으로 넘기기 위한 이관 메모다.

핵심은 아래 세 가지다.

1. 무엇을 먼저 지울지
2. 무엇을 유지한 채 재설계할지
3. 어떤 순서로 `newv1.0.1` Phase를 시작할지

## 2. 즉시 삭제 후보

`newv1.0.1` 착수 시 가장 먼저 정리해야 할 대상:

- `app/storage/version_store.py`
- `app/rag/version_manager.py`
- `app/api/routes_library.py` 내 버전 관련 endpoint
- 다중 버전 선택과 연관된 UI 및 상태 분기

이유:

- 4.21 단일 버전 전략
- 현재 런타임 연결 불일치
- 유지 비용 대비 가치 부족

## 3. 반드시 유지할 핵심 자산

### 3.1 세션/토픽 상태

- `app/session/state.py`
- `app/session/repository.py`
- `app/session/store_sql.py`

이유:

- 멀티턴 품질의 핵심 기반

### 3.2 질문 분류/해석

- `app/llm/intent_agent.py`
- `app/llm/retrieval_agent.py`
- `app/llm/answer_rewrite_agent.py`

이유:

- 질문 유형 분기
- retrieval turn / transformation turn 분리

### 3.3 API 진입 계층

- `app/api/routes_chat.py`
- `app/api/routes_library.py`
- `app/api/routes_session.py`
- `app/api/routes_shared.py`
- `app/api/schemas.py`

이유:

- 기능 재배치가 있더라도 진입점 자체는 유지 필요

## 4. 가장 먼저 리팩토링할 영역

### 4.1 `pipeline_streaming.py`

우선순위: 최상

이유:

- turn orchestration
- transform lane
- retrieval lane
- cached answer path
- clarification path

가 모두 한 파일에 섞여 있다.

권장 방향:

- turn router
- transform handler
- retrieval handler
- answer emit handler

로 분리

### 4.2 retrieval 계층

우선순위: 높음

대상:

- `retrieval.py`
- `retrieval_service.py`
- `retrieval_state_builder.py`
- `pipeline_scoring.py`

권장 방향:

- retrieval orchestration 계층으로 통합

### 4.3 answer 계층

우선순위: 높음

대상:

- `answer.py`
- `answer_format.py`
- `answer_citation.py`
- `answer_inline_citation.py`

권장 방향:

- answer assembly layer로 통합

## 5. 의존성 이관

### 유지

- `fastapi`
- `httpx`
- `pydantic-settings`
- `pymupdf`
- `python-dotenv`
- `python-multipart`
- `psycopg2-binary`
- `uvicorn`

### 추가

- `pgvector`

### 조건부 보류

- `tailwindcss`
- markdown renderer 계열 프론트 라이브러리
- `sentence-transformers`
- `numpy`

정책:

- `pgvector`는 `newv1.0.1 Phase 3`의 공식 도입 대상
- 프론트 라이브러리는 `Phase 4`에서 확정
- `sentence-transformers`와 `numpy`는 Phase 1 구조 안정화 이후 재평가

## 6. newv1.0.1 시작 순서 제안

### Step 1. Phase 1 착수 전 정리

- 버전 관련 삭제 후보 제거
- `tests/results`는 비교에 필요한 범위만 유지 여부 결정

### Step 2. Phase 1

집중 대상:

- 답변 품질
- follow-up 처리
- response-shape validation
- low-signal chunk 제거
- 지연 원인 분리

### Step 3. Phase 2

집중 대상:

- HTML 문서화
- metadata 분리
- chunk schema 정비

### Step 4. Phase 3

집중 대상:

- `pgvector` 도입
- vector schema 정비

### Step 5. Phase 4

집중 대상:

- UI/디자인
- markdown renderer
- source card
- chunk viewer 상태 UX

### Step 6. Phase 5

집중 대상:

- OCP API read-only integration
- Pod/Deployment/Service/Route 상태
- YAML 보기

## 7. 검증 관점 이관

`newv1.0.1`에서는 자동 테스트만으로 PASS를 판단하지 않는다.

반드시 아래 둘을 같이 본다.

1. 테스트 기반 검증
- compile
- unit tests
- regression scenarios

2. 실제 답변 품질 검증
- 질문에 직접 답했는가
- 형식이 맞는가
- low-signal chunk가 나오지 않는가
- 멀티턴에서 문맥을 유지하는가

## 8. 최종 메모

`new-prev1.0.1`의 결과를 기준으로 보면, `newv1.0.1`은 단순 기능 추가 프로젝트가 아니라 구조 재편 프로젝트다.

즉 첫 구현부터:

- 삭제
- 통합
- 분리
- 의존성 확정

을 동시에 진행하는 것이 맞다.

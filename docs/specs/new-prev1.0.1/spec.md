# RAG Task new-prev1.0.1 Spec

작성일: 2026-04-10  
상태: Draft

## 1. 목적

`new-prev1.0.1`은 `newv1.0.1` 재설계에 들어가기 전에, 현재 리포지토리 구조를 해부하고 정리 대상을 명확히 기록하기 위한 사전 분석 문서다.

이 문서의 목적은 다음과 같다.

- 현재 코드베이스의 책임 경계를 명확히 파악한다.
- 유지할 코드와 삭제 후보 코드를 먼저 분리한다.
- 병합/분리 대상 모듈을 식별한다.
- 재설계 착수 전에 “무엇을 남기고 무엇을 없앨지”를 문서로 고정한다.

즉 이 문서는 새 제품 스펙이 아니라, 새 제품 스펙으로 가기 위한 현재 상태 해체 문서다.

## 2. 분석 범위

우선 분석 범위는 아래 디렉터리를 중심으로 한다.

- `app/api`
- `app/llm`
- `app/rag`
- `app/session`
- `app/storage`
- `app/web`
- `scripts`
- `tests`

## 3. 현재 구조 요약

현재 애플리케이션은 다음 계층으로 나뉜다.

### 3.1 `app/api`

- HTTP 라우트
- 업로드 / 라이브러리 / 세션 / 채팅 진입점
- 응답 스키마

### 3.2 `app/llm`

- intent 분류
- retrieval query 해석
- answer rewrite

### 3.3 `app/rag`

- 인덱싱
- chunking
- retrieval
- reranking
- answer formatting
- streaming orchestration
- prompt composition
- memory adapter 일부

현재 가장 큰 문제는 이 디렉터리에 너무 많은 책임이 몰려 있다는 점이다.

### 3.4 `app/session`

- 세션 repository
- topic / state 정리

### 3.5 `app/storage`

- cache repository
- vector store
- task store
- version store

### 3.6 `app/web`

- 자료실 UI
- 채팅 UI
- citation / preview / chunk viewer

## 4. 핵심 문제 진단

### 4.1 `app/rag` 과밀 문제

현재 `app/rag`에는 아래가 함께 섞여 있다.

- retrieval planning
- retrieval scoring
- answer assembly
- answer post-processing
- streaming turn orchestration
- indexing / ingestion
- prompting / memory glue

즉 “문서 검색”, “답변 생성”, “대화 orchestration”, “색인 처리”가 강하게 얽혀 있다.

### 4.2 버전 관련 복잡성

현재 구조는 여러 버전 문서를 동시에 다루기 위한 코드가 남아 있다.
그러나 `newv1.0.1`은 `4.21 단일 버전`을 목표로 하므로, 버전 선택/버전 분기/버전 메모리 중 상당수가 정리 대상이 될 가능성이 높다.

### 4.3 전처리/렌더링 분리 부족

- PDF 추출물과 retrieval 텍스트가 충분히 분리되어 있지 않다.
- 표/코드/section metadata 활용이 불완전하다.
- 프론트 렌더링과 백엔드 payload 설계가 완전히 끊어지지 않았다.

### 4.4 테스트 기준과 실제 품질 괴리

- 자동 시나리오 PASS가 실제 답변 품질을 충분히 보장하지 못했다.
- 따라서 테스트 유틸, 결과 해석, answer validation도 정리 대상이다.

## 5. 유지 대상

아래는 현재 구조에서 “형태는 바뀔 수 있어도 개념적으로 유지해야 하는 대상”이다.

### 5.1 개념 유지

- 세션/토픽 메모리
- streaming 응답
- retrieval + reranking + answer route 분기
- 공식 문서 / 고객사 문서 구분
- citation / preview 구조
- 캐시 계층

### 5.2 코드 유지 가능성이 높은 영역

- `app/api/routes_chat.py`
- `app/api/routes_library.py`
- `app/api/routes_session.py`
- `app/api/schemas.py`
- `app/llm/base_agent.py`
- `app/llm/intent_agent.py`
- `app/llm/retrieval_agent.py`
- `app/llm/answer_rewrite_agent.py`
- `app/session/repository.py`
- `app/session/state.py`
- `app/storage/cache_repository.py`
- `app/storage/task_repository.py`
- `app/web/js/*` 일부

## 6. 삭제 후보

아래는 `newv1.0.1` 기준으로 삭제 또는 대폭 축소 가능성이 높은 대상이다.

### 6.1 버전 복잡성 관련 코드

- 다중 버전 선택 UI
- 여러 버전 혼합 retrieval 보조 경로
- 버전별 비교를 위해 추가된 임시 보정 로직

### 6.2 중복/임시 경로

- 동일 역할을 가진 answer route의 중복 처리
- render 단계에서 중복되는 citation / prose / list 후처리
- 테스트용 임시 보정 경로

### 6.3 결과/문서 레거시

- 기존 `docs/specs/v*`
- 기존 `tests/results/v*`

단, 실제 삭제는 본 분석 문서 이후 `newv1.0.1` plan에 맞춰 단계적으로 수행한다.

## 7. 통합 후보

아래는 분산된 책임을 하나의 단위로 묶을 수 있는 후보들이다.

### 7.1 retrieval 계층 통합

통합 후보:

- `app/rag/retrieval.py`
- `app/rag/retrieval_service.py`
- `app/rag/retrieval_state_builder.py`
- `app/rag/pipeline_scoring.py`

목표:

- candidate 생성
- filtering
- rerank 정책
- answer-route용 context selection

을 하나의 retrieval orchestration 계층으로 재정의

### 7.2 answer 계층 통합

통합 후보:

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

### 7.3 indexing 계층 통합

통합 후보:

- `app/rag/indexing.py`
- `app/rag/index.py`
- `app/rag/cache.py`
- `app/storage/vector_store.py`
- `app/storage/cache_repository.py`

목표:

- ingest
- embedding
- vector persistence
- cache invalidation

을 명확히 분리

## 8. 분리 후보

아래는 반대로 분리해야 하는 대상이다.

### 8.1 streaming turn orchestration 분리 강화

현재 `pipeline_streaming.py`는 너무 많은 역할을 가진다.

분리 방향:

- turn classification
- transform lane
- retrieval lane
- final answer emit

### 8.2 document rendering payload 분리

현재는 retrieval와 render payload가 강하게 붙어 있다.

분리 방향:

- retrieval payload
- preview payload
- citation payload
- chunk-view payload

## 9. 라이브러리 관점

현재 `requirements.txt` 기준 핵심 라이브러리는 다음과 같다.

- `fastapi`
- `httpx`
- `numpy`
- `pydantic-settings`
- `pymupdf`
- `psycopg2-binary`
- `sentence-transformers`
- `uvicorn`

`newv1.0.1` 기준 추가 검토 대상:

- `pgvector`

정책:

- 저장/유사도 검색 엔진으로 `pgvector` 사용 허용
- 그러나 retrieval pipeline 자체는 직접 설계 유지

## 10. 결정해야 할 것

재설계 착수 전 아래를 먼저 확정해야 한다.

1. `app/rag`를 어떤 하위 계층으로 다시 쪼갤지
2. 4.21 단일 버전 기준으로 어떤 버전 코드를 삭제할지
3. HTML 문서화 파이프라인을 언제 도입할지
4. `pgvector`를 몇 Phase에서 반영할지
5. 기존 테스트 결과를 얼마나 유지할지

## 11. 이번 문서의 의미

이 문서는 “현재 코드를 어떻게 부술지”를 정하는 문서다.

즉:

- 무엇을 유지할지
- 무엇을 삭제할지
- 무엇을 합칠지
- 무엇을 다시 나눌지

를 먼저 고정하고, 그 다음 `newv1.0.1` 본작업에 들어가기 위한 기준 문서다.

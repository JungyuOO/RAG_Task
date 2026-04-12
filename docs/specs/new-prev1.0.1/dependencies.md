# RAG Task new-prev1.0.1 Dependencies

작성일: 2026-04-10  
상태: Draft

관련 문서:

- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\spec.md)
- [plan.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\plan.md)
- [decisions.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\decisions.md)
- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\newv1.0.1\spec.md)

## 1. 목적

이 문서는 `newv1.0.1` 재설계 기준으로, 현재 의존성을 유지/추가/삭제 후보로 분류하고 `pgvector` 도입 기준을 정리하기 위한 문서다.

## 2. 현재 requirements

현재 requirements:

- `fastapi`
- `httpx`
- `numpy`
- `pydantic-settings`
- `pymupdf`
- `python-dotenv`
- `python-multipart`
- `psycopg2-binary`
- `sentence-transformers`
- `uvicorn`

## 3. 유지 확정

### 3.1 백엔드 기본

- `fastapi`
- `uvicorn`
- `pydantic-settings`
- `httpx`

이유:

- API 서버
- 설정 관리
- 외부 LLM / embedding / 문서 수집 HTTP 호출

### 3.2 문서 처리

- `pymupdf`

이유:

- PDF 원본 처리 / page image / fallback preview에 계속 필요

### 3.3 업로드 / 환경 변수

- `python-multipart`
- `python-dotenv`

이유:

- 파일 업로드
- 로컬/개발 환경 설정

### 3.4 DB 연결

- `psycopg2-binary`

이유:

- PostgreSQL 연결 유지
- `pgvector` 사용 시에도 PostgreSQL 드라이버는 계속 필요

## 4. 추가 확정

### 4.1 `pgvector`

판단:

- 추가 확정

이유:

- `newv1.0.1`의 vector index 엔진으로 채택
- PostgreSQL 안에서 벡터 저장/유사도 검색 처리
- 단, retrieval pipeline은 계속 직접 설계

도입 원칙:

- “라이브러리가 알아서 다 해주는 검색기”로 가지 않는다
- `pgvector`는 저장과 유사도 계산 엔진
- chunk schema / metadata schema / candidate selection / filtering / rerank policy는 애플리케이션이 직접 담당

### 4.2 프론트엔드 스타일링/렌더링

판단:

- 추가 검토 확정

후보:

- `tailwindcss`
- markdown renderer 계열 라이브러리

이유:

- `newv1.0.1`은 UI/디자인 전면 재설계가 포함됨
- plain text 렌더링을 markdown renderer 기반으로 전환 필요
- 운영 콘솔형 정보 구조와 SaaS 수준의 polish 확보 필요

현재 단계 결론:

- 스타일링 및 markdown 렌더링 라이브러리 도입은 `newv1.0.1 Phase 4`에서 확정
- `tailwindcss` 사용은 허용 방향

## 5. 삭제 후보

### 5.1 `numpy`

판단:

- 삭제 후보

근거:

- 현재 코드에서 직접 사용 흔적이 거의 없다.
- 다만 간접 의존 가능성이 있어 즉시 제거는 보류

결론:

- 직접 의존인지 간접 의존인지 `newv1.0.1` 환경 재구성 시 최종 판단

### 5.2 `sentence-transformers`

판단:

- 부분 축소 또는 대체 후보

근거:

- 현재는 reranker(`CrossEncoder`)에 사용
- 속도 병목이 크고, 제품 구조상 조건부 사용 또는 대체 가능성 있음

결론:

- 즉시 삭제는 아님
- `newv1.0.1 Phase 1`에서 reranker 전략 재설계 후 유지 여부 재평가

## 6. Docker 관점

현재 Dockerfile은 Python slim + build-essential 기반이다.

유지 판단:

- builder / runtime 2-stage 구조는 유지 가능
- 단, `pgvector` 도입과 프론트 build 체계 도입 여부에 따라 보강 필요

추가 검토:

- frontend build 단계가 필요한지
- node toolchain이 필요한지
- markdown/html 렌더링 자산 빌드가 필요한지

## 7. Phase 2 결론

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

- `tailwindcss` 및 markdown renderer 계열 프론트 라이브러리
- `sentence-transformers`
- `numpy`

## 8. 다음 작업

1. `new-prev1.0.1`의 Phase 2 완료 근거로 사용
2. `newv1.0.1` plan에 `pgvector`와 프론트 라이브러리 도입 기준 반영
3. 실제 코드 레벨 의존성 변경은 본 구현 Phase에서 수행

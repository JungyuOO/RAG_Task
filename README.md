# RAG Task

## 프로젝트 개선사항
- 현재 구현된 화면 자료실에서 인덱싱 수랑 페이지 수만 보이는 것을 화면 상에 청크를 아예 보여지게 구현
- 지금 사용하고 있는 자료에서 Redhat의 Openshift Cloud Platfrom 공식 자료 문서를 갖고와서 해당 자료를 사용할 것이나, 실제 고객사에서 Openshift Cloud Platform 구현 방식 ex.Pod의 구성 방식 코드나, 혹은 설치 방법, 배포 방법, 같은 것은 다를 수 있음 그래서 실제 고객사에서 어떤 식으로 OCP를 운영하는 가에 대한 메뉴얼 가상 문서를 여러개 LLM을 통해 ex.GPT Model로 미리 생성해야할 것
- 사용자가 물어본 것이 한 문서에서만 있는 것이 아닌 다양한 문서에 분포한다면, 현재 기준으로는 예: 스토리지의 PV, 네트워킹의 Service에 대해 둘다 설명해줘.와 같은 질문이면 그 두개의 자료에서 둘다 갖고와야함. 단, 나오는 문장에 바로 해당 자료의 Source가 tag형식으로 달리고 그 tag를 누르면 바로 해당 문서를 지금처럼 우측화면에서 볼 수 있으면서도 해당 Line을 볼 수 있도록 구현함. 이는 자료를 html화 하고 해당 line을 Underline으로 쳐서 이를 PDF Viewer를 통해 그렇게 표출될 수 있도록 구현
- 추가로 OCP 공식 자료 문서는 Version이 다양하게 존재하는데 이를 tag를 활용하여 각 버전별로 관리하고 이 또한 RAG를 통해 버전을 물어보거나 혹은 선택할 수 있도록 함
- 현재는 후처리 기반의 구현 방식인데 이는 현재 문서에만 초점을 두고 구현한 방식이니, 특정 현장에서 범용성 있게 다 사용할 수 있도록 구조를 하드 코딩 기반에서 아예 의도도 LLM이 판단할 수 있도록 더 다양한 Agent를 활용할 수 있도록 한다. 하지만 무조건 .env에 있는 LLM 모델만 불러서 프롬프트를 주어 Agent화 해야함
- 현재 SSE 방식으로 채팅이 Streaming형태로 나오는 화면 방식은 좋으나, 채팅창에서 "답변을 생성중입니다." 하단에 열고 닫을 수 있는 바를 만들어서 해당 바에 실제 자료의 위치 추적중.. 과 같은 상태가 실제 연동되어 표시될 수 있도록 한다.
- 멀티턴 기능 또한 하드코딩을 하지말고 10턴 이상까지 강화할 수 있도록 한다. 예를 들어 질문1. OCP 클라우드 플랫폼 단계별로 어떻게 배포까지 진행해야되는지 설명해줘 답변1. 네 총 10단계로 설명해드릴 수 있습니다. 모든 단계를 한번에 설명해드릴까요 순차적으로 설명해드릴까요? 와 같이 진짜 LLM 이지만 자료 기반 RAG 기반 LLM 의 느낌처럼 아예 구현을 변경한다. 하지만 그렇다고 하여, 중국어와 한국어를 혼재한 질문같은 것은 처리하지 않는다. 이 또한 하드코딩을 하지 않으며, 현재 자료 기준 PVC라 하더라도 사용자가 한국어로 피브이시라고 할 수 있기에 이런 것도 처리가 가능하도록 해야함.

## 1. 프로젝트 개요 및 목표

이 프로젝트는 PDF 기반 기술 문서를 읽고, 사용자의 질문에 대해 문서 근거를 붙여 답하는 RAG 시스템

**핵심 목표**

1. 문서를 업로드하면 자동으로 읽고 검색 가능한 조각으로 변환
2. 질문을 그대로 검색하지 않고 의도와 대화 문맥을 반영해 검색 질의로 변환
3. 검색 결과를 그대로 쓰지 않고 답변에 필요한 근거만 남겨 응답 생성
4. 한 번의 질문으로 끝나지 않고 후속 질문에서도 이전 문맥 유지

프로젝트의 중심은 단순한 LLM 연동이 아니라,
`문서 분할 -> 질의 해석 -> 후보 검색 -> 후보 정제 -> 근거 기반 응답 -> 세션 상태 갱신`
흐름을 코드 수준에서 끝까지 구현

---

## 2. 실행 방법

### 2.1 실행: Docker Compose

이 프로젝트는 외부 저장소 서비스, 임베딩 서비스, 애플리케이션 서버를 함께 사용하므로 Docker Compose 실행이 가장 안전

1. `.env.example`을 복사해 `.env` 생성
2. 실행 환경에 맞는 필수 값 설정
   민감한 접속 정보와 모델 연결 정보는 README에 적지 않고 `.env`에서만 관리
3. 실행

```bash
docker-compose up --build -d
```

실행 후 확인 경로:

- 앱: `http://localhost:8000`
- 라이브러리 상태 확인: `http://localhost:8000/api/library`

`docker-compose.yml` 기준 구성:

- 저장소 서비스: 세션, 작업 상태, 벡터/청크 저장
- 임베딩 서비스: 문서/질문 벡터 생성
- 임베딩 초기화 서비스: 모델 준비
- `app`: FastAPI 서버

### 2.2 로컬 실행

로컬에서도 실행할 수 있지만, 애플리케이션이 기대하는 외부 서비스와 환경 변수가 먼저 준비되어 있어야 함

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2.3 문서 색인 방식

- UI에서 PDF 업로드 시 즉시 색인 진행
- 서버 시작 시 아직 색인되지 않은 문서가 있으면 백그라운드 색인 시도
- 전체 재색인은 `/api/reindex` 또는 UI를 통해 수행

---

## 3. 현재 아키텍처 설명

현재 구조는 "책임이 다른 것만 분리하고, 얇은 전달 계층은 제거" 기준으로 정리

### 3.1 현재 app 구조

```text
app/
  app_factory.py
  config.py
  dependencies.py
  main.py
  api/
    routes.py
    schemas.py
  llm/
    agents.py
  rag/
    answer.py
    chat_service.py
    chunking.py
    context.py
    indexing.py
    ingestion.py
    llm.py
    memory.py
    pipeline.py
    policy.py
    prompting.py
    query.py
    retrieval.py
    retrieval_service.py
    retrieval_state_builder.py
    turn_flow.py
  session/
    repository.py
    state.py
    store_sql.py
  storage/
    cache_repository.py
    task_repository.py
    vector_store.py
```

### 3.2 각 계층의 역할

- `app_factory.py`
  앱 생성, startup/reindex 상태 초기화, 정적 파일 mount
- `dependencies.py`
  실제 런타임 객체 조립
- `api/routes.py`
  HTTP 요청 수신과 유스케이스 연결
- `rag/*`
  질문 해석, 검색, 문맥 판단, 답변 생성의 핵심 알고리즘
- `session/*`
  대화 기록, 요약, 주제 상태 유지
- `storage/*`
  캐시, 작업 상태, 벡터 저장 같은 영속화 책임

현재 구조는 `RagPipeline`이 전체 흐름의 오케스트레이터 역할을 유지하면서도,
`turn_flow.py`, `retrieval_state_builder.py`는 명시적 협력자 계약을 통해 동작하도록 정리

---

## 4. 프로젝트 설명

### 4.1 문서 처리 방식

문서를 잘못 자르면 뒤 단계가 모두 혼동. 그래서 이 프로젝트는 먼저 문서 분할을 중요하게 처리

핵심 파일:

- [ingestion.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/ingestion.py)
- [chunking.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/chunking.py)

처리 흐름:

1. PDF에서 텍스트 추출
2. 페이지 경계 때문에 끊긴 표, YAML, 코드 블록 복원
3. 문서 특성에 따라 두 가지 전략 중 하나 선택

- 단순 페이지 기반 분할
- 구조 기반 분할

즉 시스템은 먼저 문서를 다음 형태로 변환

`PDF -> 복구된 텍스트 -> 의미 단위 chunk -> 검색 가능한 레코드`

### 4.2 질문을 그대로 검색하지 않는 이유

사용자는 이전에 했던 질문에 기반하여 아래와 같은 질문을 할 수 있음.

- "그거 다시 설명해줘"
- "2단계만 알려줘"
- "Service 예시 yaml 보여줘"

이런 질문은 단독으로 검색하기엔 어려움이 존재. 그래서 먼저 질문을 해석하고, 필요한 경우 이전 대화에서 빠진 주어를 복원하게 처리

핵심 파일:

- [query.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/query.py)
- [context.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/context.py)
- [policy.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/policy.py)

이 단계에서 하는 일:

1. 인사, 문서 질문, 후속 질문, 일반 대화 구분
2. 현재 세션에서 어떤 주제를 이야기했는지 확인
3. 리소스 이름, 액션, 원하는 응답 형태 추출

질문은 검색어로 바로 쓰이지 않고 아래 형태로 변환

`사용자 문장 -> 의도 + 대상 + 응답 형태 + 문맥 보강된 검색 질의`

### 4.3 검색은 점수 하나로 끝나지 않는다

문서 검색은 "가장 비슷한 것 하나"로 끝내면 오답이 많음. 이 프로젝트는 검색 후보를 여러 관점으로 점수화한 뒤 마지막에 다시 정렬.

핵심 파일:

- [retrieval.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/retrieval.py)
- [retrieval_state_builder.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/retrieval_state_builder.py)
- [retrieval_service.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/retrieval_service.py)

검색 알고리즘 흐름:

1. 질문을 벡터로 변환
2. 각 chunk에 대해 dense score와 sparse score 계산
3. 두 점수를 합쳐 1차 후보 구성
4. 후보들끼리 다시 재정렬
5. 점수가 높은 페이지와 문서에 grounding 부여
6. 실제 답변에 넣을 chunk를 별도 규칙으로 재선택

여기서 `retrieval_state_builder.py`는 단순 검색 호출기가 아니라,
질문 해석부터 질의 확장, 검색, 후처리, grounding, 최종 컨텍스트 선택까지 묶어 retrieval state를 생성.

### 4.4 왜 검색 결과를 다시 걸러내는가

검색 상위 결과가 항상 좋은 답변은 아니었음.
예를 들어 `Service yaml 예시`를 물었는데 `Route 예시`, `Ingress 설명`이 같이 들어올 수 있음.

그래서 이 프로젝트는 검색 후 한 번 더 처리 진행.

핵심 로직:

- 관련 없는 형식 제거
- 페이지 grounding 기준 정렬
- 같은 문서와 같은 주제 안에서 필요한 블록 우선
- 코드 요청이면 코드 블록 우선
- 주제가 흐트러지면 precision filter 적용

검색은 "후보를 넓게 모으는 단계", 품질은 "그 후보를 얼마나 잘 버리는가"에서 결정.

### 4.5 답변은 생성과 근거 정리를 동시에 진행

핵심 파일:

- [answer.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/answer.py)
- [turn_flow.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/turn_flow.py)

답변 생성 흐름:

1. 선택된 context chunk를 prompt에 주입
2. LLM이 스트리밍 응답 생성
3. 답변에서 인용 가능한 근거 추출
4. preview 페이지와 citation payload 정리
5. 코드 예시 요청이면 추출형 응답 우선

시스템은 LLM 응답을 그대로 내보내지 않고 아래 형태로 마감

`LLM 응답 -> 정리 -> 인용/페이지 정보 부착 -> 최종 payload`

### 4.6 멀티턴은 단순 채팅 기록 저장이 X

핵심 파일:

- [memory.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/memory.py)
- [repository.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/session/repository.py)
- [state.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/session/state.py)

이 프로젝트에서 세션은 단순 기록 저장소가 아니라 다음 질문을 더 잘 해석하기 위한 상태 저장소

저장하는 것:

- 최근 대화 몇 턴
- 현재 이야기 중인 주제
- 직전에 명시된 리소스
- 마지막 예시 anchor
- 인용된 페이지
- 절차형 답변의 현재 단계

그래서 `"그거 다시 보여줘"` 같은 질문도 문장만 보고 해석하지 않고 직전 topic state를 같이 읽고 답함.

### 4.7 캐시는 어디 사용

핵심 파일:

- [cache.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/rag/cache.py)
- [cache_repository.py](/C:/Users/KJungyu/OneDrive/Desktop/Company/과제/RAG_Task/app/storage/cache_repository.py)

캐시는 두 군데에 사용

1. 임베딩 캐시
2. 답변 캐시

캐시는 속도 개선만이 아니라 이미 계산한 데이터를 다시 계산하지 않기 위한 중복 제거 장치.

---

## 5. 코드 수준에서 보는 전체 흐름

질문 한 번이 들어오면 대략 아래 순서로 흐름.

```text
사용자 입력
  -> api/routes.py
  -> chat_service.py
  -> turn_flow.py
  -> policy.py / context.py / query.py
  -> retrieval_state_builder.py
  -> retrieval.py / retrieval_service.py
  -> prompting.py
  -> llm.py
  -> answer.py
  -> memory.py / session/*
  -> SSE 응답 반환
```

좀 더 구체적으로 적으면:

1. `routes.py`가 요청 수신
2. `ChatService`가 세션과 메시지를 전달
3. `ChatTurnOrchestrator`가 이번 턴 전체 흐름 관리
4. 필요하면 `RetrievalStateBuilder`가 검색 상태 생성
5. `RagPipeline`은 협력자들을 조립하는 중심축 역할
6. `AnswerGenerator`가 최종 응답 payload 마감

**현재 구조의 핵심**

- `RagPipeline`은 orchestration 중심축
- `turn_flow.py`, `retrieval_state_builder.py`는 pipeline 내부 구현을 직접 긁지 않고 명시적 협력자 계약으로 동작

---

## 6. 일정 및 진행사항


### 2026-03-17: 초기 골격 구성

- 초기 프로젝트 설계안 작성
- FastAPI 백엔드 골격 생성
- 애플리케이션 패키지 구조 생성
- 초기 페이지 및 라우트 연결
- 필수 라이브러리 설치와 기본 설정 정의


### 2026-03-18: 문서 처리와 첫 RAG 파이프라인 구현

- PDF 업로드, 삭제 API 추가
- 설정 구조, 청킹, 캐시, 인덱스 저장 구조 추가
- PDF 추출과 마크다운 변환 로직 추가
- 외부 LLM 스트리밍 연결
- 세션 히스토리 저장
- 하이브리드 검색과 파이프라인 오케스트레이션 구현


### 2026-03-19 ~ 2026-03-20: 책임 분리와 검색 품질 보강

- 전역 파이프라인 대신 컨테이너화
- 작업 추적 추가
- RAG 책임 분리
- 폴백, policy, 메모리 확장 강화
- 짧은 질의용 LLM 호출 추가
- 하드코딩 키워드 매칭 제거
- 리랭킹 및 키워드 매칭 강화


### 2026-03-23: 배포 환경 단순화와 컨테이너화

- Docker 컨테이너화 설계 문서 작성
- Docker 컨테이너화 구현 플랜 정리
- OCR 제거
- Dockerfile 및 앱 빌드 설정 추가



### 2026-03-25: 저장소 전환, 멀티턴 강화, 임베딩 모델 확장

- 저장소 설정 반영
- 페이지 경계에서 문서가 끊기는 현상 보완
- topic 연결 기반 멀티턴 대응 강화
- source 다양성 및 후보 추출 조정
- 임베딩 모델 전환 로직 추가
- agent 기반 파이프라인 확장
- 문서 업로드 상태바와 자동 인덱싱 보강



### 2026-03-26 ~ 2026-03-27: 질의 판단, 캐시, 지연 요인 제거

- 파라미터를 config로 이동
- 캐시 적중률 로깅 추가
- 최근 대화 수 제한으로 DB 부하 감소
- 전체 인덱스 호출 대신 캐시 기반 조회
- 단순 반응 search 제거
- 관련 없는 dense 매칭 채택 방지
- 자료 보기, 프롬프트, 후속 질문 정책 보강



### 2026-03-30: BGE-M3, RRF, reranker 도입

- `BGEOllamaEmbedder` 추가
- `HybridRetriever`에 RRF 추가
- reranker 추가
- BGE, RRF, reranker를 `RagPipeline`에 연결
- 기존 Hash, E5 경로 제거
- code chunk 병합과 topic store 세분화



### 2026-03-31: 사용자 분리, 의도 라우팅, 구조 정보 강화

- user별 대화 이력 격리
- 의도 라우팅 추가
- query keyword 정규화
- 코드 메타데이터 추가
- user별 session 조회, 삭제 보강
- 질문 의도, 자료, 포맷 추출 추가



### 2026-04-01: 후속질문 정밀화와 grounding 보강

- 절차 캐시와 retrieval 흐름 정교화
- topic 확장 및 intent, route, resource 세션 메모리 저장
- 코드 섹션 분할 추출 개선
- follow-up resource 오탐 방지
- ambiguity보다 follow-up 우선 처리
- example anchor와 clarification 보강



### 2026-04-02: app 구조 재배치와 협력 구조 정리

- app, session, storage, llm, rag 하위로 책임 재배치
- legacy service, repository 경로 제거
- `turn_flow`, `retrieval_state_builder` 분리
- `RagPipeline` 협력 구조 정리
- API 라우팅과 컨테이너 재배선



### 현재까지 완료된 범위

- PDF 적재, 추출, 청킹
- 하이브리드 검색과 재정렬
- grounding 기반 답변 생성
- 멀티턴 세션, 토픽 상태 관리
- 사용자별 세션 범위 분리
- 컨테이너 기반 실행
- app 구조 리팩터링


### 현재 진행 중인 개선

- 교육자료 문서 기반 테스트 데이터셋 생성
- 테스트 데이터를 통해 weight 조절
- 문서 데이터 ingestion 방식 고도화


### 앞으로 더 개선할 것

- tokenizer-aware chunk sizing
- 모델 입력 길이를 더 직접 반영한 chunk sizing 검토
- selection policy 추가 분리 여부 검토
- retrieval selection 규칙 확대 시 별도 domain service 분리 검토
- 문서별 extraction 품질 보강
- 페이지 경계에서 끊기는 코드, 표, YAML 복원 품질 보강
- 평가용 실험 로그 정리
- retrieval acceptance, chunking 전략, follow-up 처리 전후 비교 로그 정리

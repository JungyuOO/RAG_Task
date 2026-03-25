# RAG Task

오픈소스 RAG 프레임워크(LangChain, LlamaIndex 등) 없이 직접 구현한 커스텀 RAG 시스템
PDF 문서 기반 지식베이스를 구축하고, 멀티턴 대화를 지원하며, SSE 스트리밍으로 실시간 응답을 제공

## 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | Python + FastAPI |
| Frontend | HTML / Vanilla JS |
| Database | PostgreSQL |
| PDF 파싱 | PyMuPDF (fitz) |
| 임베딩 | HashingEmbedder (자체 구현) / multilingual-e5-small (선택) |
| LLM 통신 | httpx (사내 LLM 엔드포인트 — Qwen3.5-9B) |
| 스트리밍 | Server-Sent Events (SSE) |

## 프로젝트 구조

```
RAG_Task/
├── app/
│   ├── main.py                  # FastAPI 앱 팩토리 + 시작 시 자동 인덱싱
│   ├── config.py                # Settings (Pydantic BaseSettings)
│   ├── dependencies.py          # AppContainer 싱글톤
│   ├── api/
│   │   ├── routes.py            # API 엔드포인트
│   │   └── schemas.py           # 요청/응답 Pydantic 모델
│   ├── rag/
│   │   ├── pipeline.py          # RagPipeline (핵심 오케스트레이터)
│   │   ├── retrieval.py         # HybridRetriever (Dense + BM25 + Title)
│   │   ├── embeddings.py        # HashingEmbedder (SHA-256 해싱 기반)
│   │   ├── e5_embeddings.py     # E5Embedder (multilingual-e5-small)
│   │   ├── index.py             # VectorIndex (PostgreSQL 저장)
│   │   ├── memory.py            # SessionStore (세션 메모리)
│   │   ├── llm.py               # LlmClient (스트리밍 + 비스트리밍 호출)
│   │   ├── ingestion.py         # DocumentIngestor (PDF 추출)
│   │   ├── chunking.py          # TextChunker / StructuredMarkdownChunker
│   │   ├── cache.py             # JsonFileCache (TTL + LRU)
│   │   ├── types.py             # Document, Chunk, ChatTurn 데이터클래스
│   │   ├── utils.py             # 토크나이저, 한국어 조사 스트리핑
│   │   └── artifacts.py         # 마크다운 추출 경로
│   ├── services/
│   │   ├── indexing_service.py   # 인덱싱 (ingest → chunk → embed → store)
│   │   ├── retrieval_service.py  # 검색 결과 집계 및 컨텍스트 구성
│   │   ├── answer_service.py     # 인용 추출 및 정제
│   │   ├── agent_service.py      # QueryAgent / JudgeAgent (멀티 에이전트)
│   │   ├── turn_policy_service.py # 턴 분류 (인사/RAG/일반/명확화)
│   │   └── task_service.py       # 비동기 작업 추적
│   ├── repositories/
│   │   ├── index_repository.py
│   │   ├── cache_repository.py
│   │   ├── session_repository.py
│   │   └── task_repository.py
│   └── web/                      # 프론트엔드 (HTML + JS)
│       ├── index.html
│       └── js/
│           ├── app.js            # 메인 앱 로직
│           ├── chat.js           # 채팅 UI & SSE 처리
│           ├── library.js        # 자료실 관리 + 실시간 인덱싱 상태
│           ├── preview.js        # PDF 미리보기
│           ├── session.js        # 세션 이력
│           └── shared.js         # 공통 유틸리티
├── scripts/
│   └── build_index.py            # 전체 인덱스 재구축 CLI
├── data/
│   ├── corpus/pdfs/              # PDF 원본 저장소
│   ├── extracted_markdown/       # 추출된 마크다운
│   └── cache/
│       ├── embeddings/           # 임베딩 캐시
│       └── answers/              # 응답 캐시
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 실행 방법

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 아래 항목 설정
```

`.env` 주요 설정:

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `CLLM_BASE_URL` | LLM 엔드포인트 | `` |
| `CLLM_MODEL` | LLM 모델명 | `` |
| `EMBEDDING_MODEL` | 임베딩 모델 (`hash` 또는 `e5`) | `hash` |
| `DB_HOST` | PostgreSQL 호스트 | `` |
| `DB_PORT` | PostgreSQL 포트 | `` |
| `DB_NAME` | 데이터베이스명 | `` |
| `DB_USER` | DB 사용자 | `` |
| `DB_PASSWORD` | DB 비밀번호 | `` |

### 2. Docker로 실행

```bash
docker-compose up --build -d
```

### 3. 인덱스 구축

웹 UI의 자료실에서 직접 PDF를 업로드하면 자동으로 인덱싱됩니다.
`docker-compose down -v` 후 재시작해도 로컬 PDF 파일이 남아 있으면 **시작 시 자동 재인덱싱**이 수행됩니다.

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                    Web UI (Vanilla JS)                    │
└────────────────────────┬─────────────────────────────────┘
                         │ SSE
┌────────────────────────▼─────────────────────────────────┐
│                 FastAPI API Layer                         │
│  routes.py → AppContainer (singleton)                    │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                     RagPipeline                           │
│                                                           │
│  1. TurnPolicyService  → 턴 분류 (인사/RAG/off-topic)    │
│  2. QueryAgent         → 쿼리 최적화 + 대안 쿼리 생성    │
│  3. HybridRetriever    → 하이브리드 검색                  │
│  4. JudgeAgent         → 검색 결과 적합성 판단            │
│  5. LlmClient          → LLM 스트리밍 답변 생성          │
│  6. SessionStore       → 턴 저장 + 질의 재작성           │
└──┬──────────┬──────────┬──────────┬──────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
PostgreSQL  LLM API   File Cache  Extracted MD
(chunks,    (Qwen)    (embeddings, (PDF→마크다운)
 vectors,              answers)
 sessions,
 tasks)
```

## 핵심 구현 내용

### 1. 멀티 에이전트 파이프라인

단순 검색→응답이 아닌, LLM 기반 에이전트들이 파이프라인 각 단계를 담당:

| 에이전트 | 역할 |
|----------|------|
| **QueryAgent** | 사용자 질문을 분석하여 검색 최적화 쿼리 + 대안 쿼리 + 핵심 키워드 생성. 업로드 문서명을 참고하여 용어 확장 |
| **JudgeAgent** | 검색된 컨텍스트가 사용자 질문에 적합한지 판단. 부적합 시 재질문 메시지 생성 |
| **TurnPolicy** | 인사/확인은 검색 없이 응답, off-topic 질문은 거부, 문서 질문만 RAG 수행 |

### 2. 하이브리드 검색 (Hybrid Retrieval)

세 가지 신호를 가중 결합하여 검색 정확도를 높입니다:

| 신호 | Hash 가중치 | E5 가중치 | 설명 |
|------|-------------|-----------|------|
| Dense (코사인 유사도) | 0.45 | 0.30 | 임베딩 벡터 기반 의미 검색 |
| Sparse (BM25) | 0.25 | 0.35 | Okapi BM25 키워드 검색 (한국어 조사 스트리핑 적용) |
| Title Match | 0.15 | 0.20 | 문서 제목 매칭 보너스 |

1차 검색 후 2차 리랭킹을 수행합니다:
- 키워드 오버랩 점수 (0.17)
- 타이틀 보너스 (0.08 + 0.07)
- 컴팩트 서브스트링 보너스 (0.12)

### 3. 임베딩 모델 (선택 가능)

`.env`의 `EMBEDDING_MODEL` 설정으로 전환:

**HashingEmbedder** (`hash`, 기본값):
- 외부 모델 없이 SHA-256 해싱 기반 유니그램 + 바이그램 처리
- 위치 감쇠(positional decay) 적용, 768차원 L2 정규화
- 완전 오프라인, 결정론적 동작

**E5Embedder** (`e5`):
- `intfloat/multilingual-e5-small` (384차원)
- 한국어 의미 검색에 강점, sentence-transformers 기반
- 쿼리에 `"query: "`, 문서에 `"passage: "` 접두사 자동 적용

### 4. 콘텐츠 기반 청킹

페이지 경계를 무시하고 **문서 전체를 하나의 연속 텍스트로** 처리:

- `StructuredMarkdownChunker`: 헤딩/리스트/테이블/코드 블록 구조를 인식
- 헤딩 전환이나 블록 유형 변경에서만 청크 분리
- 페이지 번호는 메타데이터로만 추적 (인용 출처 표시용)
- `TextChunker`: 구조 정보가 부족한 일반 텍스트용 (700자, 120자 오버랩)
- `auto` 모드에서 문서 특성에 따라 자동 선택

### 5. 한국어 BM25 최적화

한국어 조사/어미가 붙은 토큰의 BM25 매칭 실패를 방지:
- `strip_korean_suffix()`: "네트워킹의" → "네트워킹", "스토리지에서" → "스토리지"
- 빈도 높은 조사 30여 개를 길이 순으로 매칭하여 제거
- 결과가 2자 미만이면 원본 유지 (과도한 스트리핑 방지)

### 6. 세션 메모리 (Session Memory)

LLM 히스토리 전달이 아닌 **애플리케이션 레벨 컨텍스트 관리**:
- PostgreSQL에 턴별 원본 대화, 구조화된 JSON 요약, 토픽 상태 저장
- `add_turn()` 호출 시마다 요약과 토픽 상태 갱신
- 모호한 후속 질문은 세션 컨텍스트 기반으로 LLM이 질의를 재작성

### 7. 시작 시 자동 인덱싱

`docker-compose down -v`로 DB가 초기화되어도 로컬 PDF 파일이 남아 있으면:
- 앱 시작 시 백그라운드에서 미인덱싱 문서를 자동 감지 및 인덱싱
- 서버는 즉시 응답 가능 (인덱싱은 비동기 수행)
- 프론트엔드에서 실시간 진행 상태 표시 (2초 폴링)
  - "대기 중" → "인덱싱 중..." (깜빡임 애니메이션) → "인덱싱 완료"

### 8. SSE 스트리밍

모든 응답(채팅, 인덱싱, 디버그)을 SSE로 스트리밍:

| 이벤트 타입 | 설명 |
|------------|------|
| `context` | 검색 메타데이터 (생성 전/후 전송) |
| `token` | 텍스트 청크 (`cached: true/false` 포함) |
| `done` | 스트림 종료 신호 |

### 9. Off-topic 거부

문서와 무관한 질문(일반 상식, 번역 등)은 거부:
- TurnPolicyService가 `general_chat`으로 분류 시 검색 없이 거부 응답
- 인사/확인은 간단한 환영 메시지로 응답
- 문서 관련 질문만 RAG 파이프라인 수행

### 10. 캐싱 전략

파일 기반 JSON 캐시 (TTL + LRU 퇴거):
- **임베딩 캐시** (`data/cache/embeddings/`): 동일 청크 재계산 방지
- **응답 캐시** (`data/cache/answers/`): 동일 질의+컨텍스트 조합 재사용
- 캐시 키: SHA-256 해시, TTL: 72시간, 최대 500개 엔트리

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 UI 페이지 |
| `POST` | `/api/chat` | 채팅 (SSE 스트리밍) |
| `POST` | `/api/chat/retry` | 마지막 실패 턴 재시도 |
| `POST` | `/api/chat/upload` | PDF 업로드 + 채팅 |
| `GET` | `/api/library` | 자료실 목록 조회 (인덱싱 상태 포함) |
| `GET` | `/api/library/preview` | PDF 미리보기 |
| `GET` | `/api/library/page-image` | PDF 페이지 이미지 |
| `GET` | `/api/library/download` | PDF 다운로드 |
| `DELETE` | `/api/library` | 자료 삭제 |
| `POST` | `/api/library/upload` | PDF 업로드 (인덱싱 진행률 스트리밍) |
| `POST` | `/api/reindex` | 전체 재인덱싱 |
| `GET` | `/api/tasks/{task_id}` | 비동기 작업 상태 조회 |
| `GET` | `/api/sessions` | 세션 목록 |
| `GET` | `/api/sessions/{session_id}` | 세션 이력 내보내기 |
| `DELETE` | `/api/sessions/{session_id}` | 세션 삭제 |
| `POST` | `/api/debug/retrieval` | 검색 결과 디버그 |

## 레이어 구조

```
routes.py (API 계층)
  └─ AppContainer (dependencies.py)
       ├─ RagPipeline (pipeline.py) ← 전체 오케스트레이션
       │    ├─ IndexingService      ← 인제스트 → 청킹 → 임베딩 → 저장
       │    ├─ RetrievalService     ← 검색, 컨텍스트 구성
       │    ├─ AnswerService        ← 인용 추출, 정제
       │    ├─ QueryAgent           ← 쿼리 최적화, 대안 쿼리 생성
       │    ├─ JudgeAgent           ← 검색 결과 적합성 판단
       │    ├─ TurnPolicyService    ← 턴 분류, off-topic 거부
       │    └─ Repositories         ← 데이터 접근 계층
       └─ TaskService              ← 비동기 작업 관리
```

## 의존성

```
fastapi==0.135.1
httpx==0.28.1
numpy==2.3.4
pydantic-settings==2.13.1
pymupdf==1.27.2
python-multipart==0.0.22
psycopg2-binary==2.9.10
sentence-transformers==3.4.1
uvicorn==0.42.0
```

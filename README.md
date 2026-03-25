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
| LLM 통신 | httpx (OpenAI-compatible API) |
| 스트리밍 | Server-Sent Events (SSE) |

## 프로젝트 구조

```
RAG_Task/
├── app/
│   ├── main.py                  # FastAPI 앱 팩토리
│   ├── config.py                # Settings (Pydantic BaseSettings)
│   ├── dependencies.py          # AppContainer 싱글톤
│   ├── api/
│   │   ├── routes.py            # API 엔드포인트 (13개)
│   │   └── schemas.py           # 요청/응답 Pydantic 모델
│   ├── rag/
│   │   ├── pipeline.py          # RagPipeline (핵심 오케스트레이터)
│   │   ├── retrieval.py         # HybridRetriever (Dense + BM25 + Title)
│   │   ├── embeddings.py        # HashingEmbedder (SHA-256 해싱 기반)
│   │   ├── index.py             # VectorIndex (PostgreSQL 저장)
│   │   ├── memory.py            # SessionStore (세션 메모리)
│   │   ├── llm.py               # LlmClient (스트리밍 호출)
│   │   ├── ingestion.py         # DocumentIngestor (PDF 추출)
│   │   ├── chunking.py          # TextChunker / StructuredMarkdownChunker
│   │   ├── cache.py             # JsonFileCache (TTL + LRU)
│   │   ├── types.py             # Document, Chunk, ChatTurn 데이터클래스
│   │   ├── utils.py             # 유틸리티 함수
│   │   └── artifacts.py         # 마크다운 추출 경로
│   ├── services/
│   │   ├── indexing_service.py   # 인덱싱 (ingest → chunk → embed → store)
│   │   ├── retrieval_service.py  # 검색 결과 집계 및 컨텍스트 구성
│   │   ├── answer_service.py     # 인용 추출 및 정제
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
│           ├── library.js        # 자료실 관리
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
| `DB_HOST` | PostgreSQL 호스트 | - |
| `DB_PORT` | PostgreSQL 포트 | - |
| `DB_NAME` | 데이터베이스명 | - |
| `DB_USER` | DB 사용자 | - |
| `DB_PASSWORD` | DB 비밀번호 | - |

### 2. Docker로 실행

```bash
docker-compose up --build -d
```


### 3. 인덱스 구축

웹 UI의 자료실에서 직접 PDF를 업로드할 수 있습니다.

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
│  1. TurnPolicyService  → 턴 분류 (인사/RAG/일반)          │
│  2. SessionStore       → 세션 컨텍스트 + 질의 재작성       │
│  3. HybridRetriever    → 하이브리드 검색                  │
│  4. LlmClient          → LLM 스트리밍 호출               │
│  5. SessionStore       → 턴 저장                         │
└──┬──────────┬──────────┬──────────┬──────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
PostgreSQL  LLM API   File Cache  Extracted MD
(chunks,    (Qwen)    (embeddings, (PDF→텍스트)
 vectors,              answers)
 sessions,
 tasks)
```

## 핵심 구현 내용

### 1. 하이브리드 검색 (Hybrid Retrieval)

세 가지 신호를 가중 결합하여 검색 정확도를 높입니다:

| 신호 | 가중치 | 설명 |
|------|--------|------|
| Dense (코사인 유사도) | 0.45 | HashingEmbedder 벡터 기반 의미 검색 |
| Sparse (BM25) | 0.25 | Okapi BM25 (k1=1.2, b=0.75) 키워드 검색 |
| Title Match | 0.15 | 문서 제목 매칭 보너스 |

1차 검색 후 2차 리랭킹을 수행합니다:
- 키워드 오버랩 점수 (0.17)
- 타이틀 보너스 (0.08 + 0.07)
- 컴팩트 서브스트링 보너스 (0.12)

### 2. 커스텀 임베딩 (HashingEmbedder)

외부 임베딩 서비스 없이 자체 구현한 결정론적 임베딩:
- SHA-256 해싱 기반 유니그램 + 바이그램 처리
- 위치 감쇠(positional decay) 적용
- 768차원 L2 정규화 벡터 생성
- 완전 오프라인, 결정론적 동작

### 3. 세션 메모리 (Session Memory)

LLM 히스토리 전달이 아닌 **애플리케이션 레벨 컨텍스트 관리**:
- PostgreSQL에 턴별 원본 대화, 구조화된 JSON 요약, 토픽 상태 저장
- `add_turn()` 호출 시마다 요약과 토픽 상태 갱신
- 모호한 후속 질문은 세션 컨텍스트 기반으로 LLM이 질의를 재작성

### 4. 턴 분류 (Turn Policy)

모든 사용자 메시지를 먼저 분류하여 불필요한 검색을 방지:
- **인사/확인** → 검색 없이 바로 응답
- **RAG** → 하이브리드 검색 + 컨텍스트 주입 후 응답
- **일반 질문** → 검색 점수 임계값 미달 시 일반 답변 fallback
- **명확화 요청** → 추가 질문 유도

### 5. SSE 스트리밍

모든 응답(채팅, 인덱싱, 디버그)을 SSE로 스트리밍:

| 이벤트 타입 | 설명 |
|------------|------|
| `context` | 검색 메타데이터 (생성 전/후 전송) |
| `token` | 텍스트 청크 (`cached: true/false` 포함) |
| `done` | 스트림 종료 신호 |

### 6. 문서 처리 파이프라인

```
PDF 업로드
  → PyMuPDF 텍스트 추출
  → Markdown 내보내기 (data/extracted_markdown/)
  → 청킹 (자동 선택: TextChunker / StructuredMarkdownChunker)
  → HashingEmbedder 벡터 생성 (캐시 확인)
  → PostgreSQL 저장 (chunks + vectors + metadata)
```

**청킹 전략:**
- `TextChunker`: 일반 텍스트 (700자, 120자 오버랩)
- `StructuredMarkdownChunker`: 헤딩 인식 구조화 청킹 (1000자, 150자 오버랩)
- `auto` 모드에서 문서 구조에 따라 자동 선택

### 7. 캐싱 전략

파일 기반 JSON 캐시 (TTL + LRU 퇴거):
- **임베딩 캐시** (`data/cache/embeddings/`): 동일 청크 재계산 방지
- **응답 캐시** (`data/cache/answers/`): 동일 질의+컨텍스트 조합 재사용
- 캐시 키: SHA-256 해시, TTL: 72시간, 최대 500개 엔트리

### 8. 벡터 인덱스

- PostgreSQL에 청크, 토큰, 메타데이터, 벡터(JSON) 저장
- pgvector 미사용 — 검색 시 전체 인덱스를 메모리에 로드하여 Python에서 스코어링
- 직접 설계한 벡터 저장/검색 로직

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 UI 페이지 |
| `POST` | `/api/chat` | 채팅 (SSE 스트리밍) |
| `POST` | `/api/chat/retry` | 마지막 실패 턴 재시도 |
| `POST` | `/api/chat/upload` | PDF 업로드 + 채팅 |
| `GET` | `/api/library` | 자료실 목록 조회 |
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
       │    ├─ TurnPolicyService    ← 턴 분류
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
uvicorn==0.42.0
```

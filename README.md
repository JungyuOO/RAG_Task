# RAG Task

오픈소스 RAG 프레임워크(LangChain, LlamaIndex 등) 없이 직접 구현한 커스텀 RAG 시스템
PDF 문서 기반 지식베이스를 구축하고, 멀티턴 대화를 지원하며, SSE 스트리밍으로 실시간 응답을 제공을 목표로 함

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
| 컨테이너 | Docker + Docker Compose |

## 프로젝트 구조

```
RAG_Task/
├── app/
│   ├── main.py                  # FastAPI 앱 팩토리 + 시작 시 자동 인덱싱
│   ├── config.py                # Settings (Pydantic BaseSettings, 파라미터 설정)
│   ├── dependencies.py          # AppContainer 싱글톤 DI
│   ├── api/
│   │   ├── routes.py            # API 엔드포인트 (12개)
│   │   └── schemas.py           # 요청/응답 Pydantic 모델
│   ├── rag/
│   │   ├── pipeline.py          # RagPipeline (핵심 오케스트레이터)
│   │   ├── retrieval.py         # HybridRetriever (Dense + BM25 + Title)
│   │   ├── embeddings.py        # HashingEmbedder (SHA-256 해싱 기반, 768차원)
│   │   ├── e5_embeddings.py     # E5Embedder (multilingual-e5-small, 384차원)
│   │   ├── index.py             # VectorIndex (PostgreSQL 저장, 인메모리 검색)
│   │   ├── memory.py            # SessionStore (세션 메모리 + 자동 요약)
│   │   ├── llm.py               # LlmClient (스트리밍 + 비스트리밍 호출)
│   │   ├── ingestion.py         # DocumentIngestor (PDF 추출 + 슬라이드 감지)
│   │   ├── chunking.py          # TextChunker / StructuredMarkdownChunker
│   │   ├── cache.py             # JsonFileCache (TTL + LRU)
│   │   ├── types.py             # Document, Chunk, ChatTurn 데이터클래스
│   │   ├── utils.py             # 토크나이저, 코사인 유사도, 한국어 조사 스트리핑
│   │   └── artifacts.py         # 마크다운 추출 경로
│   ├── services/
│   │   ├── indexing_service.py   # 인덱싱 (ingest → chunk → embed → store)
│   │   ├── retrieval_service.py  # 검색 결과 집계 및 컨텍스트 구성
│   │   ├── answer_service.py     # 인용 추출 및 정제
│   │   ├── agent_service.py      # QueryAgent / JudgeAgent (멀티 에이전트)
│   │   ├── turn_policy_service.py # 턴 분류 (인사/RAG/일반/명확화/off-topic)
│   │   └── task_service.py       # 비동기 작업 추적
│   ├── repositories/
│   │   ├── index_repository.py   # VectorIndex 레포 (인메모리 캐싱)
│   │   ├── cache_repository.py   # JsonFileCache 레포
│   │   ├── session_repository.py # SessionStore 레포
│   │   └── task_repository.py    # Task CRUD (PostgreSQL)
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
| `CLLM_BASE_URL` | LLM 엔드포인트 | (필수) |
| `CLLM_MODEL` | LLM 모델명 | (필수) |
| `EMBEDDING_MODEL` | 임베딩 모델 (`hash` 또는 `e5`) | `hash` |
| `DB_HOST` | PostgreSQL 호스트 | (필수) |
| `DB_PORT` | PostgreSQL 포트 | (필수) |
| `DB_NAME` | 데이터베이스명 | (필수) |
| `DB_USER` | DB 사용자 | (필수) |
| `DB_PASSWORD` | DB 비밀번호 | (필수) |

### 2. Docker로 실행

```bash
docker-compose up --build -d
```

`docker-compose.yml`이 PostgreSQL과 앱 컨테이너를 함께 구성
- `ports: "8000:8000"` — 호스트 PC의 8000번 포트를 컨테이너 내부 8000번에 매핑
- `http://192.168.68.156:8000`으로 접속 가능


### 3. 인덱스 구축

- 웹 UI의 자료실에서 PDF를 업로드하면 자동으로 인덱싱
- 서버 시작 시 `data/corpus/pdfs/`에 파일이 있으면 미인덱싱 문서를 자동 감지하여 백그라운드 인덱싱
- CLI로 전체 재구축: `python scripts/build_index.py`


## 시스템 아키텍처

### 전체 데이터 흐름

```
사용자 질문 (Web UI)
    │
    ▼ SSE
FastAPI (/api/chat)
    │
    ▼
RagPipeline.stream_chat()
    │
    ├─ 1. TurnPolicyService ─────── 턴 분류 (greeting/document_query/general_chat/off_topic)
    │                                 60+ 한/영 패턴 매칭
    │
    ├─ 2. SessionStore ──────────── 최근 대화 + 요약 조회 (메모리 윈도우: 6턴)
    │
    ├─ 3. [후속 질문이면]
    │      LLM 쿼리 리라이트 ───── 대명사 해소 ("그거" → "네트워크 정책")
    │
    ├─ 4. QueryAgent ────────────── 검색 쿼리 최적화 + 대안 쿼리 2개 + 키워드 추출
    │
    ├─ 5. Embedder ──────────────── 쿼리 벡터화 (Hash 768차원 / E5 384차원)
    │
    ├─ 6. HybridRetriever.search()
    │      ├─ Dense score (코사인 유사도)
    │      ├─ Sparse score (BM25)
    │      ├─ Title score (키워드 오버랩)
    │      ├─ 소스 다양성 필터링
    │      └─ 2차 리랭킹 → top_k=3 반환
    │
    ├─ 7. JudgeAgent ────────────── 검색 결과 관련성 평가 (relevant/confidence/clarification)
    │
    ├─ 8. RetrievalService ──────── 페이지/소스 단위 그라운딩
    │
    ├─ 9. LLM 스트리밍 응답 생성 ── 컨텍스트 + 대화 이력 주입
    │
    ├─ 10. AnswerService ─────────── 인용 파싱 + 응답 정제
    │
    └─ 11. SessionStore ─────────── 턴 저장 + 요약/토픽 상태 자동 갱신
```

### 인덱싱 흐름

```
PDF 업로드
    │
    ▼
DocumentIngestor.ingest_paths()
    ├─ 슬라이드 PDF 감지 (가로 비율 + 이미지/텍스트 블록 비율)
    ├─ 슬라이드형: 좌표 기반 텍스트 재조립
    ├─ 일반형: 구조화된 마크다운 변환 (테이블/헤딩/코드 블록 인식)
    ├─ 푸터 패턴 감지 및 제거
    └─ 교차 페이지 테이블 병합
    │
    ▼
청킹 전략 자동 선택 (auto)
    ├─ TextChunker (슬라이드형)
    └─ StructuredMarkdownChunker (일반 문서)
    │
    ▼
Embedder.encode()
    ├─ 임베딩 캐시 조회 (SHA-256 키)
    └─ 미스 시 인코딩 후 캐시 저장
    │
    ▼
VectorIndex (PostgreSQL)
    └─ 청크 + 토큰 + 메타데이터 + 벡터(JSON) 저장
```

### 레이어 구조

```
routes.py (API 계층)
  └─ AppContainer (dependencies.py)
       ├─ RagPipeline (pipeline.py) ← 전체 오케스트레이션
       │    ├─ IndexingService      ← 인제스트 → 청킹 → 임베딩 → 저장
       │    ├─ RetrievalService     ← 검색, 컨텍스트 구성, 그라운딩
       │    ├─ AnswerService        ← 인용 추출, 코드 예제 감지, 정제
       │    ├─ QueryAgent           ← LLM 기반 쿼리 최적화
       │    ├─ JudgeAgent           ← LLM 기반 검색 결과 적합성 판단
       │    ├─ TurnPolicyService    ← 턴 분류, off-topic 거부
       │    └─ Repositories         ← 데이터 접근 추상화 계층
       └─ TaskService              ← 비동기 작업 관리
```

## 핵심 구현 상세

### 1. 임베딩 모델 (2가지)

`.env`의 `EMBEDDING_MODEL` 설정으로 전환 가능하며, 모델에 따라 검색 가중치가 자동 조정

#### HashingEmbedder (`hash`, 기본값) — 768차원

외부 임베딩 모델 없이 SHA-256 해싱으로 고정 차원 벡터를 생성


**3가지 시그널이 벡터에 반영된다:**

| 시그널 | 방법 | 이유 |
|--------|------|------|
| 유니그램 | 각 토큰을 SHA-256으로 해싱 | 개별 단어가 있는지 여부를 벡터에 반영 |
| 바이그램 | 인접 토큰 쌍 `"네트워크_정책"` 해싱 (가중치 0.5) | 단어 순서를 반영하기 위함. "네트워크 정책"과 "정책 네트워크"가 다른 벡터가 됨 |
| 위치 감쇠 | `1.0 - 0.3 × (position/total)` | 기술 문서 특성상 앞부분에 핵심 키워드가 나오는 경우가 많아서 앞쪽 토큰에 가중치를 더 줌 |

**왜 부호(+1/-1)를 랜덤으로 부여하나?**
768차원짜리 벡터에 수많은 토큰이 매핑되다 보면 같은 인덱스에 여러 토큰이 겹치는 경우가 생긴다. 이때 부호를 랜덤으로 주면 관련 없는 토큰끼리는 +/-가 상쇄되어 노이즈가 줄어든다. random projection이라는 기법의 원리를 활용한 것이다.


벡터 길이를 1로 맞추면 검색 시 코사인 유사도를 내적만으로 계산 가능

**트레이드오프:**
- 장점: 외부 모델 불필요, 결정론적, 매우 빠름, 완전 오프라인
- 단점: 의미적 유사도를 잡지 못함 (예: "PV"라고 검색했을 때 "PersistentVolume"이라고 적힌 청크를 못 찾음) → BM25와 타이틀 매칭으로 보완

#### E5Embedder (`e5`) — 384차원

`intfloat/multilingual-e5-small` 사전학습 모델을 사용

```python
# 인덱싱할 때 (스토리지.pdf에서 추출한 청크)
"passage: PersistentVolume은 클러스터 레벨의 스토리지 리소스로, Pod와 독립적으로 존재한다"

# 사용자가 질문할 때
"query: PV가 뭐야?"
```

E5 모델은 질문과 문서를 서로 다른 임베딩 공간에 매핑하도록 학습되어 있어서 `query:`/`passage:` 접두사가 필수다.
모델은 서버 시작 시 바로 불러오지 않고, 실제로 첫 인코딩 요청이 들어올 때 로드한다 (메모리 절약).
의미적으로 유사한 표현도 잡을 수 있다 — 예를 들어 "PV"로 검색하면 "PersistentVolume"이라고 적힌 청크도 찾아낸다.

#### 두 모델 비교

| | HashingEmbedder | E5Embedder |
|--|----------------|------------|
| 차원 | 768 | 384 |
| 의미 이해 | 불가 | 가능 |
| 속도 | 매우 빠름 (해시 연산만) | 느림 (신경망 추론) |
| 외부 의존성 | 없음 | sentence-transformers |
| 오프라인 동작 | 가능 | 초기 모델 다운로드 필요 |

### 2. 코사인 유사도 구현

```python
def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))
```

보통 코사인 유사도는 `(a·b) / (||a|| × ||b||)` 이렇게 분모에 벡터 크기를 나눠야 하는데, 여기서는 내적(dot product)만 하고 있다. 이게 가능한 이유는 임베딩을 만들 때 이미 L2 정규화를 해서 벡터 길이가 항상 1이기 때문이다. `||a|| = ||b|| = 1`이면 분모가 1이 되니까 내적만으로 코사인 유사도가 된다. 검색할 때마다 나눗셈을 안 해도 되니까 연산이 줄어든다.


### 3. 하이브리드 검색 & 스코어링

#### 1차 스코어링 — 3중 시그널 결합

```
score = dense × W_d + sparse × W_s + title × W_t + title_match_bonus + compact_bonus
```

| 시그널 | Hash 가중치 | E5 가중치 | 역할 |
|--------|-------------|-----------|------|
| Dense (코사인 유사도) | **0.45** | 0.30 | 벡터 간 유사도. "스토리지 관련 내용" 전체를 넓게 잡아줌 |
| Sparse (BM25) | 0.25 | **0.35** | 키워드 정확 매칭. "PersistentVolume"이 정확히 들어있는 청크를 찾음 |
| Title match | 0.15 | 0.20 | 파일명과 쿼리의 키워드 겹침. "네트워킹에서 Service" → `네트워킹.pdf` 부스트 |
| Title match bonus | 최대 0.35 | 최대 0.35 | 파일명이 쿼리에 통째로 포함될 때 확정 가산 |
| Compact bonus | 최대 0.32 | 최대 0.32 | 공백 제거 후 연속 부분문자열 매칭 |

**왜 모델마다 가중치가 다른가:**
Hash 임베딩은 키워드가 겹치는 정도는 반영하지만 의미까지는 못 잡는다. 그래서 Dense(벡터 유사도)를 0.45로 높여서 전반적인 유사성을 최대한 뽑아냈다. 반면 E5는 의미 매칭 자체가 강력하니까 Dense를 0.30으로 낮추고, 대신 BM25(0.35)와 Title(0.20)을 올려서 "정확히 이 키워드가 있는 문서"를 더 잘 찾도록 했다.

**Title match와 Title match bonus는 뭐가 다른가:**
- Title match: 토큰 단위로 겹치는 비율을 계산한다. 예를 들어 "ArgoCD 배포 방법"이라고 질문하면 `CD(ArgoCD).pdf`와 토큰이 1/3 겹치니까 0.33점
- Title match bonus: 공백을 다 빼고 파일명이 쿼리에 통째로 들어있는지 본다. "네트워킹 문서에서 Service 알려줘"라고 질문하면 "네트워킹"이 `네트워킹.pdf`에 그대로 포함되니까 0.35점을 확정 부여

왜 둘 다 필요하냐면, 한국어는 조사 때문에 토큰 분리가 달라지는 경우가 있다. "스토리지에서" → 토큰화 → "스토리지"로 되긴 하지만, 파일명이 복합어일 때 토큰 매칭이 실패하는 케이스가 있어서 부분문자열 매칭으로 보완했다.

**Compact bonus는 왜 만들었나:**
실제로 테스트하다 보니 "yaml파일작성"처럼 띄어쓰기 없이 복합어로 질문하는 경우가 있었다. 토큰 단위로는 매칭이 안 되는데, 공백을 다 빼고 연속 부분문자열로 비교하면 `yaml_파일_작성_방법.pdf`의 청크와 매칭된다. 다만 35% 미만의 짧은 우연적 매칭(예: 조사 "는"이 겹치는 정도)은 무시하고, 상한을 0.32로 두어서 보조적인 역할만 하도록 했다.

#### 소스 다양성 보장

1차 스코어링 후 후보 풀(8개)을 구성할 때, 각 문서별로 가장 높은 점수의 청크를 먼저 넣는다. 예를 들어 `네트워킹.pdf`에서 1개, `스토리지.pdf`에서 1개 이런 식으로 다양한 문서가 후보에 들어오도록 했다. 단, 1위 점수의 30% 미만인 문서는 넣지 않는다 — 안 그러면 전혀 관련 없는 `CI_순서_v2.pdf` 같은 문서가 슬롯을 차지해서 정작 관련 있는 청크가 밀려나는 문제가 있었다.

#### 2차 리랭킹

후보 풀 8개에서 최종 3개(top_k)를 선별:

```
final = score × 0.68 + keyword_overlap × 0.17 + title × 0.08 + title_bonus × 0.07 + compact × 0.12
```

| 리랭킹 시그널 | 가중치 | 역할 |
|---------------|--------|------|
| base (1차 점수) | 0.68 | 1차에서 매긴 점수를 기본으로 깔고 |
| keyword overlap | 0.17 | 쿼리의 키워드가 청크에 실제로 얼마나 들어있는지 한 번 더 확인 |
| title | 0.08 | 파일명이 쿼리랑 관련 있으면 약간 가산 |
| title bonus | 0.07 | 파일명이 정확히 매칭되면 추가 가산 |
| compact | 0.12 | 연속 부분문자열이 겹치면 구문 일치도 반영 |

#### 검색 품질 지표

검색 결과가 얼마나 믿을 만한지 판단하기 위해 아래 지표를 함께 계산한다:
- **hit_rate**: 최소 점수(0.25) 이상인 결과가 몇 개인지. 3개 중 3개 다 넘으면 좋은 검색
- **score_gap**: 1위와 2위의 점수 차이. 차이가 크면 1위 결과가 확실히 관련 있다는 뜻
- **dense_sparse_correlation**: Dense 점수 순위와 BM25 점수 순위가 얼마나 일치하는지. 둘 다 같은 청크를 1위로 뽑았으면 그 결과를 더 신뢰할 수 있음

### 4. BM25 구현


| 파라미터 | 값 | 의미 |
|---------|-----|------|
| k1 | 1.2 | "PersistentVolume"이라는 단어가 청크에 3번 나오든 10번 나오든 점수 차이가 크지 않도록 포화시킴. Elasticsearch 기본값과 동일 |
| b | 0.75 | 긴 청크일수록 단어가 많이 나올 수밖에 없으니까, 문서 길이에 비례해서 점수를 깎아줌. 0.75면 적당히 깎는 수준 |

최종 점수는 0~1 범위로 정규화해서 Dense 점수랑 같은 스케일에서 가중합할 수 있게 했다.

### 5. 청킹 전략

#### 왜 청킹이 필요한가

LLM에 문서 전체를 넣을 수 없으므로 문서를 검색 가능한 단위로 나누어야 함. 
이 조각 하나하나가 "청크"이고, 잘 자르는 것이 검색 품질을 높임

#### TextChunker — 단순 슬라이딩 윈도우

```
[========700자========]
                [==120자 오버랩==][========700자========]
                                              [==120자==][========700자========]
```

- **chunk_size: 700자**, **overlap: 120자**
- 글자 수 기준으로 기계적으로 분할
- 페이지 단위로 처리 (각 페이지의 텍스트를 독립적으로 청킹)

**왜 오버랩이 필요한가:** 예를 들어 `스토리지.pdf`에서 "PersistentVolume은 클러스터 레벨의 스토리지 리소스로, Pod와 독립적으로 존재한다"라는 문장이 딱 청크 경계에서 잘리면, 앞 청크에는 "스토리지 리소스로,"까지만 들어가고 뒷 청크에는 "Pod와 독립적으로"부터 시작하게 된다. 120자 오버랩을 두면 양쪽 청크 모두 이 문장 전체를 포함해서, "PersistentVolume" 관련 검색 시 누락되지 않는다.

**언제 쓰나:** `실습 (1).pdf` 같은 슬라이드형 PDF. 페이지당 글자가 적고(420자 이하) 불릿 위주라 구조 파싱할 내용이 별로 없음

#### StructuredMarkdownChunker — 구조 인식 청킹

- **chunk_size: 1000자**, **overlap: 150자**
- 마크다운 블록 유형을 인식하여 의미 단위로 분할

| 블록 유형 | 감지 방법 |
|----------|----------|
| heading | `#`으로 시작하거나, 40자 이하 + 콜론 종료 (예: "정적 프로비저닝:") |
| list | `- ` 또는 `1. `으로 시작 |
| table | 파이프(`\|`)가 있고 구분선(`\|---\|---\|`) 포함 |
| code | ` ``` `로 시작/종료 |
| paragraph | 나머지 전부 |

**핵심 규칙들:**

1. **새 heading이면 무조건 새 청크 시작** — `네트워킹.pdf`에서 "## Service" 다음에 "## Ingress"가 나오면 거기서 청크를 자름
2. **단독 heading 청크 방지** — "## PersistentVolume"이라는 제목만 덩그러니 있는 청크가 만들어지면 검색 시 걸려도 LLM한테 줄 내용이 없음. 그래서 다음 본문 블록과 강제로 합침
3. **heading 이월** — 본문이 길어서 청크가 여러 개로 나뉠 때, 직전 섹션 제목("## Service")을 다음 청크에도 넣어줌. 이러면 어느 청크를 검색해도 "이게 어떤 섹션 내용인지" 알 수 있음
4. **짧은 도입 문단 + 표는 합침** — "다음은 Service 유형별 포트 목록이다:"(150자 이하) 바로 뒤에 표가 오면 분리하지 않음. 표만 따로 떼면 맥락이 사라지니까
5. **같은 블록 유형이면 페이지가 달라도 안 자름** — PDF의 페이지 경계는 인위적인 거라서, `스토리지.pdf` 3페이지의 리스트가 4페이지까지 이어지면 하나의 청크로 유지
6. **1400자 초과 블록은 슬라이딩 윈도우로 강제 분할**

**언제 쓰나:** `네트워킹.pdf`, `스토리지.pdf`, `yaml_파일_작성_방법.pdf` 같은 일반 텍스트 문서. heading/table/code 구조가 있으면 자동으로 이 청커가 선택됨

#### 자동 전략 선택 (auto)

```python
if avg_chars_per_page <= 420 and (short_line_ratio >= 0.55 or bullet_ratio >= 0.2):
    return "page_window"        # TextChunker
return "structured_markdown"    # StructuredMarkdownChunker
```

| 조건 | 전략 | 이유 |
|------|------|------|
| 페이지당 420자 이하 + 짧은 줄 55%+ | TextChunker | `실습 (1).pdf` 같은 슬라이드 — 구조 파싱할 게 없음 |
| 페이지당 420자 이하 + 불릿 20%+ | TextChunker | 불릿 위주 슬라이드 |
| 그 외 | StructuredMarkdownChunker | `네트워킹.pdf`, `스토리지.pdf` 같은 일반 문서 |

### 6. 인덱싱 & 벡터 저장

#### PostgreSQL을 순수 스토리지로 사용 (pgvector 미사용)

```
chunks 테이블:
  chunk_id | doc_id | source_path | text | tokens_json | page_number | metadata_json | vector_json
```

- 벡터는 **JSON 문자열**로 저장
- 검색 시 **전체 청크를 메모리에 로드** → Python에서 점수 계산
- `IndexRepository`가 메모리 캐시를 유지하며, 문서 변경(save/upsert/delete) 시에만 무효화

**왜 pgvector를 안 쓰나:**
pgvector를 쓰면 Dense 검색은 편하지만, BM25 + Title match + 각종 보너스를 조합하는 하이브리드 스코어링을 자유롭게 짜기가 어렵다. 현재 문서가 12개 수준이라 인메모리 검색으로 충분히 빠르고, 나중에 문서가 수만 건으로 늘어나면 그때 pgvector나 FAISS로 교체할 수 있는 구조로 만들어뒀다.

#### 임베딩 캐싱

- 청크 텍스트의 SHA-256 해시를 키로 파일 기반 캐시 (`data/cache/embeddings/`)
- 동일 텍스트 재인덱싱 시 임베딩 재계산을 생략
- TTL: 72시간, 최대 500개 엔트리, 가장 오래 안 쓰인 항목부터 교체(LRU)

### 7. 한국어 BM25 최적화

한국어 조사/어미가 붙은 토큰의 BM25 매칭 실패를 방지

```python
# 예시
"네트워킹의" → "네트워킹"
"쿠버네티스에서는" → "쿠버네티스"
"스토리지에서" → "스토리지"
```

- 빈도 높은 조사 30여 개를 **길이 순으로 매칭** ("에서"가 "에"보다 먼저 매칭)
- 결과가 2자 미만이면 원본 유지 (과도한 스트리핑 방지)
- 토큰화 시점에 적용되어 인덱싱과 검색 모두에서 일관되게 동작

### 8. 멀티 에이전트 파이프라인

단순 검색→응답이 아닌, LLM 기반 에이전트들이 파이프라인 각 단계를 담당

| 에이전트 | 역할 | 구현 방식 |
|----------|------|----------|
| **TurnPolicyService** | 턴 분류 (6가지) | 60+ 한/영 패턴 매칭 |
| **QueryAgent** | 검색 쿼리 최적화 | LLM 기반 |
| **JudgeAgent** | 검색 결과 적합성 판단 | LLM 기반 |

- **TurnPolicyService**: "안녕" 같은 인사는 검색 없이 바로 응답, "오늘 날씨 알려줘" 같은 off-topic은 거부, "Service 종류 알려줘" 같은 문서 질문만 RAG 수행
- **QueryAgent**: "스토리지에서 PV 설명해줘"를 받으면 "PersistentVolume 정의 스토리지"로 최적화하고, "PV PVC 바인딩", "스토리지 볼륨 종류" 같은 대안 쿼리도 2개 생성. LLM 호출이 실패하면 원본 쿼리를 그대로 쓴다
- **JudgeAgent**: 검색된 청크가 질문이랑 맞는지 LLM이 한 번 더 판단. 안 맞으면 "좀 더 구체적으로 질문해주세요" 같은 재질문 메시지를 생성

### 9. 세션 메모리 (Session Memory)

LLM 히스토리 전달이 아닌 **애플리케이션 레벨 컨텍스트 관리**:

- PostgreSQL에 턴별 원본 대화, 구조화된 JSON 요약, 토픽 상태를 저장
- `add_turn()` 호출 시마다 자동 갱신:
  - **요약**: topic, user_goal, recent_documents, unresolved_questions
  - **토픽 상태**: active_entities, selected_sources, last_retrieval_mode
- 후속 질문의 대명사/지시어를 해소하기 위한 **쿼리 리라이트**: 예를 들어 `네트워킹.pdf`에 대해 이야기하다가 "그거 yaml 예시 보여줘"라고 하면, 세션 컨텍스트를 보고 LLM이 "네트워크 정책 yaml 예시"로 재작성
- 메모리 윈도우: 최근 6턴

### 10. SSE 스트리밍

모든 응답(채팅, 인덱싱, 디버그)을 SSE로 스트리밍 
고정 메시지(인사, 거부, 재질문 유도 등)도 글자 단위로 스트리밍하여 일관된 UX를 제공

| 이벤트 타입 | 설명 |
|------------|------|
| `context` | 검색 메타데이터 (생성 전/후 전송) |
| `token` | 텍스트 청크 (`cached: true/false` 포함) |
| `done` | 스트림 종료 신호 |

### 11. 캐싱 전략

파일 기반 JSON 캐시 (TTL + 가장 오래 안 쓰인 항목부터 교체(LRU)):

| 캐시 | 경로 | 용도 |
|------|------|------|
| 임베딩 캐시 | `data/cache/embeddings/` | 동일 청크 재계산 방지 |
| 응답 캐시 | `data/cache/answers/` | 동일 질의+컨텍스트 조합 재사용 |

- 캐시 키: SHA-256 해시
- TTL: 72시간
- 최대 500개 엔트리
- 가장 오래 안 쓰인 항목부터 교체(LRU) 방식

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

## 주요 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `retrieval_top_k` | 3 | 최종 반환 청크 수 |
| `candidate_pool_size` | 8 | 리랭킹 전 1차 후보 수 |
| `retrieval_min_score` | 0.25 | 최소 점수 임계값 (미달 시 컨텍스트 제외) |
| `retrieval_retry_min_score` | 0.10 | 재질문 유도 임계값 (0.10~0.25: 재질문, 0.10 미만: 완전 실패) |
| `bm25_k1` | 1.2 | BM25 TF 포화 계수 |
| `bm25_b` | 0.75 | BM25 문서 길이 정규화 계수 |
| `chunk_size` | 700 | TextChunker 청크 크기 |
| `chunk_overlap` | 120 | TextChunker 오버랩 |
| `structured_chunk_size` | 1000 | StructuredMarkdownChunker 청크 크기 |
| `structured_chunk_overlap` | 150 | StructuredMarkdownChunker 오버랩 |
| `memory_window_turns` | 6 | 세션 메모리 윈도우 (최근 턴 수) |
| `cache_ttl_hours` | 72 | 캐시 TTL |
| `cache_max_entries` | 500 | 캐시 최대 엔트리 수 |

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
python-dotenv==1.1.1
```

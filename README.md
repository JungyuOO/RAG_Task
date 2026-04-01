# RAG Task

■ 과제 목적

* RAG → LLM으로 이어지는 챗봇 구조를 이해하고

* 향후 OCP 환경에 적용 가능한 AI 기반 지식 저장소 구축 역량 확보

 

■ 과제 내용

* 사용자 → RAG → LLM으로 이어지는 UI 및 전체 파이프라인 직접 구현

  (오픈소스 프레임워크 사용 금지, 모든 구성 직접 개발)

* RAG 시스템 구축 (기술 스택 자유) – 자료는 금주 교육자료 및 타 팀에서 제공하는 정리문서 등 자율적으로 수집

* 멀티턴 대화 (최소 5턴 이상) 지원

* LLM 연동 – 아래 제공되는 LLM만 사용할 것

  * Endpoint: 

  * Model: Qwen/Qwen3.5-9B

 

■ 추가 요구사항

* Streaming 응답 처리 구현 (토큰 단위 또는 chunk 단위 출력)

* Vector Index 구조 직접 설계 및 구현 (단순 라이브러리 호출 지양)

* 대화 이력 기반 Context 관리 (세션 단위 memory 구조 설계)

* RAG 성능 개선 전략 1가지 이상 적용 (예: reranking, chunking 전략 등)

* 캐싱 전략 적용 (예: 동일 질의 응답 캐싱, embedding 캐싱 등)

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


### 3. 하이브리드 검색 & 스코어링

#### 1차 스코어링 — 3중 시그널 결합

```
score = dense × W_d + sparse × W_s + title × W_t + title_match_bonus + compact_bonus
```

| 시그널 | Hash 가중치 | E5 가중치 | 역할 |
|--------|-------------|-----------|------|
| Dense (코사인 유사도) | **0.45** | 0.30 | 벡터 유사도 — 전체적인 내용 유사성 |
| Sparse (BM25) | 0.25 | **0.35** | 키워드 정확 매칭 — 특정 용어가 있는 청크 |
| Title match | 0.15 | 0.20 | 소스 파일명과 쿼리의 키워드 겹침 비율 |
| Title match bonus | 최대 0.35 | 최대 0.35 | 파일명이 쿼리에 통째로 포함될 때 확정 부스트 |
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

1차 스코어링 후 후보 풀(8개)을 구성할 때, 각 소스(문서)별로 최고 점수 청크를 먼저 선발 
단, 1위 점수의 30% 미만인 소스는 제외하여 무관한 문서가 슬롯을 차지하는 것을 방지

#### 2차 리랭킹

후보 풀 8개에서 최종 3개(top_k)를 선별:

```
final = score × 0.68 + keyword_overlap × 0.17 + title × 0.08 + title_bonus × 0.07 + compact × 0.12
```

| 리랭킹 시그널 | 가중치 | 역할 |
|---------------|--------|------|
| base (1차 점수) | 0.68 | 1차 검색 점수 유지 |
| keyword overlap | 0.17 | 쿼리-청크 간 키워드 겹침으로 정밀도 보강 |
| title | 0.08 | 파일명 매칭 시 출처 관련성 보정 |
| title bonus | 0.07 | 파일명 정확 매칭 보너스 |
| compact | 0.12 | 연속 부분문자열 매칭으로 구문 일치도 반영 |

#### 검색 품질 지표

검색 결과의 신뢰도를 측정하기 위해 다음 지표를 계산한다:
- **hit_rate**: min_score 이상인 결과 비율
- **score_gap**: 1위와 2위의 점수 차이 (높을수록 1위가 확실)
- **dense_sparse_correlation**: Dense와 Sparse 점수 순위의 스피어만 상관도 (둘 다 높은 순위에 합의하면 결과 신뢰도 높음)

### 4. BM25 구현


| 파라미터 | 값 | 의미 |
|---------|-----|------|
| k1 | 1.2 | TF 포화 속도 — 같은 단어가 반복될수록 점수 증가하지만 한계가 있음 (Elasticsearch/Lucene 기본값) |
| b | 0.75 | 문서 길이 정규화 — 1.0이면 완전 정규화, 0.0이면 정규화 없음. 짧은 청크에 약간의 TF 부스트를 주면서 긴 청크의 과대 매칭을 억제 |

최종 점수를 `max_possible`로 나눠 0~1 범위로 정규화하여 Dense 점수와 동일 스케일에서 가중합이 가능

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

**오버랩을 두는 이유:** 청크 경계에서 문장이 잘리면 앞 청크는 문장 전반부만, 뒷 청크는 후반부만 포함하게 됨
120자 오버랩이 있으면 양쪽 청크 모두 경계 부근의 전체 문장을 포함하여 검색 누락을 방지

**사용 시점:** 슬라이드형 PDF처럼 페이지당 글자가 적고(420자 이하), 짧은 줄(55%+)이나 불릿(20%+)이 많은 경우

#### StructuredMarkdownChunker — 구조 인식 청킹

- **chunk_size: 1000자**, **overlap: 150자**
- 마크다운 블록 유형을 인식하여 의미 단위로 분할

| 블록 유형 | 감지 방법 |
|----------|----------|
| heading | `#`으로 시작하거나, 40자 이하 + 콜론 종료 ("정적 프로비저닝:") |
| list | `- ` 또는 `1. `으로 시작 |
| table | 파이프(`\|`)가 있고 구분선(`\|---\|---\|`) 포함 |
| code | ` ``` `로 시작/종료 |
| paragraph | 나머지 전부 |

**핵심 규칙들:**

1. **새 heading이면 무조건 새 청크 시작** — 섹션 전환 시 분리
2. **단독 heading 청크 방지** — 제목만 있는 청크는 검색에 무의미하므로, 다음 블록과 합침
3. **heading 이월** — 청크가 분리될 때 직전 섹션 heading을 이월하여 모든 청크에 소속 섹션 제목이 포함되도록 함
4. **짧은 도입 문단 + 표는 합침** — 150자 이하 문단 뒤의 표는 분리하지 않음 (표의 맥락 유지)
5. **같은 블록 유형이면 페이지가 달라도 안 자름** — PDF 페이지 경계는 인위적이므로, 같은 유형의 흐름이면 유지
6. **1400자 초과 블록은 슬라이딩 윈도우로 강제 분할**

**사용 시점:** 일반 텍스트 문서 (구조화된 마크다운이 감지되는 경우).

#### 자동 전략 선택 (auto)

```python
if avg_chars_per_page <= 420 and (short_line_ratio >= 0.55 or bullet_ratio >= 0.2):
    return "page_window"        # TextChunker
return "structured_markdown"    # StructuredMarkdownChunker
```

| 조건 | 전략 | 이유 |
|------|------|------|
| 페이지당 420자 이하 + 짧은 줄 55%+ | TextChunker | 슬라이드형 PDF — 구조 파싱할 내용 부족 |
| 페이지당 420자 이하 + 불릿 20%+ | TextChunker | 불릿 위주 슬라이드 |
| 그 외 | StructuredMarkdownChunker | 일반 문서 — 구조 인식이 품질 향상 |

### 6. 인덱싱 & 벡터 저장

#### PostgreSQL을 순수 스토리지로 사용 (pgvector 미사용)

```
chunks 테이블:
  chunk_id | doc_id | source_path | text | tokens_json | page_number | metadata_json | vector_json
```

- 벡터는 **JSON 문자열**로 저장
- 검색 시 **전체 청크를 메모리에 로드** → Python에서 점수 계산
- `IndexRepository`가 메모리 캐시를 유지하며, 문서 변경(save/upsert/delete) 시에만 무효화

**pgvector를 사용하지 않는 이유:**
- 외부 확장 의존성 없이 순수 구현
- 문서 수가 수백~수천 건 수준에서는 인메모리 검색이 충분히 빠름
- 하이브리드 스코어링(Dense + BM25 + Title + 보너스)을 자유롭게 커스터마이즈 가능
- 대규모 시 pgvector나 FAISS로 교체 가능한 구조

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
| **TurnPolicyService** | 턴 분류 — 인사/확인/후속질문/문서질의/명확화/off-topic (6가지) | 60+ 한/영 패턴 매칭 |
| **QueryAgent** | 사용자 질문 → 검색 최적화 쿼리 + 대안 쿼리 2개 + 키워드 추출 | LLM 기반 |
| **JudgeAgent** | 검색된 컨텍스트가 질문에 적합한지 판단 (relevant/confidence/clarification) | LLM 기반 |

- **TurnPolicyService**: 인사/확인은 검색 없이 즉시 응답, off-topic은 거부, 문서 질문만 RAG 수행
- **QueryAgent**: 실패 시 원본 쿼리를 그대로 반환 (graceful degradation)
- **JudgeAgent**: 부적합 판정 시 재질문 메시지를 생성하여 사용자에게 반환

### 9. 세션 메모리 (Session Memory)

LLM 히스토리 전달이 아닌 **애플리케이션 레벨 컨텍스트 관리**:

- PostgreSQL에 턴별 원본 대화, 구조화된 JSON 요약, 토픽 상태를 저장
- `add_turn()` 호출 시마다 자동 갱신:
  - **요약**: topic, user_goal, recent_documents, unresolved_questions
  - **토픽 상태**: active_entities, selected_sources, last_retrieval_mode
- 후속 질문의 대명사/지시어를 해소하기 위한 **쿼리 리라이트**: 세션 컨텍스트를 포함하여 LLM이 "그거" → "네트워크 정책"으로 재작성
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

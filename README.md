# RAG Task — OCP 문서 + Live Cluster 통합 Copilot

OCP(Red Hat OpenShift Container Platform) 공식 문서와 고객사 매뉴얼을 기반으로 질문에 답하고, 실제 운영 중인 클러스터 상태까지 **한 번의 대화 안에서** 확인할 수 있는 RAG 기반 Copilot 입니다.

LangChain / LlamaIndex 같은 프레임워크 없이 **색인 · 검색 · 임베딩 · 멀티턴 세션 · OCP REST 연동 · 액션 실행 승인 워크플로우를 전부 직접 구현**했습니다.

> 발표 슬라이드는 [`README_PPT.md`](./README_PPT.md) 참고.

---

## 1. 시스템 전체 구조 

사용자는 **웹 브라우저**에서 채팅/대시보드를 사용하고, 뒤에서는 다음과 같이 동작합니다.

<p align="center">
  <img src="docs/images/System Architecture.png" alt="System Architecture Overview" width="860"/>
</p>

> 핵심 아이디어: **"문서(정적 지식)"** 와 **"클러스터 상태(동적 정보)"** 를 **한 답변 안에서 동시에** 제공합니다.

---

## 2. 질문이 들어왔을 때의 알고리즘 흐름

아래 세 가지 상황을 **비전공자 관점**에서 풀어서 보여드립니다.

### 2-A. 공식 문서 질문 흐름 (예: "pod 확인하는 명령어 뭐야?")

<p align="center">
  <img src="docs/images/official docs.png" alt="Official Docs pipeline" width="300"/>
</p>


| # | 단계 | 쉬운 설명 |
| :-: | --- | --- |
| 1 | 의도 파악 | "단순 인사인지 · 문서 질문인지 · 이전 대화 이어가기인지" 판단 |
| 2 | 라우팅 | "문서만 찾으면 되는지, 실제 클러스터도 봐야 하는지" 결정 |
| 3 | 질문 다듬기 | "그거 yaml로" 같은 생략 표현을 이전 대화에서 복원해 검색어로 변환 |
| 4 | 하이브리드 검색 | 의미(벡터) + 키워드(BM25) 두 가지를 **동시에** 돌려 후보를 넓게 수집 |
| 5 | 점수 융합 | RRF로 두 점수를 공정하게 합치고, 비슷한 내용은 MMR로 중복 제거 |
| 6 | 근거 선택 | 상위 후보 중 답변에 실제 쓸 조각만 최종 선별 |
| 7 | 인용 매핑 | 답변 문장 하나하나에 어떤 문서 조각이 근거인지 연결 |
| 8 | LLM 답변 | 선별된 근거만 LLM 에 투입 → 환각 최소화 + 실시간 스트리밍 |

### 2-B. OCP API 연동 흐름 (예: "지금 pandas 관련 pod 보여줘")

<p align="center">
  <img src="docs/images/ocp live.png" alt="OCP API Pipeline" width="860"/>
</p>


| # | 단계 | 쉬운 설명 |
| :-: | --- | --- |
| ⓪-1 | 클러스터 등록 | 사용자가 OCP 접속 정보(URL + 토큰)를 한 번만 등록 |
| ⓪-2 | 암호화 보관 | Broker가 토큰을 암호화해 Vault 스토어에 보관 (JSON 평문 보관 X) |
| ⓪-3 | 유효성 검증 | 실제로 접속 가능한지 즉시 테스트 (`/api/v1/namespaces`) |
| ⓪-4 | 리스 감시 | 토큰 만료 · 회수를 감시해 재발급 필요 시 알림 |
| ①-1 | 질문 계획 | "어떤 리소스를 어느 namespace 에서 조회할지" LLM 이 계획 |
| ①-2 | 실 REST 호출 | `oc get pods` 와 동일한 결과를 주는 REST 엔드포인트 호출 |
| ①-3 | 결과 정제 | 방대한 응답에서 사용자 질문(`pandas`)에 맞는 부분만 추림 |
| ①-4 | 자연어 답변 | 정제된 결과를 LLM이 사람 친화적 문장으로 재작성 |

### 2-C. Mixed 모드 (예: "공식 문서의 pod yaml이랑 실제 내 pod yaml은 뭐가 달라?")

<p align="center">
  <img src="docs/images/mixed.png" alt="official, ocp api mixed pipeline" width="860"/>
</p>


> 결과 payload에는 `answer_route = "mixed_doc_ocp"` 플래그가 포함되어 UI 에서 섹션을 자동 분리합니다.


---

## 3. 액션(변경) 실행 승인 워크플로우

단순 조회가 아닌 **apply / scale / delete** 같은 변경 작업은 바로 실행하지 않고 **승인 단계**를 거칩니다.

<p align="center">
  <img src="docs/images/ocp action workflow.png" alt="ocp action workflow" width="860"/>
</p>

---

## 4. 주요 기능 요약

- **문서 색인**: PDF/HTML/MD → Parser Selector → Block Normalizer → Metadata Enricher → Chunker → BGE-M3 → PostgreSQL(pgvector)
- **하이브리드 검색**: BGE-M3 Dense + BM25 Sparse (한국어 char n-gram 지원) + **RRF** 융합 + **MMR** 다양성 + Cross-Encoder rerank
- **의도·라우팅**: `IntentAgent` → `QueryRouter` → `QueryRewriteAgent` → `QuestionNormalizer` 의 다단 LLM Agent 파이프라인
- **인용 그라운딩**: `CitationGroundingValidator` 가 답변 문장 ↔ 근거 조각 매핑을 검증 (환각 방지)
- **OCP 연결 관리**: `OcpConnectionBroker` + Vault 암호화 스토어 + `LeaseScheduler` 로 토큰 수명 관리
- **OCP Live Chat**: `LiveQuestionPlanner` → `LiveToolExecutor` → `LiveAnswerComposer` 의 tool-driven 구조
- **Mixed 모드**: 한 질문에서 문서 RAG + 라이브 OCP를 자동 결합 (`answer_route = "mixed_doc_ocp"`)
- **액션 승인**: Preview (dry-run diff) → Request → Approve → Execute → Audit
- **Streaming**: FastAPI SSE 기반 토큰 스트리밍 + `progress` / `stage` 이벤트 실시간 전송
- **멀티턴 세션**: 브라우저 localStorage 기반 세션 persistence + 서버 측 `recent_turns` 컨텍스트

---

## 5. 기술 스택

| 영역 | 사용 기술 |
| --- | --- |
| **Frontend** | React 19 · Vite · TypeScript · Tailwind · shadcn/ui · Radix UI · lucide-react · react-markdown |
| **Backend** | FastAPI · uvicorn · Python 3.13 · Pydantic v2 |
| **LLM** | 사내 vLLM 엔드포인트 (`CLLM_BASE_URL`) — OpenAI 호환 API |
| **Embedding** | BGE-M3 (Ollama 또는 TEI 백엔드 선택) |
| **Store (문서)** | PostgreSQL + pgvector (`pgvector/pgvector:pg16`) |
| **Store (상태)** | SQLite (연결 프로파일 · 액션 감사 · 배치 작업) |
| **PDF 파싱** | PyMuPDF (`fitz`) |
| **HTML 파싱** | BeautifulSoup + markdownify |
| **Scraper** | Playwright (Red Hat 공식 문서 수집용 스크립트) |
| **Deploy** | Docker Compose 멀티 서비스 (api · web · postgres · ollama) |

---

## 7. 프로젝트 구조

```text
apps/
  api/                              # FastAPI 백엔드
    main.py                         # Uvicorn entry (apps.api.main:app)
    app_factory.py                  # FastAPI 앱 조립 + 라우터 등록
    runtime.py                      # 전역 싱글톤 wiring (DI 대체)
    core/                           # 설정·로깅·텍스트 유틸
      llm_settings.py
      pgvector_settings.py
      runtime_persistence.py
      logging.py
    api/
      routes/                       # HTTP 라우트 (/api/v1/*)
        chat.py                     #   /chat/query · /chat/stream
        ocp.py                      #   /ocp/* live 조회
        auth.py                     #   /auth/ocp-connections
        library.py                  #   /library/* 문서 관리
        indexing.py                 #   /index/* 색인 트리거
        docs_preview.py             #   /docs/preview/{id}
        actions.py                  #   /actions/* 승인 워크플로우
      schemas/                      # Pydantic 스키마
    rag/
      query/                        # 질의 전처리 Agent
        intent_agent.py             # 의도 분류 (rag / live / mixed / greeting)
        query_router.py             # lane 결정 + 검색어 생성
        query_rewrite_agent.py      # follow-up 복원
        question_normalizer.py      # 정규화
        synonym_expansion.py        # 약어·동의어 확장
        query_features.py           # top-k·source budget 계산
        chat_memory.py              # 최근 턴 요약
      retrieval/                    # 검색 레이어
        document_retriever.py       # Sparse (BM25 + char n-gram)
        pgvector_bridge.py          # Dense (pgvector)
        hybrid_fusion.py            # RRF + MMR 융합
        korean_tokenizer.py         # 한국어 토크나이저
        rerank_decider.py           # rerank 필요성 판단
        embedding_clients.py        # Ollama / TEI 클라이언트
      generation/                   # 답변 생성
        unified_copilot_service.py  # 🌟 전체 오케스트레이터
        llm_client.py               # OpenAI 호환 스트리밍 클라이언트
        citation_grounding.py       # 인용 검증
        answer_planner.py           # 답변 구조 계획
        response_cache.py           # 응답 캐시
      indexing/                     # 색인 파이프라인
        parsers/                    #   파일 형식별 파서
        normalize/                  #   block_normalizer
        enrich/                     #   metadata_enricher
        chunking/                   #   block_chunker
        index_service.py
        batch_index_service.py
        batch_job_service.py
        index_writer_bridge.py      # pgvector 쓰기
      library/
        document_service.py
        summary_service.py
    integrations/
      ocp/
        auth/                       # 인증 계층
          broker.py                 # OcpConnectionBroker
          verifier.py               # OcpConnectionVerifier
          lease_scheduler.py        # 토큰 리스 감시
          vault_store.py            # 시크릿 암호화 저장소
          alert_notifier.py
        live_service.py             # ConnectedOcpService (REST 호출)
        live_chat_service.py        # 🌟 Live chat 오케스트레이터
        live_question_planner.py    # 조회 계획 생성
        live_tool_executor.py       # REST 실행
        live_answer_composer.py     # 자연어 답변 합성
        manifest_sanitizer.py       # apply 전 YAML 정제
        action_preview_service.py   # dry-run diff
        action_request_service.py   # 승인 요청
        action_execution_service.py # 실제 실행
        action_audit_service.py     # 감사 로그
    persistence/                    # 영속화 계층
      sqlite.py                     # SQLite 기반 저장소 구현
      sqlite_runtime_repositories.py
      artifacts.py
      json_persistence.py
      protected_secret_persistence.py
      secrets.py
    tests/                          # unit + integration 테스트

  web/                              # React 프론트엔드
    src/
      App.tsx                       # SPA entry · route switch
      main.tsx
      pages/                        # 라우트별 페이지
        ChatPage.tsx
        DashboardPage.tsx
        ResourcesPage.tsx
        LibraryPage.tsx
        ActionsPage.tsx
        ConnectionPage.tsx
      features/                     # 기능별 모듈 (api · components · hooks · types)
        chat/
        connection/
        dashboard/
        resources/
        library/
        actions/
      components/
        layout/                     # AppShell · Sidebar · PageHeader
        ui/                         # shadcn/ui 프리미티브 + 커스텀
      lib/cn.ts                     # 클래스 병합 유틸
      styles/globals.css            # Tailwind base + 다크 토큰
    Dockerfile                      # nginx 기반 프로덕션 이미지
    nginx.conf                      # /api → FastAPI reverse proxy
    tsconfig.json
    vite.config.ts

packages/                           # 공유 타입 · UI (스캐폴딩)
  contracts/
  ui/

scripts/                            # 운영 · 평가 스크립트
  scrape_redhat_docs.py             # 공식 문서 크롤러
  build_section_index.py            # 섹션 인덱스 생성
  build_chat_eval_dataset.py        # 평가 데이터셋 빌드
  run_chat_eval_dataset.py          # 평가 실행
  build_service_eval_v2.py
  run_service_eval.py
  inventory_redhat_docs.py
  reindex_official_pdfs.py
  test_multiturn_v202.py

docs/
  architecture/                     # 아키텍처 결정 문서
  superpowers/plans/                # 개발 계획
  codex-tasks/                      # 외부 에이전트 작업 지시
  specs/                            # 설계 사양
  images/                           # 다이어그램 이미지

tests/                              # 시나리오 · 데이터 기반 테스트
```

---

## 8. 설치 및 실행

### 8.1 Docker Compose (권장)

```bash
# 1) 환경 변수 준비
cp .env.example .env
#   → CLLM_BASE_URL / CLLM_MODEL / DB_* / OCP_API_* · TEI_BASE_URL 또는 OLLAMA_BASE_URL 값 입력

# 2) 빌드 & 실행
docker compose up --build -d

# 3) 로그 확인
docker compose logs -f app
docker compose logs -f web
```

`docker-compose.yml` 서비스 구성:

| 서비스 | 역할 | 포트 |
| --- | --- | :-: |
| `postgres` | 세션 · 문서 청크 · 벡터 저장소 (`pgvector/pgvector:pg16`) | `5432` |
| `ollama` | BGE-M3 임베딩 서버 (선택적) | `11434` |
| `ollama-init` | 최초 실행 시 `bge-m3` 모델 pull | — |
| `app` | FastAPI 백엔드 (`apps.api.main:app`) | `8000` |
| `web` | React SPA (nginx) — `/api` → `app` 리버스 프록시 | `5173` |

접속 URL:

| URL | 설명 |
| --- | --- |
| http://localhost:5173 | 웹 UI (Connection → Chat → Dashboard …) |
| http://localhost:8000/docs | FastAPI Swagger |
| http://localhost:8000/healthz | 헬스체크 |

### 8.2 Local 개발 모드

```bash
# 백엔드
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

# 프런트엔드 (별도 터미널)
cd apps/web
npm install
npm run dev                     # http://localhost:5173
```

### 8.3 최초 색인

```bash
# PDF 투입
cp /path/to/*.pdf data/corpus/pdfs/

# API로 재색인 트리거 (또는 웹 UI 의 Library 탭에서 업로드)
curl -X POST http://localhost:8000/api/v1/index/reindex
```

`STARTUP_AUTO_INDEX_ENABLED=true` 이면 서버 기동 시 미색인 문서를 자동 처리합니다.

---

## 9. 환경 변수 요약

`.env.example` 참고. 주요 항목:

| 변수 | 설명 |
| --- | --- |
| `CLLM_BASE_URL` · `CLLM_MODEL` | LLM 엔드포인트 · 모델명 (OpenAI 호환) |
| `DB_HOST` · `DB_PORT` · `DB_NAME` · `DB_USER` · `DB_PASSWORD` | PostgreSQL |
| `EMBEDDING_BACKEND` | `tei` 또는 `ollama` |
| `TEI_BASE_URL` · `TEI_EMBEDDING_MODEL` | TEI 사용 시 |
| `OLLAMA_BASE_URL` · `OLLAMA_EMBEDDING_MODEL` | Ollama 사용 시 |
| `OCP_API_BASE_URL` · `OCP_API_TOKEN` · `OCP_API_VERIFY_SSL` · `OCP_DEFAULT_NAMESPACE` | OCP REST 기본 연결 (UI 에서 추가 등록도 가능) |
| `RAG_SOURCE_DIR` · `RAG_CACHE_DIR` · `RAG_EXTRACT_DIR` | 색인 경로 |
| `RAG_STRUCTURED_CHUNK_SIZE` · `_OVERLAP` · `_MIN_CHARS` | 청킹 파라미터 |
| `STARTUP_AUTO_INDEX_ENABLED` | 기동 시 자동 색인 여부 |
| `EMBEDDING_BATCH_SIZE` · `EMBEDDING_PARALLEL_WORKERS` | 임베딩 배치 성능 조절 |

---

## 10. 평가 · 운영 스크립트

| 스크립트 | 용도 |
| --- | --- |
| `scripts/scrape_redhat_docs.py` | Playwright 로 Red Hat 공식 문서 수집 |
| `scripts/inventory_redhat_docs.py` | 수집 문서 카탈로그 생성 |
| `scripts/build_section_index.py` | 섹션 단위 보조 인덱스 구축 |
| `scripts/build_chat_eval_dataset.py` · `run_chat_eval_dataset.py` | 챗봇 평가 데이터셋 빌드/실행 |
| `scripts/build_service_eval_v2.py` · `run_service_eval.py` | 서비스 단위 회귀 평가 |
| `scripts/reindex_official_pdfs.py` | 공식 PDF 일괄 재색인 |
| `scripts/test_multiturn_v202.py` | 멀티턴 시나리오 스모크 테스트 |

---

## 11. 로드맵

- 테스트 데이터 기반 **검색 가중치 자동 튜닝**
- **Tokenizer-aware chunk sizing** (모델 입력 길이 반영)
- **버전 간 차이 비교 UI**
- 페이지 경계(표 · 코드 · YAML) **복원 품질 강화**
- **Selection policy 분리** (도메인 서비스화)
- **Action 실행 SSE 스트리밍** (실행 중 로그 tail)
- retrieval acceptance · chunking 전략 · follow-up 전후 **실험 로그 정리**

---

## 12. 참고

- 🏛️ 아키텍처 문서: [`docs/architecture/`](./docs/architecture/)
- 📝 개발 계획: [`docs/superpowers/plans/`](./docs/superpowers/plans/)

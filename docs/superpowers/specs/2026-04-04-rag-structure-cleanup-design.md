# RAG 코드 구조 개편 설계

**Date:** 2026-04-04  
**Scope:** `app/rag/` 파일 수 축소 + 하드코딩 제거 (v3.0.1 spec 기반)  
**Goal:** 41개 → 31개 파일, 불필요 파일 삭제 및 레거시 하드코딩 제거

---

## 배경 및 원칙

v3.0.1 spec.md의 핵심 원칙:
- **하드코딩 없이 LLM Agent가 판단** — 마커 기반 하드코딩 로직 제거
- **500줄 미만 유지** — 500줄 제한을 위한 기계적 분할은 허용, 역할 없는 파일은 불허
- **Agent 기반 아키텍처** — IntentAgent → RetrievalAgent → AnswerAgent 흐름

---

## 변경 사항

### 1. 삭제할 파일 (10개)

| 파일 | 줄수 | 삭제 이유 | 처리 |
|------|------|-----------|------|
| `app/rag/ingestion.py` | 6 | 순수 re-export만 있는 파일 | 삭제 후 import 2곳 직접 수정 |
| `app/rag/resource_markers.py` | 44 | 하드코딩 OCP 마커 상수 — Agent가 처리 | 삭제 후 참조 3곳 제거 |
| `app/rag/policy.py` | 19 | `TurnPolicyDecision` dataclass 1개뿐 | `types.py`에 병합 |
| `app/rag/artifacts.py` | 30 | 경로 헬퍼 함수 2개뿐 | `utils.py`에 병합 |
| `app/rag/memory_summary.py` | 52 | SessionStore Mixin — memory.py에 흡수 가능 | `memory.py`에 병합 (→ ~345줄) |
| `app/rag/ingestion_loader.py` | 47 | DocumentIngestor — ingestion_pdf.py에 흡수 가능 | `ingestion_pdf.py`에 병합 (→ ~189줄) |
| `app/rag/pipeline_orchestrator.py` | 43 | RagPipeline의 얇은 wrapper 3개 메서드 | `pipeline.py`에 병합 |
| `app/rag/pipeline_context_support.py` | 245 | PipelineContextMixin — pipeline.py에 흡수 | `pipeline.py`에 병합 (합계 ~495줄) |
| `app/rag/reranker.py` | 38 | BGEReranker — retrieval 도메인 소속 | `retrieval.py`에 병합 (→ ~246줄) |
| `app/rag/chat_service.py` | 59 | API 레이어 소속 코드가 rag/에 위치 | dataclass → `api/schemas.py`, ChatService → `dependencies.py` |

### 2. 파일 이름 변경 (1개)

| 변경 전 | 변경 후 | 이유 |
|---------|---------|------|
| `pipeline_retrieval_support.py` | `pipeline_scoring.py` | 실제 역할(청크 점수 계산)을 이름에 반영 |

### 3. 코드 레벨 하드코딩 제거

#### 3-1. `context.py`
- `TurnContextResolver.RESOURCE_MARKERS` 클래스 변수 제거
- `_has_explicit_resource_reference()` 메서드 제거
- `resolve()` 내 해당 호출부 제거
- 이유: IntentAgent가 리소스를 LLM으로 분류하므로 규칙 기반 감지 불필요

#### 3-2. `pipeline_scoring.py` (구 pipeline_retrieval_support.py)
- `_compute_focus_multiplier()` 내 `RESOURCE_MARKERS.keys()` 참조 제거
- 대체: `query_interpretation`의 `resources` 필드(Agent 결과)를 활용하거나 단순화
- 이유: 하드코딩 리소스 목록 없이도 intent keywords로 동일 효과

#### 3-3. `retrieval_agent.py`
- `ACTION_MARKERS`, `FORMAT_MARKERS`, `REFERENTIAL_MARKERS`, `MULTITURN_MARKERS`, `CODE_MARKERS`, `PROCEDURE_MARKERS`, `TABLE_MARKERS` 상수 제거
- 관련 `_extract_resources()`, `_extract_actions()`, `_extract_formats()` 등 메서드 제거
- LLM 프롬프트를 확장하여 이 분류들을 LLM이 직접 반환하도록 수정
- 이유: spec "하드코딩 없이 LLM 판단" 원칙에 위배

---

## 병합 후 파일 크기 검증

| 수신 파일 | 기존 줄수 | 추가 내용 | 병합 후 예상 |
|-----------|----------|----------|-------------|
| `types.py` | 39 | TurnPolicyDecision (+19) | ~55줄 ✅ |
| `utils.py` | 152 | artifacts 함수 (+30) | ~180줄 ✅ |
| `memory.py` | 293 | SummaryMixin (+52) | ~345줄 ✅ |
| `ingestion_pdf.py` | 142 | DocumentIngestor (+47) | ~189줄 ✅ |
| `retrieval.py` | 208 | BGEReranker (+38) | ~246줄 ✅ |
| `pipeline.py` | 207 | orchestrator(+43) + context(+245) | ~495줄 ✅ |
| `api/schemas.py` | 기존 | ChatTurnRequest, RetryChatRequestModel | 증가폭 소 ✅ |
| `dependencies.py` | 기존 | ChatService | 증가폭 소 ✅ |

모든 병합 파일이 500줄 미만 유지 확인.

---

## 최종 app/rag/ 파일 구성 (31개)

```
app/rag/
├── __init__.py
│
├── # 공통
├── types.py               ← + TurnPolicyDecision
├── utils.py               ← + artifacts 경로 헬퍼
│
├── # 청킹
├── chunking.py            (공통 유틸/타입)
├── chunking_text.py       (TextChunker)
├── chunking_markdown.py   (StructuredMarkdownChunker)
├── chunking_markdown_support.py
│
├── # 문서 수집/인덱싱
├── ingestion_pdf_extract.py
├── ingestion_pdf_merge.py
├── ingestion_pdf.py       ← + DocumentIngestor
├── indexing.py
├── index.py
├── cache.py
│
├── # 임베딩
├── bge_embeddings.py
│
├── # 검색
├── retrieval.py           ← + BGEReranker
├── retrieval_service.py
├── retrieval_state_builder.py
│
├── # 메모리/세션
├── memory_schema.py
├── memory_topics.py
├── memory.py              ← + SummaryMixin
│
├── # 컨텍스트/프롬프팅
├── context.py             (RESOURCE_MARKERS 참조 제거)
├── prompting.py
├── llm.py
│
├── # 파이프라인
├── pipeline.py            ← + orchestrator 메서드 + context Mixin (~495줄)
├── pipeline_scoring.py    ← pipeline_retrieval_support.py 이름 변경
├── pipeline_runtime_support.py
├── pipeline_streaming.py
├── pipeline_streaming_support.py
│
└── # 버전 관리
    └── version_manager.py
```

---

## 영향받는 import 경로 (전체)

### 삭제 파일 → 새 위치

| 구 경로 | 새 경로 |
|---------|---------|
| `app.rag.policy.TurnPolicyDecision` | `app.rag.types.TurnPolicyDecision` |
| `app.rag.artifacts.extracted_markdown_path` | `app.rag.utils.extracted_markdown_path` |
| `app.rag.artifacts.extracted_markdown_candidates` | `app.rag.utils.extracted_markdown_candidates` |
| `app.rag.memory_summary.SessionStoreSummaryMixin` | (memory.py 내부로 통합) |
| `app.rag.ingestion.DocumentIngestor` | `app.rag.ingestion_pdf.DocumentIngestor` |
| `app.rag.ingestion_loader.DocumentIngestor` | `app.rag.ingestion_pdf.DocumentIngestor` |
| `app.rag.reranker.BGEReranker` | `app.rag.retrieval.BGEReranker` |
| `app.rag.chat_service.ChatTurnRequest` | `app.api.schemas.ChatTurnRequest` |
| `app.rag.chat_service.RetryChatRequestModel` | `app.api.schemas.RetryChatRequestModel` |
| `app.rag.chat_service.ChatService` | `app.dependencies.ChatService` |
| `app.rag.resource_markers.*` | 삭제 (참조 코드 제거) |
| `app.rag.pipeline_retrieval_support.*` | `app.rag.pipeline_scoring.*` |

---

## 테스트 전략

- 각 Task 완료 후 `python -m pytest tests/ -q` 실행
- 모든 79개 기존 테스트가 계속 통과해야 함
- import 오류가 가장 흔한 실패 원인 — 각 파일 삭제 후 즉시 검증

---

## 제약 조건

- 500줄 미만 규칙 유지
- UTF-8 인코딩 유지
- 기능 변경 없음 — 구조 정리만 수행
- `retrieval_agent.py` 하드코딩 제거 시 LLM 프롬프트 확장 필요 (기능 동일 유지)

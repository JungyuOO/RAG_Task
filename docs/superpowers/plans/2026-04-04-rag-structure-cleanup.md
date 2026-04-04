# RAG 구조 개편 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** app/rag/ 파일 수를 41개 → 31개로 축소하고, 하드코딩 마커를 제거하여 v3.0.1 spec의 Agent 기반 원칙에 부합하도록 구조를 정리한다.

**Architecture:** 기능 변경 없는 순수 구조 정리. micro-파일 병합, Mixin 인라인화, 하드코딩 상수 제거 후 LLM 프롬프트로 이관. 각 Task 완료 후 `python -m pytest tests/ -q`로 회귀를 즉시 검증한다.

**Tech Stack:** Python 3.11+, FastAPI, psycopg2, pytest

---

## 파일 변경 지도

| 삭제 파일 | 코드 이동 대상 |
|-----------|--------------|
| `app/rag/policy.py` | `app/rag/types.py` |
| `app/rag/artifacts.py` | `app/rag/utils.py` |
| `app/rag/memory_summary.py` | `app/rag/memory.py` |
| `app/rag/ingestion_loader.py` | `app/rag/ingestion_pdf.py` |
| `app/rag/ingestion.py` | (re-export 삭제, import 2곳 직접 수정) |
| `app/rag/reranker.py` | `app/rag/retrieval.py` |
| `app/rag/pipeline_orchestrator.py` | `app/rag/pipeline.py` |
| `app/rag/pipeline_context_support.py` | `app/rag/pipeline.py` |
| `app/rag/resource_markers.py` | (삭제 — 참조 3곳 제거) |
| `app/rag/chat_service.py` | `app/api/schemas.py` + `app/dependencies.py` |

| 이름 변경 | |
|----------|---|
| `app/rag/pipeline_retrieval_support.py` | → `app/rag/pipeline_scoring.py` |

---

## Task 1: policy.py → types.py 병합

**Files:**
- Modify: `app/rag/types.py`
- Delete: `app/rag/policy.py`
- Modify (import 수정): `app/rag/pipeline_context_support.py`, `app/rag/pipeline_streaming.py`, `app/rag/pipeline_streaming_support.py`, `app/rag/retrieval_state_builder.py`

- [ ] **Step 1: types.py 끝에 TurnPolicyDecision 추가**

`app/rag/types.py` 파일 끝에 다음을 추가한다:

```python
@dataclass(slots=True)
class TurnPolicyDecision:
    turn_type: str
    response_mode: str
    use_retrieval: bool
    use_memory_rewrite: bool
    allow_preview: bool
    allow_citations: bool
    needs_clarification: bool = False
    clarification_reason: str = ""
    clarification_prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 2: 4개 파일의 import 경로 수정**

각 파일에서 아래 줄을 찾아 교체한다:

```python
# 변경 전 (4개 파일 모두 동일)
from app.rag.policy import TurnPolicyDecision

# 변경 후
from app.rag.types import TurnPolicyDecision
```

대상 파일:
- `app/rag/pipeline_context_support.py` line 9
- `app/rag/pipeline_streaming.py` line 12
- `app/rag/pipeline_streaming_support.py` line 5
- `app/rag/retrieval_state_builder.py` line 10

- [ ] **Step 3: policy.py 삭제**

```bash
git rm app/rag/policy.py
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 5: 커밋**

```bash
git add app/rag/types.py app/rag/pipeline_context_support.py app/rag/pipeline_streaming.py app/rag/pipeline_streaming_support.py app/rag/retrieval_state_builder.py
git commit -m "refactor: TurnPolicyDecision을 types.py로 통합, policy.py 삭제"
```

---

## Task 2: artifacts.py → utils.py 병합

**Files:**
- Modify: `app/rag/utils.py`
- Delete: `app/rag/artifacts.py`
- Modify (import 수정): `app/api/routes_shared.py`, `app/rag/indexing.py`, `app/rag/ingestion_pdf_merge.py`

- [ ] **Step 1: utils.py 끝에 artifacts 함수 3개 추가**

`app/rag/utils.py` 파일 끝에 다음을 추가한다. `from pathlib import Path`가 utils.py에 없으면 상단 import에 추가한다:

```python
from pathlib import Path  # 이미 있으면 추가 불필요


def extracted_markdown_file_name(source_path: Path) -> str:
    return f"{source_path.stem}-{stable_hash(str(source_path))[:8]}.md"


def extracted_markdown_path(extract_dir: Path, source_path: Path) -> Path:
    return extract_dir / extracted_markdown_file_name(source_path)


def extracted_markdown_candidates(extract_dir: Path, source_path: Path) -> list[Path]:
    stem = source_path.stem
    glob_matches = list(extract_dir.glob(f"{stem}-????????.md"))
    if glob_matches:
        return glob_matches
    candidates: list[Path] = []
    for candidate_source in (source_path, source_path.resolve()):
        candidate_path = extracted_markdown_path(extract_dir, candidate_source)
        if candidate_path not in candidates:
            candidates.append(candidate_path)
    return candidates
```

- [ ] **Step 2: 3개 파일의 import 수정**

```python
# 변경 전
from app.rag.artifacts import extracted_markdown_candidates   # routes_shared.py
from app.rag.artifacts import extracted_markdown_path          # indexing.py, ingestion_pdf_merge.py

# 변경 후
from app.rag.utils import extracted_markdown_candidates
from app.rag.utils import extracted_markdown_path
```

- [ ] **Step 3: artifacts.py 삭제**

```bash
git rm app/rag/artifacts.py
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 5: 커밋**

```bash
git add app/rag/utils.py app/api/routes_shared.py app/rag/indexing.py app/rag/ingestion_pdf_merge.py
git commit -m "refactor: artifacts 함수를 utils.py로 통합, artifacts.py 삭제"
```

---

## Task 3: memory_summary.py → memory.py 병합

**Files:**
- Modify: `app/rag/memory.py`
- Delete: `app/rag/memory_summary.py`

- [ ] **Step 1: memory.py에서 memory_summary import를 제거하고 내용을 직접 인라인**

`app/rag/memory.py` 상단에서 다음 줄을 삭제한다:

```python
from app.rag.memory_summary import SessionStoreSummaryMixin
```

대신 `memory_summary.py`의 내용(import 포함)을 memory.py 상단 import 블록 뒤, 클래스 정의 전에 추가한다:

```python
# memory_summary.py에서 이관
import psycopg2.extras

from app.session.state import build_rewrite_context_payload, build_summary_bundle, extract_entities
from app.session.store_sql import persist_summary


class SessionStoreSummaryMixin:
    def build_rewrite_context(self, session_id: str, user_message: str) -> dict | None:  # noqa: ARG002
        recent = self.recent_turns(session_id)
        summary = self.structured_summary(session_id)
        topic_state = self.topic_state(session_id)
        return build_rewrite_context_payload(recent, summary, topic_state)

    def _recent_turns_for_refresh(self, session_id: str, limit: int = 10) -> list[ChatTurn]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT role, content, metadata FROM (
                        SELECT role, content, metadata, turn_id
                        FROM session_turns
                        WHERE session_id = %s
                        ORDER BY turn_id DESC
                        LIMIT %s
                    ) sub ORDER BY turn_id ASC
                    """,
                    (session_id, limit),
                )
                rows = cursor.fetchall()
        return [
            ChatTurn(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def _refresh_summary(self, session_id: str) -> None:
        turns = self._recent_turns_for_refresh(session_id, limit=10)
        summary_json, topic_state, summary = build_summary_bundle(turns)
        with self._connection() as connection:
            with connection.cursor() as cursor:
                persist_summary(cursor, session_id, summary, summary_json, topic_state)

    def _extract_entities(self, text: str) -> list[str]:
        return extract_entities(text)
```

주의: `json`, `ChatTurn`은 이미 memory.py에서 import되어 있으므로 중복 추가 금지.
`psycopg2.extras`와 `build_rewrite_context_payload`, `build_summary_bundle`, `extract_entities`, `persist_summary`는 memory.py에 없으면 추가한다.

- [ ] **Step 2: memory_summary.py 삭제**

```bash
git rm app/rag/memory_summary.py
```

- [ ] **Step 3: 줄수 확인 (500줄 미만 검증)**

```bash
wc -l app/rag/memory.py
```

Expected: < 500

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 5: 커밋**

```bash
git add app/rag/memory.py
git commit -m "refactor: SessionStoreSummaryMixin을 memory.py로 인라인, memory_summary.py 삭제"
```

---

## Task 4: ingestion_loader.py → ingestion_pdf.py 병합, ingestion.py 삭제

**Files:**
- Modify: `app/rag/ingestion_pdf.py`
- Delete: `app/rag/ingestion_loader.py`
- Delete: `app/rag/ingestion.py`
- Modify (import 수정): `app/rag/indexing.py`, `app/rag/pipeline.py`

- [ ] **Step 1: ingestion_pdf.py 끝에 DocumentIngestor 추가**

`app/rag/ingestion_pdf.py` 파일 끝에 다음을 추가한다:

```python
class DocumentIngestor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pdf_extractor = PdfExtractor(settings)

    def ingest_paths(self, paths: list[Path], progress_callback=None) -> tuple[list[Document], list[Path]]:
        documents: list[Document] = []
        skipped: list[Path] = []

        for path in paths:
            if not path.exists() or not path.is_file():
                skipped.append(path)
                continue

            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pdf_documents, markdown_sections = self.pdf_extractor.extract_pdf(path, progress_callback=progress_callback)
                documents.extend(pdf_documents)
                self.pdf_extractor.export_markdown(path, pdf_documents, markdown_sections)
            elif suffix in {".txt", ".md"}:
                text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
                if not text:
                    skipped.append(path)
                    continue
                documents.append(
                    Document(
                        doc_id=stable_hash(str(path)),
                        source_path=str(path),
                        page_number=None,
                        text=text,
                        metadata={"file_name": path.name, "loader": "text"},
                    )
                )
            else:
                skipped.append(path)

        return documents, skipped
```

주의: `normalize_text`, `stable_hash`, `Document`, `Settings`, `Path`는 이미 ingestion_pdf.py에서 import된 것을 확인 후 중복 추가 금지.

- [ ] **Step 2: 2개 파일의 import 수정**

```python
# app/rag/indexing.py — 변경 전
from app.rag.ingestion import DocumentIngestor

# 변경 후
from app.rag.ingestion_pdf import DocumentIngestor
```

```python
# app/rag/pipeline.py — 변경 전
from app.rag.ingestion import DocumentIngestor

# 변경 후
from app.rag.ingestion_pdf import DocumentIngestor
```

- [ ] **Step 3: 두 파일 삭제**

```bash
git rm app/rag/ingestion.py app/rag/ingestion_loader.py
```

- [ ] **Step 4: 줄수 확인**

```bash
wc -l app/rag/ingestion_pdf.py
```

Expected: < 500

- [ ] **Step 5: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 6: 커밋**

```bash
git add app/rag/ingestion_pdf.py app/rag/indexing.py app/rag/pipeline.py
git commit -m "refactor: DocumentIngestor를 ingestion_pdf.py로 통합, ingestion*.py 정리"
```

---

## Task 5: reranker.py → retrieval.py 병합

**Files:**
- Modify: `app/rag/retrieval.py`
- Delete: `app/rag/reranker.py`
- Modify: `app/rag/pipeline.py`

- [ ] **Step 1: retrieval.py 끝에 BGEReranker 추가**

`app/rag/retrieval.py` 파일 끝에 다음을 추가한다. `from sentence_transformers import CrossEncoder` import를 파일 상단에 추가한다:

```python
# retrieval.py 상단 import에 추가:
from sentence_transformers import CrossEncoder


# 파일 끝에 추가:
class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder 리랭커."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 5) -> None:
        self.top_k = top_k
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[: self.top_k]
```

- [ ] **Step 2: pipeline.py import 수정**

```python
# 변경 전
from app.rag.reranker import BGEReranker

# 변경 후
from app.rag.retrieval import BGEReranker
```

- [ ] **Step 3: reranker.py 삭제**

```bash
git rm app/rag/reranker.py
```

- [ ] **Step 4: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 5: 커밋**

```bash
git add app/rag/retrieval.py app/rag/pipeline.py
git commit -m "refactor: BGEReranker를 retrieval.py로 통합, reranker.py 삭제"
```

---

## Task 6: pipeline_orchestrator.py + pipeline_context_support.py → pipeline.py 병합

**Files:**
- Modify: `app/rag/pipeline.py`
- Delete: `app/rag/pipeline_orchestrator.py`
- Delete: `app/rag/pipeline_context_support.py`

- [ ] **Step 1: pipeline.py에서 Mixin import 및 상속 제거**

`app/rag/pipeline.py`에서 다음 두 줄을 삭제한다:

```python
# 삭제 대상
from app.rag.pipeline_context_support import PipelineContextMixin
```

클래스 선언에서 `PipelineContextMixin` 제거:

```python
# 변경 전
class RagPipeline(PipelineContextMixin, PipelineRetrievalMixin, PipelineRuntimeMixin):

# 변경 후
class RagPipeline(PipelineRetrievalMixin, PipelineRuntimeMixin):
```

- [ ] **Step 2: pipeline_context_support.py의 import 블록을 pipeline.py 상단에 추가**

`pipeline_context_support.py`에서 사용하는 import 중 pipeline.py에 없는 것들을 추가한다:

```python
import asyncio  # 없으면 추가
import re       # 이미 있음
from app.rag.utils import normalize_text, tokenize  # tokenize 없으면 추가
```

- [ ] **Step 3: pipeline_context_support.py의 모든 메서드를 RagPipeline 클래스 본문에 추가**

`pipeline_context_support.py`에 정의된 메서드들 (`_resolve_turn_context`, `_policy_from_intent`, `_build_non_retrieval_state`, `_domain_guard_state`, `_should_skip_procedure_shortcut`, `_detect_procedure_followup`, `_looks_like_step_navigation_without_state`, `_resolve_requested_resource_kinds`, `_infer_item_resource_kinds`, `_expand_query_with_resource_aliases`, `_build_procedure_followup_answer`, `_prefer_block_type_items`, `_heading_overlap_score`)를 `RagPipeline` 클래스 내부에 그대로 붙여넣는다. `self`는 이미 정상 동작하므로 수정 불필요.

- [ ] **Step 4: pipeline_orchestrator.py의 메서드를 RagPipeline 클래스에 추가**

`app/rag/pipeline.py`의 `RagPipeline` 클래스 끝에 다음 메서드들을 추가한다:

```python
    async def classify_intent(self, user_message: str, context: dict) -> dict:
        return await self.intent_agent.classify(user_message, context)

    async def expand_query(self, user_message: str, intent_result: dict, available_sources: list) -> dict:
        return await self.retrieval_agent.expand(user_message, intent_result, available_sources)

    async def process_turn(self, session_id: str, user_message: str, allowed_sources: list | None = None) -> dict:
        context = await self._resolve_turn_context(session_id, user_message)
        intent = await self.classify_intent(user_message, context)
        if intent.get("intent") == "greeting":
            return {"intent": intent, "retrieval": None, "mode": "conversational"}
        if intent.get("intent") in {"rag", "clarification"}:
            expanded = await self.expand_query(user_message, intent, allowed_sources or [])
            return {"intent": intent, "retrieval": expanded, "mode": "grounded"}
        return {"intent": intent, "retrieval": None, "mode": intent.get("intent", "general")}

    async def check_procedure(self, user_message: str, context_items: list) -> dict:
        return await self.answer_agent.check_procedure(user_message, context_items)

    async def handle_step_navigation(self, intent: dict, session_id: str, context_items: list) -> dict:
        step_target = intent.get("step_target", "next")
        session_state = self.session_repository.topic_state(session_id)
        procedure = session_state.get("procedure_state", {})
        current = procedure.get("current_step", 0)
        total = procedure.get("total_steps", 0)
        if step_target == "next":
            target_step = min(current + 1, total) if total > 0 else current + 1
        elif step_target == "prev":
            target_step = max(current - 1, 1)
        else:
            target_step = int(step_target) if str(step_target).isdigit() else current
        return {"target_step": target_step, "total_steps": total, "context_items": context_items}
```

- [ ] **Step 5: pipeline_orchestrator.py, pipeline_context_support.py 삭제**

```bash
git rm app/rag/pipeline_orchestrator.py app/rag/pipeline_context_support.py
```

- [ ] **Step 6: 테스트 파일의 import 수정**

`PipelineOrchestrator`를 import하는 테스트 파일들을 수정한다. `PipelineOrchestrator`는 이제 `RagPipeline`으로 대체된다:

```python
# tests/test_pipeline_orchestrator.py, test_procedure_flow.py, test_phase0_pipeline.py, test_integration_v301.py
# 변경 전
from app.rag.pipeline_orchestrator import PipelineOrchestrator

# 변경 후
from app.rag.pipeline import RagPipeline as PipelineOrchestrator
```

- [ ] **Step 7: 줄수 확인**

```bash
wc -l app/rag/pipeline.py
```

Expected: < 500

- [ ] **Step 8: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 9: 커밋**

```bash
git add app/rag/pipeline.py tests/test_pipeline_orchestrator.py tests/test_procedure_flow.py tests/test_phase0_pipeline.py tests/test_integration_v301.py
git commit -m "refactor: pipeline_orchestrator/context_support를 pipeline.py로 인라인 통합"
```

---

## Task 7: chat_service.py → api/schemas.py + dependencies.py 이동

**Files:**
- Modify: `app/api/schemas.py`
- Modify: `app/dependencies.py`
- Delete: `app/rag/chat_service.py`
- Modify: `app/api/routes_chat.py`

- [ ] **Step 1: api/schemas.py에 dataclass 추가**

`app/api/schemas.py` 상단 import에 추가:

```python
from dataclasses import dataclass
from typing import Any
```

파일 끝에 추가:

```python
@dataclass(slots=True)
class ChatTurnRequest:
    session_id: str
    message: str
    allowed_source_paths: set[str] | None = None
    append_user_turn: bool = True


@dataclass(slots=True)
class RetryChatRequestModel:
    session_id: str
    message: str
    owner_id: str | None = None
    allowed_source_paths: set[str] | None = None
    append_user_turn: bool = True
```

- [ ] **Step 2: dependencies.py에 ChatService 추가**

`app/dependencies.py`에서:

```python
# 삭제
from app.rag.chat_service import ChatService

# 추가 (기존 import 블록 내)
from typing import Any
```

`AppContainer` dataclass 위에 ChatService 클래스를 추가한다:

```python
class ChatService:
    """Public entrypoint for chat and retry flows."""

    def __init__(self, pipeline: Any, session_repository: Any) -> None:
        self.pipeline = pipeline
        self.session_repository = session_repository

    def stream(self, request: "ChatTurnRequest"):
        return self.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=request.message,
            allowed_source_paths=request.allowed_source_paths,
            append_user_turn=request.append_user_turn,
        )

    def retry(self, request: "RetryChatRequestModel"):
        requested_message = (request.message or "").strip()
        pending_message = self.session_repository.pending_user_message(
            request.session_id,
            owner_id=request.owner_id,
        )
        user_message = pending_message or requested_message
        if not user_message:
            raise LookupError("No pending user message found for retry.")
        append_user_turn = request.append_user_turn
        if pending_message and (not requested_message or pending_message == requested_message):
            append_user_turn = False
        return self.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=user_message,
            allowed_source_paths=request.allowed_source_paths,
            append_user_turn=append_user_turn,
        )
```

- [ ] **Step 3: routes_chat.py import 수정**

```python
# 변경 전
from app.rag.chat_service import ChatTurnRequest, RetryChatRequestModel

# 변경 후
from app.api.schemas import ChatTurnRequest, RetryChatRequestModel
```

- [ ] **Step 4: chat_service.py 삭제**

```bash
git rm app/rag/chat_service.py
```

- [ ] **Step 5: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 6: 커밋**

```bash
git add app/api/schemas.py app/dependencies.py app/api/routes_chat.py
git commit -m "refactor: ChatService와 request model을 API 레이어로 이동, chat_service.py 삭제"
```

---

## Task 8: pipeline_retrieval_support.py → pipeline_scoring.py 이름 변경

**Files:**
- Rename: `app/rag/pipeline_retrieval_support.py` → `app/rag/pipeline_scoring.py`
- Modify: `app/rag/pipeline.py`

- [ ] **Step 1: 파일 이름 변경**

```bash
git mv app/rag/pipeline_retrieval_support.py app/rag/pipeline_scoring.py
```

- [ ] **Step 2: pipeline.py import 수정**

```python
# 변경 전
from app.rag.pipeline_retrieval_support import PipelineRetrievalMixin

# 변경 후
from app.rag.pipeline_scoring import PipelineRetrievalMixin
```

- [ ] **Step 3: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 4: 커밋**

```bash
git add app/rag/pipeline_scoring.py app/rag/pipeline.py
git commit -m "refactor: pipeline_retrieval_support.py → pipeline_scoring.py 이름 변경"
```

---

## Task 9: resource_markers.py 삭제 및 참조 3곳 정리

**Files:**
- Delete: `app/rag/resource_markers.py`
- Modify: `app/rag/context.py` (RESOURCE_MARKERS 참조 제거)
- Modify: `app/rag/pipeline_scoring.py` (RESOURCE_MARKERS 참조 제거)
- Modify: `app/llm/retrieval_agent.py` (RESOURCE_MARKERS 참조 제거)

### 9-1. context.py 수정

- [ ] **Step 1: context.py에서 resource_markers import 제거**

```python
# 삭제 대상 줄
from app.rag.resource_markers import RESOURCE_MARKERS, marker_in_text
```

- [ ] **Step 2: TurnContextResolver 클래스에서 RESOURCE_MARKERS 클래스 변수 및 메서드 제거**

다음 줄을 삭제한다:

```python
RESOURCE_MARKERS = RESOURCE_MARKERS  # 클래스 변수 삭제
```

다음 메서드 전체를 삭제한다:

```python
def _has_explicit_resource_reference(self, normalized_message: str) -> bool:
    for markers in self.RESOURCE_MARKERS.values():
        if any(marker_in_text(normalized_message, marker) for marker in markers):
            return True
    return False
```

- [ ] **Step 3: resolve() 메서드에서 해당 호출 제거**

`resolve()` 메서드 내 아래 두 줄을 찾아 수정한다:

```python
# 변경 전
explicit_resource = self._has_explicit_resource_reference(normalized)
if looks_like_referent and looks_like_code_request and second and ambiguity_gap < 0.15 and not explicit_resource:

# 변경 후 (explicit_resource 줄 삭제, 조건에서 and not explicit_resource 제거)
if looks_like_referent and looks_like_code_request and second and ambiguity_gap < 0.15:
```

### 9-2. pipeline_scoring.py 수정

- [ ] **Step 4: pipeline_scoring.py에서 RESOURCE_MARKERS import 제거**

```python
# 삭제 대상
from app.rag.resource_markers import RESOURCE_MARKERS
```

- [ ] **Step 5: _compute_focus_multiplier() 내 RESOURCE_MARKERS.keys() 참조 제거**

```python
# 변경 전
known_resources = set(RESOURCE_MARKERS.keys())
target_count = lowered_text.count(target_resource)
sibling_count = 0
for resource in known_resources:
    if resource != target_resource and resource not in all_requested_resources:
        sibling_count += lowered_text.count(resource)

# 변경 후 (all_requested_resources에 있는 리소스만 비교)
target_count = lowered_text.count(target_resource)
sibling_count = sum(
    lowered_text.count(r)
    for r in all_requested_resources
    if r != target_resource
)
```

### 9-3. retrieval_agent.py 수정

- [ ] **Step 6: retrieval_agent.py에서 resource_markers import 제거**

```python
# 삭제 대상
from app.rag.resource_markers import RESOURCE_MARKERS, marker_in_text
```

- [ ] **Step 7: _extract_resources() 메서드를 topic_state 기반으로 단순화**

`_extract_resources()` 전체를 다음으로 교체한다. RESOURCE_MARKERS 없이 topic_state의 기존 데이터와 LLM 결과의 keywords를 활용한다:

```python
def _extract_resources(self, normalized_keywords: list[str], topic_state: dict) -> list[str]:
    """LLM 결과 keywords와 topic_state에서 리소스를 추출한다."""
    resources = list(normalized_keywords)
    if not resources and self._should_inherit_resources(topic_state):
        anchor = topic_state.get("last_example_anchor") or {}
        anchor_resource = str(anchor.get("resource_kind") or "").lower().strip()
        if anchor_resource:
            resources.append(anchor_resource)
        for resource in topic_state.get("last_explicit_resources", []) or []:
            normalized_resource = str(resource).lower().strip()
            if normalized_resource and normalized_resource not in resources:
                resources.append(normalized_resource)
        last_code_resource_kind = str(topic_state.get("last_code_resource_kind") or "").lower().strip()
        if last_code_resource_kind and last_code_resource_kind not in resources:
            resources.append(last_code_resource_kind)
    return resources
```

- [ ] **Step 8: _should_inherit_resources() 단순화 — REFERENTIAL_MARKERS/CODE_MARKERS/PROCEDURE_MARKERS 제거**

`_should_inherit_resources()` 메서드 시그니처와 내용을 교체한다:

```python
def _should_inherit_resources(self, topic_state: dict) -> bool:
    return bool(
        topic_state.get("last_explicit_resources")
        or topic_state.get("last_code_resource_kind")
        or (topic_state.get("last_example_anchor") or {}).get("resource_kind")
    )
```

- [ ] **Step 9: interpret() 메서드에서 _extract_resources() 호출 시그니처 업데이트**

`interpret()` 내부에서 호출하는 부분을 수정한다:

```python
# 변경 전
resources = self._extract_resources(normalized_message, normalized_keywords, topic_state)

# 변경 후
resources = self._extract_resources(normalized_keywords, topic_state)
```

- [ ] **Step 10: 하드코딩 상수 클래스 변수 삭제**

`RetrievalAgent` 클래스에서 다음 상수 블록 전체를 삭제한다:

```python
ACTION_MARKERS = {
    "create": ("생성", "만들", "작성", "create"),
    ...
}
FORMAT_MARKERS = {
    "yaml": ("yaml", "yml", "manifest", "매니페스트"),
    ...
}
REFERENTIAL_MARKERS = ("그거", "그건", ...)
MULTITURN_MARKERS = ("다음", "계속", ...)
CODE_MARKERS = ("yaml", "manifest", ...)
PROCEDURE_MARKERS = ("단계", "절차", ...)
TABLE_MARKERS = ("표", "table")
```

- [ ] **Step 11: _extract_actions(), _extract_formats() 메서드를 LLM 결과 기반으로 교체**

```python
def _extract_actions(self, query_result: dict) -> list[str]:
    return list(query_result.get("actions") or [])

def _extract_formats(self, query_result: dict) -> list[str]:
    return list(query_result.get("format_constraints") or [])
```

- [ ] **Step 12: interpret() 전체를 LLM 결과 기반으로 교체**

`interpret()` 메서드 전체를 다음으로 교체한다:

```python
def interpret(self, user_message: str, query_result: dict | None = None, topic_state: dict | None = None) -> dict:
    query_result = query_result or {}
    topic_state = topic_state or {}
    normalized_keywords = normalize_query_keywords(user_message, query_result.get("search_keywords", []))
    resources = self._extract_resources(normalized_keywords, topic_state)
    actions = self._extract_actions(query_result)
    format_constraints = self._extract_formats(query_result)
    response_shape = str(query_result.get("response_shape") or "text")
    needs_multiturn_state = any(m in normalize_text(user_message).lower() for m in ("다음", "계속", "step", "단계"))
    intent = self._determine_intent(response_shape, format_constraints, actions)
    is_document_query = bool(
        resources
        or actions
        or format_constraints
        or any(h in normalize_text(user_message).lower() for h in self.DOCUMENT_QUERY_HINTS)
        or (
            (topic_state.get("active_topic") or topic_state.get("selected_sources"))
            and len(user_message) <= 40
        )
    )
    return {
        "intent": intent,
        "is_document_query": is_document_query,
        "resources": resources,
        "actions": actions,
        "format_constraints": format_constraints,
        "response_shape": response_shape,
        "normalized_keywords": normalized_keywords,
        "needs_multiturn_state": needs_multiturn_state,
    }
```

- [ ] **Step 13: _determine_response_shape() 메서드 삭제**

더 이상 TABLE_MARKERS, CODE_MARKERS, PROCEDURE_MARKERS를 사용하지 않으므로 메서드를 삭제한다. `_determine_intent()`는 LLM이 반환한 `response_shape`을 입력받으므로 유지한다:

```python
# _determine_response_shape() 메서드 전체 삭제
```

- [ ] **Step 14: _is_document_query() 메서드 삭제**

`interpret()` 내부에 인라인되었으므로 별도 메서드 삭제:

```python
# _is_document_query() 메서드 전체 삭제
```

- [ ] **Step 15: RETRIEVAL_SYSTEM_PROMPT 업데이트 — resources/actions/format_constraints/response_shape 추가**

`app/llm/retrieval_agent.py` 상단의 `RETRIEVAL_SYSTEM_PROMPT`를 다음으로 교체한다:

```python
RETRIEVAL_SYSTEM_PROMPT = """당신은 RAG 시스템의 검색 최적화 에이전트입니다.
사용자의 질문을 분석하여 벡터 검색에 최적화된 쿼리를 생성합니다.

역할:
1. 사용자 질문을 영어/한국어 혼합 검색 쿼리로 확장
2. 대안 쿼리 2-3개 생성
3. 특정 버전이 언급되면 target_versions에 포함
4. 여러 주제가 혼합된 질문이면 multi_source: true
5. 언급된 OCP 리소스 유형 추출 (예: pod, pvc, deployment, service 등)
6. 사용자 의도 동작 추출 (예: create, explain, compare, delete)
7. 원하는 출력 형식 추출 (예: yaml, cli, table — 없으면 빈 배열)
8. 응답 형태 결정 (text | code | table | procedure | comparison)

사용 가능한 문서 목록:
{available_sources}

의도 분석 결과:
{intent_result}

응답은 항상 JSON 객체로 반환하세요:
{{
    "expanded_query": "검색에 최적화된 확장 쿼리",
    "alternatives": ["대안 쿼리 1", "대안 쿼리 2"],
    "target_versions": [],
    "multi_source": false,
    "resources": ["pod", "pvc"],
    "actions": ["create"],
    "format_constraints": ["yaml"],
    "response_shape": "code"
}}
"""
```

- [ ] **Step 16: resource_markers.py 삭제**

```bash
git rm app/rag/resource_markers.py
```

- [ ] **Step 17: 테스트 실행**

```bash
python -m pytest tests/ -q
```

Expected: 79 passed

- [ ] **Step 18: 커밋**

```bash
git add app/rag/context.py app/rag/pipeline_scoring.py app/llm/retrieval_agent.py
git commit -m "refactor: resource_markers 하드코딩 제거, retrieval_agent를 LLM 기반으로 단순화"
```

---

## Task 10: 최종 검증

**Files:** 없음 (검증만)

- [ ] **Step 1: 전체 테스트 실행**

```bash
python -m pytest tests/ -v
```

Expected: 79 passed, 0 failed

- [ ] **Step 2: 500줄 제한 검증**

```bash
find app/ -name "*.py" -exec wc -l {} + | sort -rn | head -15
```

Expected: 모든 파일 < 500줄

- [ ] **Step 3: 삭제된 파일 참조 잔존 여부 확인**

```bash
grep -rn "resource_markers\|pipeline_orchestrator\|pipeline_context_support\|ingestion_loader\|chat_service\|reranker\|memory_summary\|ingestion\.py\|artifacts\|policy\.py" app/ tests/ --include="*.py"
```

Expected: 0건 (없어야 함)

- [ ] **Step 4: app/rag/ 파일 수 확인**

```bash
ls app/rag/*.py | wc -l
```

Expected: 31개 이하

- [ ] **Step 5: 최종 커밋**

```bash
git add -A
git commit -m "refactor: RAG 구조 개편 완료 — 41개 → 31개 파일, 하드코딩 마커 제거"
```

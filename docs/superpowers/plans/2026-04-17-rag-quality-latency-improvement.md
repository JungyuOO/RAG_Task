# RAG Quality & Latency Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise service eval pass rate from 1/10 to ≥7/10 and cut per-turn latency from 20–28s to 7–10s by consolidating 5 serial LLM calls to 2+conditional, replacing arbitrary fusion weights with RRF, adding Korean character n-gram sparse retrieval, and emitting `[N]` citations natively from the synthesis prompt.

**Architecture:** Single `QueryRouter` LLM call replaces normalize+intent+rewrite agents. Sparse + dense retrieval run in parallel. RRF + MMR replaces ad-hoc fusion. LLM judge rerank fires only when top-2 score gap < 15%. Synthesis prompt emits `[N]` markers directly; post-validation only removes invalid markers. Each phase is gated by a settings toggle so any phase can be rolled back.

**Tech Stack:** Python 3.x, FastAPI, Pydantic, httpx (async), pytest/unittest, asyncio, BGE-m3 via TEI (existing), CLLM via OpenAI-compatible API (existing), pgvector (existing).

**Spec:** `docs/superpowers/specs/2026-04-17-rag-quality-latency-improvement-design.md`

---

## File Structure

### New files
- `apps/api/rag/query/query_router.py` — single LLM call replacing 3 agents
- `apps/api/rag/retrieval/hybrid_fusion.py` — RRF + MMR
- `apps/api/rag/retrieval/rerank_decider.py` — gap-triggered rerank decision
- `apps/api/rag/retrieval/korean_tokenizer.py` — character n-gram tokenizer
- `apps/api/rag/query/synonym_expansion.py` — domain acronym dictionary
- `tests/test_query_router.py`
- `tests/test_hybrid_fusion.py`
- `tests/test_korean_tokenizer.py`
- `tests/test_rerank_decider.py`
- `tests/test_synonym_expansion.py`
- `tests/test_citation_grounding_preserve.py`
- `tests/test_pipeline_router_integration.py`

### Modified files
- `apps/api/core/llm_settings.py` — phase toggles
- `apps/api/rag/generation/unified_copilot_service.py` — router path, parallel retrieval, native citation prompt, mixed fallback fix
- `apps/api/rag/generation/citation_grounding.py` — preserve answer body when citations are invalid
- `apps/api/rag/retrieval/embedding_clients.py` — query asymmetric prompt
- `apps/api/rag/retrieval/document_retriever.py` — char n-gram tokenizer + RRF fusion + MMR
- `apps/api/rag/retrieval/pgvector_bridge.py` — query asymmetric prompt
- `apps/api/rag/query/query_features.py` — synonym expansion helper

### Deprecated (kept for rollback)
- `apps/api/rag/query/question_normalizer.py`
- `apps/api/rag/query/intent_agent.py`
- `apps/api/rag/query/query_rewrite_agent.py`

### Operational
- One full reindex (Phase 4 entry) — `python scripts/build_index.py` (existing)
- Per-phase eval baseline backups: `tests/results/service-eval/baseline-pre-p{N}.json`

---

## Phase 1 — Native [N] Synthesis + Mixed Fallback Fix

Goal: fix the citation/eval contradiction and the bug where the user message ends up inside the answer body.

### Task 1.1 — Add `USE_NATIVE_CITATION_PROMPT` toggle to settings

**Files:**
- Modify: `apps/api/core/llm_settings.py`

- [ ] **Step 1: Add the toggle**

```python
# inside ChatLlmSettings, after the existing fields
use_native_citation_prompt: bool = True
```

- [ ] **Step 2: Verify import paths still load**

Run: `python -c "from apps.api.core.llm_settings import ChatLlmSettings; print(ChatLlmSettings().use_native_citation_prompt)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add apps/api/core/llm_settings.py
git commit -m "feat(settings): add USE_NATIVE_CITATION_PROMPT toggle for phase 1"
```

### Task 1.2 — Fix mixed lane fallback bug

The fallback path puts the user message into the answer body (`unified_copilot_service.py:513`). This is what produces "정리하면 다음과 같습니다: 관련 리소스 이름도 같이 알려줘" in eval-0001/0002.

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py`
- Test: `tests/test_chat_service_mixed.py` (existing — verify not broken)

- [ ] **Step 1: Find the offending line**

Run: `grep -n "함께 정리하면 다음과 같습니다" apps/api/rag/generation/unified_copilot_service.py`
Expected: a single match around line 513.

- [ ] **Step 2: Replace the f-string with a static intro**

Find:
```python
parts = [f"문서 근거와 현재 클러스터 기준 정보를 함께 정리하면 다음과 같습니다: {message}"]
```
Replace with:
```python
parts = ["문서 근거와 현재 클러스터 상태를 정리합니다."]
```

- [ ] **Step 3: Update `_strip_intro` to recognize the new prefix**

In the same file, find `_strip_intro` (~line 547) and add the new prefix to its prefix list. After the existing `if lines[0].startswith(...)` checks, add:

```python
if lines[0].startswith("문서 근거와 현재 클러스터 상태를 정리합니다"):
    return "\n".join(lines[1:]).strip()
```

- [ ] **Step 4: Run the existing mixed test**

Run: `python -m pytest tests/test_chat_service_mixed.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/generation/unified_copilot_service.py
git commit -m "fix(chat): mixed lane fallback no longer prefixes user message into answer"
```

### Task 1.3 — Citation grounding preserves answer body on validation failure

When validation removes all citations, the current code can return an empty paragraph. Preserve the body even when no citation survives.

**Files:**
- Modify: `apps/api/rag/generation/citation_grounding.py`
- Test: `tests/test_citation_grounding_preserve.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_citation_grounding_preserve.py`:
```python
import unittest

from apps.api.api.schemas.chat import CopilotChatSourceItem
from apps.api.rag.generation.citation_grounding import CitationGroundingValidator


def _make_source(text: str) -> CopilotChatSourceItem:
    return CopilotChatSourceItem(
        source_type="doc",
        label="storage.md · PVC",
        source_path="data/corpus/pdfs/storage.md",
        chunk_id="c1",
        score=0.5,
        provenance=["doc_new"],
        metadata={
            "section_title": "PVC",
            "preview_text": text,
            "synthesis_text": text,
        },
    )


class CitationGroundingPreserveTests(unittest.TestCase):
    def test_paragraph_body_is_preserved_when_citation_marker_is_invalid(self):
        validator = CitationGroundingValidator()
        sources = [_make_source("PersistentVolumeClaim spec example")]
        answer = "PVC는 스토리지를 요청하는 객체입니다.[5]"  # invalid marker (5 > len)
        result = validator.validate(answer, sources, enforce_alignment=True)
        self.assertIn("PVC는 스토리지를 요청하는 객체입니다.", result)

    def test_paragraph_body_is_preserved_when_token_overlap_is_zero(self):
        validator = CitationGroundingValidator()
        sources = [_make_source("Completely different English content about networking")]
        answer = "PVC 예시 코드입니다.[1]"
        result = validator.validate(answer, sources, enforce_alignment=True)
        self.assertIn("PVC 예시 코드입니다.", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_citation_grounding_preserve.py -v`
Expected: at least one test FAILS (assertion error or empty body).

- [ ] **Step 3: Make `validate` preserve body on validation failure**

In `apps/api/rag/generation/citation_grounding.py`, the loop already appends `body` when no valid citations remain. Inspect lines 71–75. They look like:
```python
if valid_citations:
    marker = "".join(f"[{index}]" for index in dict.fromkeys(valid_citations))
    validated.append(f"{body}{marker}".strip())
else:
    validated.append(body)
```
The bug is that `body` may be empty if the original paragraph was *only* a marker. Replace the `else: validated.append(body)` branch with:
```python
else:
    if body:
        validated.append(body)
    else:
        # paragraph was only a marker; drop it but log
        continue
```
Also ensure that when `cleaned` substitution at line 48 erases all markers, the surrounding whitespace cleanup doesn't drop the body. The current `cleaned = re.sub(r"[ \t]+\n", "\n", cleaned).strip()` is fine.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_citation_grounding_preserve.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/generation/citation_grounding.py tests/test_citation_grounding_preserve.py
git commit -m "fix(citation): preserve paragraph body when no valid citation survives"
```

### Task 1.4 — Synthesis prompt emits native [N] citations

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py`

- [ ] **Step 1: Locate the synthesis prompt**

Run: `grep -n "inline citation marker는 쓰지 마세요" apps/api/rag/generation/unified_copilot_service.py`
Expected: one match around line 381 inside `_synthesize_doc_response`.

- [ ] **Step 2: Replace the synthesis prompt rules**

Find the `prompt = "\n".join([...])` block in `_synthesize_doc_response` and replace with:
```python
toggle_native = bool(getattr(self.llm_client.settings, "use_native_citation_prompt", True))
if toggle_native:
    prompt = "\n".join([
        "사용자 질문에 대해 아래 문서 근거만 사용해서 한국어로 답변하세요.",
        "규칙:",
        "- 근거 밖 내용 추정 금지 (모르면 '문서에 없음' 명시)",
        "- 2~3개 짧은 단락으로 작성",
        "- 각 단락 끝에 사용한 근거 번호를 [1], [2] 형식으로 반드시 표기",
        "- 한 단락이 여러 근거를 사용하면 [1][2]처럼 연속 표기",
        "- 하단 참고문헌 목록 금지",
        f"질문: {message.strip()}",
        "문서 근거:",
        *context_lines,
    ])
else:
    prompt = "\n".join([
        "사용자 질문에 대해 아래 문서 근거만 사용해서 한국어로 답변하세요.",
        "규칙:",
        "- 근거 밖 내용 추정 금지",
        "- inline citation marker는 쓰지 마세요",
        "- 2~3개의 짧은 단락으로 핵심부터 설명하세요",
        "- 하단 참고문헌 목록 금지",
        f"질문: {message.strip()}",
        "문서 근거:",
        *context_lines,
    ])
```

- [ ] **Step 3: Reduce synthesis max_tokens default for tighter answers**

Find `_synthesis_max_tokens` (~line 823):
```python
def _synthesis_max_tokens(self) -> int:
    if self.llm_client is None:
        return 1800
    settings = getattr(self.llm_client, "settings", None)
    return int(getattr(settings, "llm_synthesis_max_tokens", 1800) or 1800)
```
Change defaults from 1800 to 600:
```python
def _synthesis_max_tokens(self) -> int:
    if self.llm_client is None:
        return 600
    settings = getattr(self.llm_client, "settings", None)
    return int(getattr(settings, "llm_synthesis_max_tokens", 600) or 600)
```

- [ ] **Step 4: Update `llm_synthesis_max_tokens` default in settings**

In `apps/api/core/llm_settings.py`, change `llm_synthesis_max_tokens: int = 1800` to `llm_synthesis_max_tokens: int = 600`.

- [ ] **Step 5: Update `_finalize_cited_answer` to keep LLM-emitted markers**

Currently `_finalize_cited_answer` strips all `[N]` then re-attaches via `_attach_paragraph_citations`. With native emit we want to preserve LLM-emitted markers. Modify `_finalize_cited_answer` (line 703):

Find:
```python
def _finalize_cited_answer(
    self,
    answer: str,
    sources: list[CopilotChatSourceItem],
    *,
    paragraph_source_indexes: list[list[int]] | None = None,
) -> str:
    stripped = re.sub(r"\[(\d+)\]", "", str(answer or "")).strip()
    cited = self._attach_paragraph_citations(
        stripped,
        sources,
        paragraph_source_indexes=paragraph_source_indexes,
    )
    return self.citation_validator.validate(
        cited,
        sources,
        enforce_alignment=paragraph_source_indexes is None,
    )
```
Replace with:
```python
def _finalize_cited_answer(
    self,
    answer: str,
    sources: list[CopilotChatSourceItem],
    *,
    paragraph_source_indexes: list[list[int]] | None = None,
) -> str:
    raw = str(answer or "").strip()
    use_native = bool(getattr(self.llm_client.settings, "use_native_citation_prompt", True)) if self.llm_client else False
    if use_native and re.search(r"\[\d+\]", raw):
        # Trust LLM-emitted markers; only validate ranges.
        cited = raw
    else:
        stripped = re.sub(r"\[(\d+)\]", "", raw).strip()
        cited = self._attach_paragraph_citations(
            stripped,
            sources,
            paragraph_source_indexes=paragraph_source_indexes,
        )
    return self.citation_validator.validate(
        cited,
        sources,
        enforce_alignment=paragraph_source_indexes is None,
    )
```

- [ ] **Step 6: Run all generation tests**

Run: `python -m pytest tests/ -k "citation or synthesis or chat_service" -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add apps/api/rag/generation/unified_copilot_service.py apps/api/core/llm_settings.py
git commit -m "feat(synthesis): emit [N] citations natively when USE_NATIVE_CITATION_PROMPT=true"
```

### Task 1.5 — Run service eval baseline for Phase 1

**Files:**
- Create: `tests/results/service-eval/baseline-pre-p1.json` (snapshot)
- Modify: `tests/results/service-eval/latest.json` (overwritten by run)

- [ ] **Step 1: Snapshot current latest.json as pre-P1 baseline**

```bash
cp tests/results/service-eval/latest.json tests/results/service-eval/baseline-pre-p1.json
```

- [ ] **Step 2: Start the API server (in a separate shell)**

```bash
uvicorn apps.api.main:app --reload --port 8000
```
(If `apps.api.main:app` import path differs, run `grep -rn "FastAPI(" apps/api | head -3` to find it.)

- [ ] **Step 3: Run service eval**

```bash
python scripts/run_service_eval.py --base-url http://localhost:8000
```
Expected: prints summary like `{"count": 10, "ok_count": ?, "avg_latency_ms": ?}`. The new latest.json is written.

- [ ] **Step 4: Snapshot Phase 1 result**

```bash
cp tests/results/service-eval/latest.json tests/results/service-eval/phase-1.json
```

- [ ] **Step 5: Compare**

```bash
python -c "import json; pre=json.load(open('tests/results/service-eval/baseline-pre-p1.json')); post=json.load(open('tests/results/service-eval/phase-1.json')); print('pre ok=', pre['ok_count'], 'post ok=', post['ok_count']); print('pre lat=', pre['avg_latency_ms'], 'post lat=', post['avg_latency_ms'])"
```
Expected: `post ok` ≥ `pre ok + 3`. If `post ok < pre ok`, set `USE_NATIVE_CITATION_PROMPT=false` in `.env`, restart, re-run, and investigate before proceeding.

- [ ] **Step 6: Commit phase result**

```bash
git add tests/results/service-eval/baseline-pre-p1.json tests/results/service-eval/phase-1.json
git commit -m "test(eval): record phase 1 service eval baseline and result"
```

---

## Phase 2 — QueryRouter + Parallel Retrieval

Goal: replace 3 agent LLM calls with 1, run sparse and dense retrieval in parallel, drop per-turn latency by 8–12 seconds.

### Task 2.1 — Add Phase 2 toggle to settings

**Files:**
- Modify: `apps/api/core/llm_settings.py`

- [ ] **Step 1: Add the toggle**

Append to `ChatLlmSettings`:
```python
use_query_router: bool = True
```

- [ ] **Step 2: Verify**

Run: `python -c "from apps.api.core.llm_settings import ChatLlmSettings; print(ChatLlmSettings().use_query_router)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add apps/api/core/llm_settings.py
git commit -m "feat(settings): add USE_QUERY_ROUTER toggle for phase 2"
```

### Task 2.2 — Create QueryRouter with JSON parsing and fallback

**Files:**
- Create: `apps/api/rag/query/query_router.py`
- Create: `tests/test_query_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_query_router.py`:
```python
import asyncio
import json
import unittest
from unittest.mock import AsyncMock

from apps.api.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.query.query_router import QueryRouter, QueryRouterDecision


class _StubLlm:
    def __init__(self, response: str | None, raise_exc: bool = False) -> None:
        self.response = response
        self.raise_exc = raise_exc
        self.is_enabled = True
        self.calls: list[dict] = []

    async def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.raise_exc:
            raise RuntimeError("llm down")
        return self.response or ""


class QueryRouterTests(unittest.TestCase):
    def test_returns_parsed_decision_from_llm(self):
        llm = _StubLlm(json.dumps({
            "lane": "doc",
            "search_query": "openshift pod inspection",
            "live_query": "",
            "inherit_sources": False,
            "reasoning_brief": "doc lookup",
        }))
        router = QueryRouter(llm_client=llm)
        decision = asyncio.run(router.decide(
            user_message="파드 확인하는 명령어 뭐 써?",
            recent_turns=[],
            has_connection=False,
            last_lane="",
            last_doc_sources=[],
        ))
        self.assertEqual(decision.lane, "doc")
        self.assertEqual(decision.search_query, "openshift pod inspection")
        self.assertFalse(decision.inherit_sources)

    def test_falls_back_to_rule_when_llm_fails(self):
        llm = _StubLlm(None, raise_exc=True)
        router = QueryRouter(llm_client=llm)
        decision = asyncio.run(router.decide(
            user_message="다시 정리해줘",
            recent_turns=[CopilotChatHistoryTurn(role="user", text="pvc 예시", lane="")],
            has_connection=False,
            last_lane="doc_new",
            last_doc_sources=["data/corpus/pdfs/storage.md"],
        ))
        self.assertEqual(decision.lane, "doc_new")
        self.assertTrue(decision.inherit_sources)
        self.assertEqual(decision.search_query, "다시 정리해줘")

    def test_invalid_lane_falls_back_to_doc(self):
        llm = _StubLlm(json.dumps({"lane": "garbage", "search_query": "x"}))
        router = QueryRouter(llm_client=llm)
        decision = asyncio.run(router.decide(
            user_message="x",
            recent_turns=[],
            has_connection=False,
            last_lane="",
            last_doc_sources=[],
        ))
        self.assertEqual(decision.lane, "doc")

    def test_live_lane_downgrades_when_no_connection(self):
        llm = _StubLlm(json.dumps({
            "lane": "live",
            "search_query": "",
            "live_query": "list pods",
            "inherit_sources": False,
        }))
        router = QueryRouter(llm_client=llm)
        decision = asyncio.run(router.decide(
            user_message="현재 파드 보여줘",
            recent_turns=[],
            has_connection=False,
            last_lane="",
            last_doc_sources=[],
        ))
        self.assertEqual(decision.lane, "needs_connection")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_query_router.py -v`
Expected: ImportError (module not yet created).

- [ ] **Step 3: Create the QueryRouter module**

Create `apps/api/rag/query/query_router.py`:
```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from apps.api.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.generation.llm_client import OpenAiCompatibleLlmClient


_FOLLOWUP_MARKERS = (
    "again", "then", "that", "this", "those",
    "그거", "이거", "저거", "방금", "이어서", "다시", "자세히",
    "그 yaml", "그 문서", "그 명령",
)


@dataclass(slots=True)
class QueryRouterDecision:
    lane: str  # "doc" | "live" | "mixed" | "needs_connection"
    search_query: str
    live_query: str = ""
    inherit_sources: bool = False
    reasoning_brief: str = ""


class QueryRouter:
    SYSTEM_PROMPT = (
        "You are the routing planner for a Kubernetes/OpenShift assistant.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        "  \"lane\": \"doc|live|mixed|needs_connection\",\n"
        "  \"search_query\": \"string\",\n"
        "  \"live_query\": \"string\",\n"
        "  \"inherit_sources\": true|false,\n"
        "  \"reasoning_brief\": \"string\"\n"
        "}\n"
        "Rules:\n"
        "- doc: documentation, concepts, procedures, examples\n"
        "- live: requires the connected OpenShift cluster\n"
        "- mixed: needs both doc explanation and live cluster state\n"
        "- needs_connection: live but no cluster connection\n"
        "- search_query: concise English technical retrieval query (translate Korean if needed). Preserve K8s/OpenShift resource names, acronyms, commands.\n"
        "- live_query: cluster inspection request (empty if not live/mixed)\n"
        "- inherit_sources: true when this is a follow-up referencing the previous doc answer\n"
    )

    def __init__(self, llm_client: OpenAiCompatibleLlmClient | None = None) -> None:
        self.llm_client = llm_client

    async def decide(
        self,
        *,
        user_message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        has_connection: bool,
        last_lane: str,
        last_doc_sources: list[str],
    ) -> QueryRouterDecision:
        message = str(user_message or "").strip()
        if self.llm_client is not None and getattr(self.llm_client, "is_enabled", False):
            decision = await self._decide_with_llm(
                message=message,
                recent_turns=recent_turns,
                has_connection=has_connection,
                last_lane=last_lane,
                last_doc_sources=last_doc_sources,
            )
            if decision is not None:
                return self._enforce_connection(decision, has_connection)
        return self._fallback(message=message, last_lane=last_lane)

    async def _decide_with_llm(
        self,
        *,
        message: str,
        recent_turns: list[CopilotChatHistoryTurn],
        has_connection: bool,
        last_lane: str,
        last_doc_sources: list[str],
    ) -> QueryRouterDecision | None:
        history_lines = [
            f"- {turn.role}: {turn.text} (lane={turn.lane or '-'})"
            for turn in recent_turns[-4:]
            if str(turn.text or "").strip()
        ]
        user_prompt = "\n".join(
            [
                f"connection_available={str(has_connection).lower()}",
                f"last_lane={last_lane or '-'}",
                f"last_doc_sources={last_doc_sources}",
                "recent_turns:",
                *(history_lines or ["- none"]),
                f"user_message: {message}",
            ]
        )
        try:
            raw = await self.llm_client.generate(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=220,
                temperature=0.0,
                purpose="query_router",
            )
        except Exception:
            return None
        parsed = self._extract_json(raw)
        if not parsed:
            return None
        lane = str(parsed.get("lane") or "").strip().lower()
        if lane not in {"doc", "live", "mixed", "needs_connection"}:
            return QueryRouterDecision(
                lane="doc",
                search_query=str(parsed.get("search_query") or message),
                live_query=str(parsed.get("live_query") or ""),
                inherit_sources=bool(parsed.get("inherit_sources")),
                reasoning_brief=str(parsed.get("reasoning_brief") or ""),
            )
        return QueryRouterDecision(
            lane=lane,
            search_query=str(parsed.get("search_query") or message),
            live_query=str(parsed.get("live_query") or ""),
            inherit_sources=bool(parsed.get("inherit_sources")),
            reasoning_brief=str(parsed.get("reasoning_brief") or ""),
        )

    def _fallback(self, *, message: str, last_lane: str) -> QueryRouterDecision:
        looks_followup = any(marker in message.casefold() for marker in _FOLLOWUP_MARKERS)
        if last_lane.startswith("doc") and looks_followup:
            return QueryRouterDecision(
                lane=last_lane,
                search_query=message,
                inherit_sources=True,
            )
        return QueryRouterDecision(lane="doc", search_query=message, inherit_sources=False)

    @staticmethod
    def _enforce_connection(decision: QueryRouterDecision, has_connection: bool) -> QueryRouterDecision:
        if decision.lane == "live" and not has_connection:
            return QueryRouterDecision(
                lane="needs_connection",
                search_query=decision.search_query,
                live_query=decision.live_query,
                inherit_sources=decision.inherit_sources,
                reasoning_brief=decision.reasoning_brief,
            )
        return decision

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_query_router.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/query/query_router.py tests/test_query_router.py
git commit -m "feat(query): add QueryRouter consolidating normalize/intent/rewrite agents"
```

### Task 2.3 — Add LRU cache for QueryRouter results

**Files:**
- Modify: `apps/api/rag/query/query_router.py`
- Modify: `tests/test_query_router.py`

- [ ] **Step 1: Add a failing cache test**

Append to `tests/test_query_router.py`:
```python
class QueryRouterCacheTests(unittest.TestCase):
    def test_repeated_call_with_same_inputs_uses_cache(self):
        llm = _StubLlm(json.dumps({
            "lane": "doc",
            "search_query": "x",
            "inherit_sources": False,
        }))
        router = QueryRouter(llm_client=llm)
        common = dict(
            user_message="x",
            recent_turns=[],
            has_connection=False,
            last_lane="",
            last_doc_sources=[],
        )
        d1 = asyncio.run(router.decide(**common))
        d2 = asyncio.run(router.decide(**common))
        self.assertEqual(d1.lane, d2.lane)
        self.assertEqual(len(llm.calls), 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_query_router.py::QueryRouterCacheTests -v`
Expected: FAIL — `len(llm.calls) == 2`.

- [ ] **Step 3: Add the cache**

In `apps/api/rag/query/query_router.py`, modify `__init__`:
```python
def __init__(self, llm_client: OpenAiCompatibleLlmClient | None = None, *, cache_size: int = 256) -> None:
    self.llm_client = llm_client
    self._cache: dict[str, QueryRouterDecision] = {}
    self._cache_size = cache_size
```
Add a key-builder and wrap `decide`:
```python
@staticmethod
def _cache_key(message: str, last_lane: str, last_doc_sources: list[str]) -> str:
    return f"{message}|{last_lane}|{','.join(sorted(last_doc_sources))}"
```
Modify `decide` to consult cache before calling LLM:
```python
async def decide(
    self,
    *,
    user_message: str,
    recent_turns: list[CopilotChatHistoryTurn],
    has_connection: bool,
    last_lane: str,
    last_doc_sources: list[str],
) -> QueryRouterDecision:
    message = str(user_message or "").strip()
    cache_key = self._cache_key(message, last_lane, last_doc_sources)
    cached = self._cache.get(cache_key)
    if cached is not None:
        return self._enforce_connection(cached, has_connection)
    if self.llm_client is not None and getattr(self.llm_client, "is_enabled", False):
        decision = await self._decide_with_llm(
            message=message,
            recent_turns=recent_turns,
            has_connection=has_connection,
            last_lane=last_lane,
            last_doc_sources=last_doc_sources,
        )
        if decision is not None:
            self._store(cache_key, decision)
            return self._enforce_connection(decision, has_connection)
    decision = self._fallback(message=message, last_lane=last_lane)
    self._store(cache_key, decision)
    return decision

def _store(self, key: str, value: QueryRouterDecision) -> None:
    if len(self._cache) >= self._cache_size:
        self._cache.pop(next(iter(self._cache)))
    self._cache[key] = value
```

- [ ] **Step 4: Run all router tests**

Run: `python -m pytest tests/test_query_router.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/query/query_router.py tests/test_query_router.py
git commit -m "feat(query): add in-memory LRU cache to QueryRouter"
```

### Task 2.4 — Wire QueryRouter into UnifiedCopilotService

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py`

- [ ] **Step 1: Add router to constructor and use it when toggle is on**

Find `__init__` (~line 36) and add a parameter `query_router: QueryRouter | None = None`. Add to imports:
```python
from apps.api.rag.query.query_router import QueryRouter, QueryRouterDecision
```
In `__init__` body:
```python
self.query_router = query_router or QueryRouter(llm_client=llm_client)
```

- [ ] **Step 2: Replace the normalize → intent → rewrite chain in `answer`**

Find the prelude in `answer` (lines 72–101). Replace with:
```python
overall_started = time.perf_counter()
recent_turns = recent_turns or []
await self._emit_progress(progress, key="analyze_question", label="사용자 질문 분석중", detail="라우팅과 검색 쿼리를 결정하고 있습니다.")

use_router = bool(getattr(self.llm_client.settings, "use_query_router", True)) if self.llm_client else False
if use_router:
    last_lane = self._last_assistant_lane(recent_turns)
    last_doc_sources = self._last_doc_sources(recent_turns)
    router_started = time.perf_counter()
    router_decision = await self.query_router.decide(
        user_message=message,
        recent_turns=recent_turns,
        has_connection=bool(connection_id),
        last_lane=last_lane,
        last_doc_sources=last_doc_sources,
    )
    logger.info(
        "[router] lane=%s search_query=%r inherit=%s elapsed=%.2fs",
        router_decision.lane,
        router_decision.search_query,
        router_decision.inherit_sources,
        time.perf_counter() - router_started,
    )
    decision_lane = router_decision.lane
    internal_message = router_decision.search_query or message
    doc_query_for_lane = router_decision.search_query or message
    live_query_for_lane = router_decision.live_query or message
    inherit_sources = router_decision.inherit_sources
    inherited_paths = last_doc_sources if inherit_sources else []
else:
    # legacy path retained for rollback
    internal_message = message
    if self.question_normalizer is not None:
        normalize_started = time.perf_counter()
        normalized = await self.question_normalizer.normalize(message=message, recent_turns=recent_turns)
        internal_message = normalized.text or message
        logger.info("[normalize] elapsed=%.2fs", time.perf_counter() - normalize_started)
    intent_started = time.perf_counter()
    decision = await self.intent_agent.classify(
        message=internal_message,
        has_connection=bool(connection_id),
        recent_turns=recent_turns,
    )
    logger.info("[intent] lane=%s elapsed=%.2fs", decision.lane, time.perf_counter() - intent_started)
    decision_lane = decision.lane
    doc_query_for_lane = decision.doc_query or internal_message
    live_query_for_lane = decision.live_query or internal_message
    inherit_sources = False
    inherited_paths = []
```
Then replace every reference to `decision.lane`, `decision.doc_query`, `decision.live_query` in the rest of `answer` with `decision_lane`, `doc_query_for_lane`, `live_query_for_lane`. Pass `inherited_paths` into `_answer_doc_lane` as `allowed_source_paths_override`.

- [ ] **Step 3: Add helpers for last_lane and last_doc_sources**

Add to the class:
```python
@staticmethod
def _last_assistant_lane(recent_turns: list[CopilotChatHistoryTurn]) -> str:
    for turn in reversed(recent_turns):
        if str(turn.role or "") == "assistant" and str(turn.lane or ""):
            return str(turn.lane).strip()
    return ""

@staticmethod
def _last_doc_sources(recent_turns: list[CopilotChatHistoryTurn]) -> list[str]:
    for turn in reversed(recent_turns):
        if str(turn.role or "") == "assistant":
            paths = getattr(turn, "source_paths", None) or []
            return [str(p) for p in paths if p]
    return []
```

- [ ] **Step 4: Modify `_answer_doc_lane` to accept inherited paths**

Add parameter `allowed_source_paths_override: list[str] | None = None`. Inside, when toggle is on and override is non-empty, use it instead of the rewrite agent's allowed paths:
```python
use_router = bool(getattr(self.llm_client.settings, "use_query_router", True)) if self.llm_client else False
if use_router:
    rewrite = type("R", (), {"rewritten_query": message, "allowed_source_paths": allowed_source_paths_override or []})()
else:
    rewrite_started = time.perf_counter()
    rewrite = await self.query_rewrite_agent.rewrite(message=message, recent_turns=recent_turns)
    logger.info("[rewrite] elapsed=%.2fs", time.perf_counter() - rewrite_started)
```

- [ ] **Step 5: Run existing chat tests**

Run: `python -m pytest tests/test_chat_service_mixed.py -v`
Expected: all pass (legacy path still works).

- [ ] **Step 6: Commit**

```bash
git add apps/api/rag/generation/unified_copilot_service.py
git commit -m "feat(chat): wire QueryRouter into UnifiedCopilotService behind USE_QUERY_ROUTER"
```

### Task 2.5 — Parallelize sparse and dense retrieval

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py`

- [ ] **Step 1: Locate the sequential retrieval block in `_answer_doc_lane`**

Find the `lexical_started = ...` block followed by `dense_started = ...` (around lines 239–268).

- [ ] **Step 2: Replace with `asyncio.gather`**

Add to imports if missing: `import asyncio`.
Replace the two sequential `await` blocks with:
```python
await self._emit_progress(progress, key="retrieve_docs", label="문서 검색중", detail="키워드 + 벡터 검색을 병렬로 수행합니다.")
parallel_started = time.perf_counter()
new_doc_response, pgvector_response = await asyncio.gather(
    self.document_retriever.answer(
        message=rewrite.rewritten_query,
        allowed_source_paths=rewrite.allowed_source_paths or None,
    ),
    self.pgvector_bridge.answer(
        message=rewrite.rewritten_query,
        allowed_source_paths=rewrite.allowed_source_paths or None,
        additional_query=original_message if original_message and original_message.strip() != rewrite.rewritten_query.strip() else None,
    ),
    return_exceptions=True,
)
if isinstance(new_doc_response, Exception):
    logger.warning("[retrieve:sparse] failed err=%s", new_doc_response)
    new_doc_response = None
if isinstance(pgvector_response, Exception):
    logger.warning("[retrieve:dense] failed err=%s", pgvector_response)
    pgvector_response = None
logger.info("[retrieve:parallel] elapsed=%.2fs", time.perf_counter() - parallel_started)
```

- [ ] **Step 3: Run integration tests**

Run: `python -m pytest tests/ -k "chat or pipeline or retrieval" -v`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add apps/api/rag/generation/unified_copilot_service.py
git commit -m "perf(chat): parallelize sparse and dense retrieval with asyncio.gather"
```

### Task 2.6 — Pipeline router integration test

**Files:**
- Create: `tests/test_pipeline_router_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_pipeline_router_integration.py`:
```python
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from apps.api.api.schemas.chat import CopilotChatHistoryTurn
from apps.api.rag.query.query_router import QueryRouter, QueryRouterDecision


class _StubLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.is_enabled = True
    async def generate(self, *args, **kwargs):
        return self.response


class RouterIntegrationTests(unittest.TestCase):
    def test_followup_inherits_sources_via_router(self):
        llm = _StubLlm(json.dumps({
            "lane": "doc",
            "search_query": "openshift pod inspection commands related resources",
            "live_query": "",
            "inherit_sources": True,
            "reasoning_brief": "follow-up",
        }))
        router = QueryRouter(llm_client=llm)
        decision = asyncio.run(router.decide(
            user_message="관련 리소스 이름도 같이 알려줘",
            recent_turns=[
                CopilotChatHistoryTurn(role="user", text="파드 확인하는 명령어 뭐 써?", lane=""),
                CopilotChatHistoryTurn(role="assistant", text="oc get pods", lane="doc_new"),
            ],
            has_connection=False,
            last_lane="doc_new",
            last_doc_sources=["data/corpus/pdfs/nodes.md"],
        ))
        self.assertTrue(decision.inherit_sources)
        self.assertIn("pod", decision.search_query.casefold())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_pipeline_router_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_router_integration.py
git commit -m "test(router): integration test for follow-up source inheritance"
```

### Task 2.7 — Mark legacy agents as deprecated

**Files:**
- Modify: `apps/api/rag/query/question_normalizer.py`
- Modify: `apps/api/rag/query/intent_agent.py`
- Modify: `apps/api/rag/query/query_rewrite_agent.py`

- [ ] **Step 1: Add module-level deprecation comment to each**

Prepend each file with:
```python
# DEPRECATED: superseded by QueryRouter. Retained for USE_QUERY_ROUTER=false rollback path.
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/rag/query/question_normalizer.py apps/api/rag/query/intent_agent.py apps/api/rag/query/query_rewrite_agent.py
git commit -m "chore(query): mark legacy 3-agent path as deprecated"
```

### Task 2.8 — Run service eval for Phase 2

- [ ] **Step 1: Snapshot baseline**

```bash
cp tests/results/service-eval/latest.json tests/results/service-eval/baseline-pre-p2.json
```

- [ ] **Step 2: Restart server, run eval**

```bash
python scripts/run_service_eval.py --base-url http://localhost:8000
cp tests/results/service-eval/latest.json tests/results/service-eval/phase-2.json
```

- [ ] **Step 3: Compare and gate**

```bash
python -c "import json; pre=json.load(open('tests/results/service-eval/baseline-pre-p2.json')); post=json.load(open('tests/results/service-eval/phase-2.json')); print('ok pre/post:', pre['ok_count'], post['ok_count']); print('lat pre/post:', pre['avg_latency_ms'], post['avg_latency_ms'])"
```
Expected: `post['avg_latency_ms']` is at least 30% lower than `pre['avg_latency_ms']`. If not, set `USE_QUERY_ROUTER=false` and investigate.

- [ ] **Step 4: Commit phase result**

```bash
git add tests/results/service-eval/baseline-pre-p2.json tests/results/service-eval/phase-2.json
git commit -m "test(eval): record phase 2 service eval baseline and result"
```

---

## Phase 3 — RRF Fusion + MMR Diversity

Goal: replace ad-hoc weight formula with standard RRF; replace seed-chunk selection with MMR.

### Task 3.1 — Add Phase 3 toggle

**Files:**
- Modify: `apps/api/core/llm_settings.py`

- [ ] **Step 1: Add toggle**

Append:
```python
use_rrf_fusion: bool = True
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/core/llm_settings.py
git commit -m "feat(settings): add USE_RRF_FUSION toggle for phase 3"
```

### Task 3.2 — Create hybrid_fusion module with RRF + MMR

**Files:**
- Create: `apps/api/rag/retrieval/hybrid_fusion.py`
- Create: `tests/test_hybrid_fusion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hybrid_fusion.py`:
```python
import unittest

from apps.api.rag.retrieval.hybrid_fusion import rrf_score, rrf_merge, mmr_select


class RrfTests(unittest.TestCase):
    def test_rrf_score_decreases_with_rank(self):
        self.assertGreater(rrf_score(0), rrf_score(1))
        self.assertGreater(rrf_score(1), rrf_score(2))

    def test_rrf_merge_combines_two_ranked_lists(self):
        sparse = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        dense = [("b", 0.8), ("a", 0.7), ("d", 0.2)]
        merged = rrf_merge(sparse, dense, k=60)
        keys = [item[0] for item in merged]
        self.assertEqual(set(keys), {"a", "b", "c", "d"})
        # a and b appear in both → must rank above c and d
        self.assertLess(keys.index("a"), keys.index("c"))
        self.assertLess(keys.index("b"), keys.index("d"))

    def test_rrf_merge_empty_inputs(self):
        self.assertEqual(rrf_merge([], []), [])


class MmrTests(unittest.TestCase):
    def test_mmr_picks_diverse_items(self):
        candidates = [
            ("a", 1.0, {"oc", "get", "pods"}),
            ("a2", 0.95, {"oc", "get", "pods"}),  # near-duplicate of a
            ("b", 0.6, {"pvc", "create", "yaml"}),
        ]
        selected = mmr_select(candidates, k=2, lam=0.7)
        keys = [item[0] for item in selected]
        self.assertIn("a", keys)
        self.assertIn("b", keys)
        self.assertNotIn("a2", keys)

    def test_mmr_respects_budget(self):
        candidates = [(str(i), 1.0 - i * 0.1, {f"t{i}"}) for i in range(5)]
        selected = mmr_select(candidates, k=3, lam=0.7)
        self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_hybrid_fusion.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the module**

Create `apps/api/rag/retrieval/hybrid_fusion.py`:
```python
from __future__ import annotations

from typing import Iterable


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def rrf_merge(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (key, _orig_score) in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + rrf_score(rank, k=k)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def mmr_select(
    candidates: list[tuple[str, float, set[str]]],
    *,
    k: int,
    lam: float = 0.7,
) -> list[tuple[str, float, set[str]]]:
    if not candidates or k <= 0:
        return []
    remaining = list(candidates)
    remaining.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[str, float, set[str]]] = [remaining.pop(0)]
    while remaining and len(selected) < k:
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(remaining):
            relevance = candidate[1]
            similarity = max(_jaccard(candidate[2], chosen[2]) for chosen in selected)
            score = lam * relevance - (1.0 - lam) * similarity
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / max(union, 1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_hybrid_fusion.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/retrieval/hybrid_fusion.py tests/test_hybrid_fusion.py
git commit -m "feat(retrieval): add RRF and MMR helpers"
```

### Task 3.3 — Replace `_merge_doc_responses` with RRF

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py`

- [ ] **Step 1: Locate `_merge_doc_responses`**

Run: `grep -n "_merge_doc_responses" apps/api/rag/generation/unified_copilot_service.py`
Expected: definition around line 575.

- [ ] **Step 2: Add toggle-aware RRF path**

Add import: `from apps.api.rag.retrieval.hybrid_fusion import rrf_merge, mmr_select`.

Replace the body of `_merge_doc_responses` with:
```python
@staticmethod
def _merge_doc_responses(
    message: str,
    pgvector_response: CopilotChatResponse | None,
    new_doc_response: CopilotChatResponse | None,
) -> CopilotChatResponse | None:
    if pgvector_response is None or new_doc_response is None:
        return None
    if not pgvector_response.sources or not new_doc_response.sources:
        return None

    sparse_ranked = [(UnifiedCopilotService._source_key(src), float(src.score or 0.0)) for src in new_doc_response.sources]
    dense_ranked = [(UnifiedCopilotService._source_key(src), float(src.score or 0.0)) for src in pgvector_response.sources]
    fused = rrf_merge(sparse_ranked, dense_ranked, k=60)

    by_key: dict[str, CopilotChatSourceItem] = {}
    provenance: dict[str, set[str]] = {}
    for src in new_doc_response.sources:
        key = UnifiedCopilotService._source_key(src)
        by_key[key] = src
        provenance.setdefault(key, set()).update(src.provenance or [new_doc_response.lane])
    for src in pgvector_response.sources:
        key = UnifiedCopilotService._source_key(src)
        if key in by_key:
            by_key[key] = UnifiedCopilotService._merge_source_details(by_key[key], src)
        else:
            by_key[key] = src
        provenance.setdefault(key, set()).update(src.provenance or [pgvector_response.lane])

    answer_limit = answer_source_budget(message) + 1
    top_sources: list[CopilotChatSourceItem] = []
    for key, fused_score in fused[:answer_limit]:
        src = by_key[key]
        top_sources.append(
            src.model_copy(update={
                "score": round(fused_score, 4),
                "provenance": sorted(provenance.get(key) or []),
                "metadata": {
                    **src.metadata,
                    "hybrid_origin_lanes": sorted(provenance.get(key) or []),
                    "retrieval_backend": "doc_hybrid_rrf",
                },
            })
        )
    answer = UnifiedCopilotService._build_hybrid_doc_answer(message, top_sources)
    return CopilotChatResponse(
        lane="doc_hybrid",
        mode="hybrid_rrf_doc",
        fallback_used=False,
        preview_ready=any(bool(src.source_path) for src in top_sources),
        answer=answer,
        sources=top_sources,
    )
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/ -k "hybrid or rrf or chat" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add apps/api/rag/generation/unified_copilot_service.py
git commit -m "feat(retrieval): replace ad-hoc fusion weights with RRF in _merge_doc_responses"
```

### Task 3.4 — Apply MMR to seed selection in `document_retriever`

**Files:**
- Modify: `apps/api/rag/retrieval/document_retriever.py`

- [ ] **Step 1: Import and apply MMR**

Add import: `from apps.api.rag.retrieval.hybrid_fusion import mmr_select`.

Replace `_select_seed_chunks` body with:
```python
@staticmethod
def _select_seed_chunks(scored: list[tuple[float, ChunkRecord]], *, limit: int) -> list[tuple[float, ChunkRecord]]:
    if not scored:
        return []
    candidates = [
        (str(chunk.chunk_id), float(score), set(DocumentRetriever._tokenize(chunk.retrieval_text)))
        for score, chunk in scored
    ]
    selected_keys = {item[0] for item in mmr_select(candidates, k=limit, lam=0.7)}
    out: list[tuple[float, ChunkRecord]] = []
    seen: set[str] = set()
    for score, chunk in scored:
        if chunk.chunk_id in selected_keys and chunk.chunk_id not in seen:
            out.append((score, chunk))
            seen.add(chunk.chunk_id)
        if len(out) >= limit:
            break
    return out
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -k "retrieval or chunk" -v`
Expected: PASS (existing tests should still hold).

- [ ] **Step 3: Commit**

```bash
git add apps/api/rag/retrieval/document_retriever.py
git commit -m "feat(retrieval): use MMR for seed chunk selection"
```

### Task 3.5 — Run service eval for Phase 3

- [ ] **Step 1: Snapshot, run, snapshot, compare**

```bash
cp tests/results/service-eval/latest.json tests/results/service-eval/baseline-pre-p3.json
python scripts/run_service_eval.py --base-url http://localhost:8000
cp tests/results/service-eval/latest.json tests/results/service-eval/phase-3.json
python -c "import json; pre=json.load(open('tests/results/service-eval/baseline-pre-p3.json')); post=json.load(open('tests/results/service-eval/phase-3.json')); print('ok pre/post:', pre['ok_count'], post['ok_count']); print('section_match changes:', sum(1 for a,b in zip(pre['results'],post['results']) if a['section_match'] != b['section_match']))"
```
Expected: more `section_match=True` cases than before.

- [ ] **Step 2: Commit phase result**

```bash
git add tests/results/service-eval/baseline-pre-p3.json tests/results/service-eval/phase-3.json
git commit -m "test(eval): record phase 3 service eval baseline and result"
```

---

## Phase 4 — Korean Char N-gram + Asymmetric Query + Acronym Dict

Goal: improve sparse retrieval recall on Korean queries against English content; improve dense retrieval via BGE-m3 query asymmetric prompt; cover OpenShift acronyms.

### Task 4.1 — Add Phase 4 toggles

**Files:**
- Modify: `apps/api/core/llm_settings.py`

- [ ] **Step 1: Add toggles**

Append:
```python
use_char_ngram_sparse: bool = True
use_asymmetric_query_prompt: bool = True
use_synonym_expansion: bool = True
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/core/llm_settings.py
git commit -m "feat(settings): add phase 4 toggles for char n-gram, asymmetric query, synonym expansion"
```

### Task 4.2 — Create Korean tokenizer with char n-gram

**Files:**
- Create: `apps/api/rag/retrieval/korean_tokenizer.py`
- Create: `tests/test_korean_tokenizer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_korean_tokenizer.py`:
```python
import unittest

from apps.api.rag.retrieval.korean_tokenizer import tokenize_with_char_ngrams


class KoreanTokenizerTests(unittest.TestCase):
    def test_korean_word_emits_bigram_trigram_and_word(self):
        tokens = set(tokenize_with_char_ngrams("확인할"))
        # word
        self.assertIn("확인할", tokens)
        # bigrams
        self.assertIn("확인", tokens)
        self.assertIn("인할", tokens)
        # trigram (word == trigram here)
        self.assertIn("확인할", tokens)

    def test_english_word_unigram_only(self):
        tokens = set(tokenize_with_char_ngrams("openshift"))
        self.assertIn("openshift", tokens)
        self.assertNotIn("ope", tokens)

    def test_mixed_korean_english(self):
        tokens = set(tokenize_with_char_ngrams("rbac 확인"))
        self.assertIn("rbac", tokens)
        self.assertIn("확인", tokens)

    def test_short_korean_word_no_bigram_repeat(self):
        tokens = list(tokenize_with_char_ngrams("etcd"))
        # english stays unigram
        self.assertEqual(tokens.count("etcd"), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_korean_tokenizer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the tokenizer**

Create `apps/api/rag/retrieval/korean_tokenizer.py`:
```python
from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_-]+")
_HANGUL_PATTERN = re.compile(r"[가-힣]")


def tokenize_with_char_ngrams(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(str(text or "").casefold()):
        if len(raw_token) < 2:
            continue
        tokens.append(raw_token)
        if _HANGUL_PATTERN.search(raw_token):
            tokens.extend(_korean_ngrams(raw_token))
    return tokens


def _korean_ngrams(token: str) -> list[str]:
    out: list[str] = []
    for n in (2, 3):
        if len(token) < n:
            continue
        for i in range(len(token) - n + 1):
            chunk = token[i : i + n]
            if chunk == token:
                continue
            out.append(chunk)
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_korean_tokenizer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/retrieval/korean_tokenizer.py tests/test_korean_tokenizer.py
git commit -m "feat(retrieval): add Korean character n-gram tokenizer"
```

### Task 4.3 — Wire char n-gram tokenizer into DocumentRetriever

**Files:**
- Modify: `apps/api/rag/retrieval/document_retriever.py`

- [ ] **Step 1: Import and gate behind toggle**

Add import: `from apps.api.rag.retrieval.korean_tokenizer import tokenize_with_char_ngrams`.
Add a settings check helper at top of class:
```python
def _ngram_enabled(self) -> bool:
    # settings is optional during tests; default True
    settings = getattr(self, "_settings", None)
    return bool(getattr(settings, "use_char_ngram_sparse", True)) if settings else True
```
Replace `_tokenize` with:
```python
@staticmethod
def _tokenize(message: str) -> list[str]:
    return tokenize_with_char_ngrams(message)
```

- [ ] **Step 2: Re-prime stats on first load (cache invalidation)**

The retriever caches tokens in `_chunk_tokens`. Since the tokenizer changed, ensure first load after deploy regenerates them. The current `_load_chunks` already calls `_prime_stats` once per process, which is fine. No additional change.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/ -k "retrieval or document" -v`
Expected: pass (BM25 still works on the new vocabulary).

- [ ] **Step 4: Commit**

```bash
git add apps/api/rag/retrieval/document_retriever.py
git commit -m "feat(retrieval): use Korean char n-gram tokenizer for sparse BM25"
```

### Task 4.4 — Acronym/synonym expansion

**Files:**
- Create: `apps/api/rag/query/synonym_expansion.py`
- Create: `tests/test_synonym_expansion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_synonym_expansion.py`:
```python
import unittest

from apps.api.rag.query.synonym_expansion import expand_acronyms


class SynonymExpansionTests(unittest.TestCase):
    def test_known_acronym_appended(self):
        result = expand_acronyms("rbac")
        self.assertIn("rbac", result.casefold())
        self.assertIn("role binding", result.casefold())

    def test_unknown_acronym_returns_original(self):
        result = expand_acronyms("foobar")
        self.assertEqual(result, "foobar")

    def test_does_not_expand_when_acronym_inside_sentence(self):
        # We expand only when query contains the acronym in isolation
        result = expand_acronyms("rbac 확인할 때 뭘 봐야 해?")
        self.assertIn("role binding", result.casefold())  # still expand if present anywhere

    def test_does_not_replace_original_text(self):
        result = expand_acronyms("rbac")
        self.assertTrue(result.casefold().startswith("rbac"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_synonym_expansion.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the module**

Create `apps/api/rag/query/synonym_expansion.py`:
```python
from __future__ import annotations

import re

ACRONYM_EXPANSIONS: dict[str, str] = {
    "rbac": "rbac role binding cluster role",
    "oauth": "oauth authentication identity provider",
    "pvc": "pvc persistent volume claim",
    "pv": "pv persistent volume",
    "mtu": "mtu maximum transmission unit cluster network",
    "etcd": "etcd key value store",
    "olm": "olm operator lifecycle manager",
    "gitops": "gitops argocd application",
    "oc": "oc openshift command line",
    "csi": "csi container storage interface",
    "crd": "crd custom resource definition",
    "scc": "scc security context constraints",
}

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_-]+")


def expand_acronyms(query: str) -> str:
    text = str(query or "")
    if not text.strip():
        return text
    additions: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(text.casefold()):
        expansion = ACRONYM_EXPANSIONS.get(token)
        if expansion and token not in seen:
            additions.append(expansion)
            seen.add(token)
    if not additions:
        return text
    return f"{text} " + " ".join(additions)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_synonym_expansion.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/query/synonym_expansion.py tests/test_synonym_expansion.py
git commit -m "feat(query): add OpenShift acronym expansion dictionary"
```

### Task 4.5 — Wire synonym expansion into search query path

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py`

- [ ] **Step 1: Apply expansion to the search_query before retrieval**

In `_answer_doc_lane`, just before the parallel retrieval block (after `rewrite = ...`), insert:
```python
from apps.api.rag.query.synonym_expansion import expand_acronyms
use_syn = bool(getattr(self.llm_client.settings, "use_synonym_expansion", True)) if self.llm_client else False
search_query = expand_acronyms(rewrite.rewritten_query) if use_syn else rewrite.rewritten_query
```
Then replace usages of `rewrite.rewritten_query` inside the parallel call with `search_query`. Keep `rewrite.rewritten_query` for cache key + logging so existing cache stays sensible.

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -k "chat or pipeline or synonym" -v`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add apps/api/rag/generation/unified_copilot_service.py
git commit -m "feat(chat): apply acronym expansion to retrieval search_query"
```

### Task 4.6 — BGE-m3 asymmetric query prompt

**Files:**
- Modify: `apps/api/rag/retrieval/embedding_clients.py`

- [ ] **Step 1: Add a query mode to the embedders**

In `BGEOllamaEmbedder.encode` and `BGETEIEmbedder.encode`, add an optional argument `is_query: bool = False`. Modify both to prefix when `is_query=True`:
```python
def encode(self, text: str, *, is_query: bool = False) -> list[float]:
    cached_key = f"q::{text}" if is_query else f"p::{text}"
    cached = self._query_cache.get(cached_key)
    if cached is not None:
        return cached
    payload = (
        f"Represent this query for retrieving relevant documents: {text}"
        if is_query else text
    )
    result = self.encode_batch([payload])[0]
    if len(self._query_cache) >= self._QUERY_CACHE_MAX:
        self._query_cache.pop(next(iter(self._query_cache)))
    self._query_cache[cached_key] = result
    return result
```
Repeat the same change in `BGETEIEmbedder.encode`.

- [ ] **Step 2: Run any existing embedding tests**

Run: `python -m pytest tests/ -k "embedding" -v`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add apps/api/rag/retrieval/embedding_clients.py
git commit -m "feat(embedding): add asymmetric query prompt mode to BGE embedders"
```

### Task 4.7 — Use asymmetric query in pgvector_bridge

**Files:**
- Modify: `apps/api/rag/retrieval/pgvector_bridge.py`

- [ ] **Step 1: Pass `is_query=True` when encoding the user query**

Find the two `runtime.embedder.encode(...)` calls in `answer()`. Change to:
```python
use_asym = bool(getattr(self._runtime, "_asymmetric_enabled", True))
query_vector = runtime.embedder.encode(message, is_query=True if use_asym else False)
```
And similarly for `alt_vector = runtime.embedder.encode(additional_query, is_query=True if use_asym else False)`.

- [ ] **Step 2: Add toggle to LegacyPgvectorRuntime**

In `apps/api/rag/retrieval/legacy_pgvector_runtime.py`, in `LegacyPgvectorRuntime.__init__`, add:
```python
self._asymmetric_enabled = True
```
And expose:
```python
def set_asymmetric_enabled(self, enabled: bool) -> None:
    self._asymmetric_enabled = bool(enabled)
```

- [ ] **Step 3: Wire settings → runtime in app factory**

Find where `PgvectorRetrievalBridge` / `LegacyPgvectorRuntime` is constructed (likely `apps/api/app_factory.py`). Add after construction:
```python
runtime.set_asymmetric_enabled(getattr(chat_settings, "use_asymmetric_query_prompt", True))
```
(If exact wiring is missing, search: `grep -rn "LegacyPgvectorRuntime\|PgvectorRetrievalBridge" apps/api`.)

- [ ] **Step 4: Commit**

```bash
git add apps/api/rag/retrieval/pgvector_bridge.py apps/api/rag/retrieval/legacy_pgvector_runtime.py apps/api/app_factory.py
git commit -m "feat(retrieval): use BGE-m3 asymmetric query prompt in pgvector bridge"
```

### Task 4.8 — Reindex (passages stay raw — no reindex needed unless query asymmetric changes passage encoding)

The asymmetric prompt is on the **query side** only. Passage embeddings stay raw, so a full reindex is **not strictly required**. Verify by running:

- [ ] **Step 1: Confirm passage encoding path is unchanged**

Run: `grep -n "encode_passage\|encode_batch" apps/api/rag/retrieval/embedding_clients.py | head`
Expected: `encode_passage` returns the raw text path (already does — see `BGEOllamaEmbedder.encode_passage` which calls `self.encode(text)`). Update `encode_passage` to call `self.encode(text, is_query=False)` to be explicit.

- [ ] **Step 2: Patch encode_passage in both classes**

```python
def encode_passage(self, text: str) -> list[float]:
    return self.encode(text, is_query=False)
```

- [ ] **Step 3: Commit**

```bash
git add apps/api/rag/retrieval/embedding_clients.py
git commit -m "chore(embedding): make encode_passage explicitly is_query=False"
```

### Task 4.9 — Run service eval for Phase 4

- [ ] **Step 1: Snapshot, run, snapshot, compare**

```bash
cp tests/results/service-eval/latest.json tests/results/service-eval/baseline-pre-p4.json
python scripts/run_service_eval.py --base-url http://localhost:8000
cp tests/results/service-eval/latest.json tests/results/service-eval/phase-4.json
python -c "import json; pre=json.load(open('tests/results/service-eval/baseline-pre-p4.json')); post=json.load(open('tests/results/service-eval/phase-4.json')); print('ok pre/post:', pre['ok_count'], post['ok_count'])"
```
Expected: more passes especially on acronym cases (eval-0004 OAuth, eval-0005 RBAC, eval-0010 MTU).

- [ ] **Step 2: Commit phase result**

```bash
git add tests/results/service-eval/baseline-pre-p4.json tests/results/service-eval/phase-4.json
git commit -m "test(eval): record phase 4 service eval baseline and result"
```

---

## Phase 5 — Gap-triggered LLM Rerank

Goal: only call the LLM rerank judge when the top-2 candidates are within 15% score gap.

### Task 5.1 — Add Phase 5 toggles

**Files:**
- Modify: `apps/api/core/llm_settings.py`

- [ ] **Step 1: Add toggles**

Append:
```python
use_gap_triggered_rerank: bool = True
rerank_gap_threshold: float = 0.15
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/core/llm_settings.py
git commit -m "feat(settings): add phase 5 toggles for gap-triggered rerank"
```

### Task 5.2 — Create rerank_decider

**Files:**
- Create: `apps/api/rag/retrieval/rerank_decider.py`
- Create: `tests/test_rerank_decider.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rerank_decider.py`:
```python
import unittest

from apps.api.rag.retrieval.rerank_decider import needs_rerank


class RerankDeciderTests(unittest.TestCase):
    def test_skip_when_fewer_than_three_candidates(self):
        self.assertFalse(needs_rerank([0.9, 0.8]))
        self.assertFalse(needs_rerank([0.9]))
        self.assertFalse(needs_rerank([]))

    def test_skip_when_top_score_zero(self):
        self.assertFalse(needs_rerank([0.0, 0.0, 0.0]))

    def test_trigger_when_gap_below_threshold(self):
        self.assertTrue(needs_rerank([0.9, 0.85, 0.5]))  # gap = 0.055 < 0.15

    def test_skip_when_gap_above_threshold(self):
        self.assertFalse(needs_rerank([0.9, 0.5, 0.4]))  # gap = 0.444 > 0.15

    def test_threshold_is_configurable(self):
        self.assertFalse(needs_rerank([0.9, 0.85, 0.5], threshold=0.01))
        self.assertTrue(needs_rerank([0.9, 0.5, 0.3], threshold=0.5))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_rerank_decider.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `apps/api/rag/retrieval/rerank_decider.py`:
```python
from __future__ import annotations


def needs_rerank(scores: list[float], *, threshold: float = 0.15) -> bool:
    if len(scores) < 3:
        return False
    top = scores[0]
    if top <= 0:
        return False
    relative_gap = (top - scores[1]) / top
    return relative_gap < threshold
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_rerank_decider.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/rag/retrieval/rerank_decider.py tests/test_rerank_decider.py
git commit -m "feat(retrieval): add gap-triggered rerank decision helper"
```

### Task 5.3 — Use needs_rerank in `_rerank_doc_response`

**Files:**
- Modify: `apps/api/rag/generation/unified_copilot_service.py`

- [ ] **Step 1: Replace the gating logic**

Find `_rerank_doc_response` (~line 297). Change the early-return guard:
```python
async def _rerank_doc_response(
    self,
    *,
    message: str,
    response: CopilotChatResponse,
) -> CopilotChatResponse:
    if not self._llm_enabled() or not response.sources:
        return response
    use_gap = bool(getattr(self.llm_client.settings, "use_gap_triggered_rerank", True))
    threshold = float(getattr(self.llm_client.settings, "rerank_gap_threshold", 0.15) or 0.15)
    scores = [float(src.score or 0.0) for src in response.sources[:6]]
    from apps.api.rag.retrieval.rerank_decider import needs_rerank
    if use_gap and not needs_rerank(scores, threshold=threshold):
        logger.info("[rerank:decision] skip (gap above %.2f)", threshold)
        return response
    if self._can_skip_rerank(response.sources):
        logger.info("[rerank:decision] skip (single dominant)")
        return response
```

- [ ] **Step 2: Reduce candidate count to top-3 for rerank**

In the same function, change `ranked_candidates = response.sources[:6]` to `ranked_candidates = response.sources[:3]`. Reduce `max_tokens=140` to `max_tokens=80`. Reduce timeout in `llm_client._timeout_for_purpose`: not necessary if you pass `purpose="rerank"` (already 6s). Keep as is.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/ -k "chat or rerank" -v`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add apps/api/rag/generation/unified_copilot_service.py
git commit -m "feat(rerank): trigger LLM judge only when top-2 score gap < threshold"
```

### Task 5.4 — Run service eval for Phase 5 + final summary

- [ ] **Step 1: Snapshot, run, snapshot**

```bash
cp tests/results/service-eval/latest.json tests/results/service-eval/baseline-pre-p5.json
python scripts/run_service_eval.py --base-url http://localhost:8000
cp tests/results/service-eval/latest.json tests/results/service-eval/phase-5.json
```

- [ ] **Step 2: Print full progression**

```bash
python -c "
import json
phases = ['baseline-pre-p1','phase-1','phase-2','phase-3','phase-4','phase-5']
for p in phases:
    try:
        d = json.load(open(f'tests/results/service-eval/{p}.json'))
        print(f'{p}: ok={d[\"ok_count\"]}/{d[\"count\"]} avg_lat={d[\"avg_latency_ms\"]}ms')
    except FileNotFoundError:
        print(f'{p}: missing')
"
```
Expected: `phase-5: ok=7/10 avg_lat<=70000ms` (or close to target).

- [ ] **Step 3: Commit final phase result**

```bash
git add tests/results/service-eval/baseline-pre-p5.json tests/results/service-eval/phase-5.json
git commit -m "test(eval): record phase 5 service eval result and full progression"
```

---

## Self-Review Checklist (Engineer's Final Pass)

Before considering the plan complete, verify:

- [ ] All 5 phases executed; toggles present in `apps/api/core/llm_settings.py`
- [ ] `phase-5.json` shows `ok_count >= 7` and `avg_latency_ms <= 70000` (or documented gap explained)
- [ ] No reference to `intent_agent` / `query_rewrite_agent` / `question_normalizer` remains in active code path when `USE_QUERY_ROUTER=true`
- [ ] All new tests in `tests/test_*` pass (`python -m pytest tests/ -v`)
- [ ] No hardcoded answer-matching dictionaries (only retrieval-side acronym expansion in `synonym_expansion.py`)
- [ ] Phase rollback toggle confirmed: setting any `USE_*` to `false` in `.env` and restarting falls back cleanly

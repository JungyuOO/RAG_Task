# RAG Quality & Latency Improvement — Design

- Date: 2026-04-17
- Branch: dev-ver2
- Status: Approved (brainstorming)
- Owner: JungyuOO

## 1. Problem Statement

`tests/results/service-eval/latest.json` 평가 결과:

- pass rate: **10%** (1/10)
- avg latency: **68.9초/케이스** (per-turn ≈ 20–28초)
- failure 패턴 핵심: source_match=true / section_match=false (5건), citation_alignment=false (대부분), mixed lane fallback에서 사용자 질문이 답변 본문에 박힘 (eval-0001/0002)

진단 결과 chunk_size 자체는 핵심 원인이 아니며 (`MAX_MERGED_TOKENS=440` 적정), 실제 원인은 retrieval 정밀도, synthesis 프롬프트와 평가 기준의 자가 충돌, LLM 호출 직렬 다단계로 인한 latency 누적.

## 2. Goals / Non-goals

### Goals
- pass rate ≥ **70%** (7/10), per-turn latency ≤ **7–10초**
- 인프라 추가 없이 (기존 TEI/CLLM/pgvector만 사용) 달성
- 각 변경을 독립 토글하여 phase별로 효과 측정 가능

### Non-goals
- 새 reranker 서비스 (cross-encoder) 도입 — 제약
- 답변 매칭형 하드코딩 (예: 질문 → 정답 매핑 사전) — 사용자 명시 금지
- 청크 크기/구조 전면 재설계 (현 chunker는 충분히 적정)
- LLM 모델 교체

## 3. Constraints

- 기존 인프라 그대로: BGE-m3 via TEI (dense embedding only), CLLM (OpenAI-compatible), pgvector
- BGE-m3는 embedding 모델 — cross-encoder rerank 불가
- 새 외부 서비스/모델 불가
- 도메인 dictionary는 허용하나 답변 매칭이 아닌 retrieval 보조 용도여야 함

## 4. Architecture Overview

### 현행
```
question_normalize (LLM) → intent_classify (LLM) → query_rewrite (LLM)
  → sparse retrieval ─┐
  → dense retrieval ──┴→ merge → rerank (LLM, 6s timeout)
  → synthesis (LLM, 14s timeout, post-attach citations)
```
LLM 5회 직렬, 턴당 ≈ 20–28초.

### 신규
```
                   ┌─ sparse (char-ngram BM25) ─┐
QueryRouter (LLM) ─┤                             ├→ RRF merge
    1회 호출         └─ dense (BGE-m3 asym) ──────┘
                                                   │
                                  score gap < 0.15? │
                                       │           │
                                  yes  ↓     no    │
                              LLM judge       skip │
                                       └────┬──────┘
                                            ↓
                                  MMR diversity select
                                            ↓
                                Synthesis (LLM, native [N])
```
LLM 2회 + 조건부 1회, 턴당 ≈ 7–10초.

## 5. Components

### 5.1 QueryRouter (신규, agents 3개 통합)

단일 LLM 호출로 lane 분류, search_query (한→영 번역 포함), live_query, source 상속을 동시 결정.

- 입력: `user_message`, `recent_turns`, `has_connection`, `last_lane`, `last_doc_sources`
- 출력 (JSON 강제):
  ```json
  {
    "lane": "doc|live|mixed|needs_connection",
    "search_query": "string (English-leaning, retrieval-friendly)",
    "live_query": "string (empty unless live/mixed)",
    "inherit_sources": true|false,
    "reasoning_brief": "string"
  }
  ```
- timeout 2.5s, 1 attempt, temperature 0.0, max_tokens 220
- Fallback rule (LLM 실패 시):
  - `lane = (last_lane.startswith("doc") and looks_followup(msg)) ? last_lane : "doc"`
  - `search_query = msg.strip()`
  - `inherit_sources = looks_followup(msg) and last_lane.startswith("doc")`
- 기존 `QuestionNormalizer`/`IntentAgent`/`QueryRewriteAgent`는 deprecated 표시 후 보존 (토글 롤백 가능)

### 5.2 Retrieval Signal Upgrade

#### 5.2.1 BGE-m3 asymmetric query prompt
- query 인코딩 시 instruction prefix 부착:
  - `query_text = "Represent this query for retrieving relevant documents: " + raw_query`
- passage 인덱스는 raw 유지하되 **재인덱싱 1회 필요**

#### 5.2.2 Korean character n-gram sparse
- 한글 토큰: char bigram + trigram + 어절 통합 (예: "확인할" → ["확인", "인할", "확인할"])
- 영문/숫자: unigram 그대로
- BM25는 통합 vocabulary 위에서 동작

#### 5.2.3 Acronym/synonym dictionary (룰 기반, 약 10개)
```
rbac → "rbac role binding cluster role"
oauth → "oauth authentication identity provider"
pvc → "pvc persistent volume claim"
mtu → "mtu maximum transmission unit cluster network"
etcd → "etcd key value store"
olm → "olm operator lifecycle manager"
gitops → "gitops argocd application"
oc → "oc openshift command line"
csi → "csi container storage interface"
crd → "crd custom resource definition"
```
- search_query에 acronym이 단독일 때만 expansion **추가** (대체 X)
- domain dictionary (검색 보조), 답변 매칭 아님

#### 5.2.4 RRF fusion
```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)
fused = rrf(dense_rank) + rrf(sparse_rank)
```
임의 가중치(현재 1.8 vs 0.7) 폐기.

#### 5.2.5 MMR diversity selection
- RRF top-8 후보 → MMR로 최종 `answer_source_budget(query)` 개 선택 (현재 budget 함수는 2~3 반환)
- `score = λ * relevance - (1-λ) * max_similarity_to_selected`, λ = 0.7
- 같은 section 중복 방지

### 5.3 Reranker (gap-triggered LLM judge)

- Trigger:
  ```python
  def needs_rerank(scores):
      if len(scores) < 3 or scores[0] <= 0:
          return False
      return (scores[0] - scores[1]) / scores[0] < 0.15
  ```
- 호출 시: top-3 후보만, JSON 한 줄 출력, max_tokens 80, timeout 3s, 1 attempt
- 실패/timeout → silent fallback to RRF order

### 5.4 Synthesis + Citation

- 프롬프트가 `[N]` citation을 직접 emit하도록 변경:
  ```
  - 각 단락 끝에 사용한 근거 번호를 [1], [2] 형태로 반드시 표기
  - 한 단락이 여러 근거를 쓰면 [1][2]처럼 연속 표기
  ```
- max_tokens 1800 → **600**, temperature 0.1, timeout 12s, 2 attempts
- post-validation: 범위 밖 `[N]` marker만 제거, 본문 보존
- 단락 citation 없으면 → `_attach_paragraph_citations` 1회 fallback
- citation_validator: 검증 실패 시 답변 본문 보존 (현재는 비우는 케이스 있음)

### 5.5 Mixed Lane Fallback Bug Fix

`unified_copilot_service.py:513`의 사용자 질문이 답변 본문에 prefix되는 버그 제거:
```python
# before
parts = [f"문서 근거와 현재 클러스터 기준 정보를 함께 정리하면 다음과 같습니다: {message}"]
# after
parts = ["문서 근거와 현재 클러스터 상태를 정리합니다."]
```

## 6. Data Flow

1. `chat.py` route 진입 → `UnifiedCopilotService.answer()` 호출
2. `QueryRouter.decide()` — LLM 1회 (또는 fallback rule), 결과로 lane/search_query/live_query/inherit_sources 결정
3. lane이 doc/mixed면 `asyncio.gather(sparse_search, dense_search)` 병렬 실행
4. `hybrid_fusion.rrf_merge(sparse, dense)` → top-8 후보
5. `rerank_decider.needs_rerank(scores)` → true면 LLM judge 1회로 top-3 reorder, false면 skip
6. `mmr_select(candidates, λ=0.7)` → `answer_source_budget(query)` 개 (보통 2~3)
7. `synthesize(search_query, original_message, sources)` — LLM stream, native `[N]` emit
8. `finalize(answer, sources)` — citation 범위 검증/보존
9. lane이 mixed면 live OCP 응답 합성 (synthesis 전 또는 병렬)

## 7. Caching, Parallelization, Observability

- Retrieval 병렬화: `asyncio.gather(sparse, dense, return_exceptions=True)`
- Router cache: in-memory LRU 256 entries, 키 = `(user_message, last_lane, hash(last_doc_sources))`, TTL 10분
- Embedding query cache: 기존 `BGETEIEmbedder._query_cache` 활용
- Response cache: `ChatResponseCache` 유지, 키에 `router_version` 추가 (배포 시 자동 invalidation)
- 단계별 timing 로그 보강: `[router]`, `[retrieve:rrf]`, `[rerank:decision]`, `[citation:emit]`, `[citation:fallback]`

## 8. Error Handling

- 모든 LLM 호출은 try/except + timeout, 실패 시 silent fallback (현재 패턴 유지)
- Router LLM fail → rule-based fallback
- Retrieval 한 쪽 실패 → 나머지로 진행
- Rerank fail → RRF order 사용
- Synthesis fail → 기존 retrieval 기반 답변 (raw chunk concat) + post-attach citation 시도
- Citation validation fail → 답변 본문 보존 (marker만 제거)

## 9. Settings Toggles

`apps/api/core/llm_settings.py`에 추가:
```
USE_QUERY_ROUTER=true        # false면 legacy 3-agent 경로
USE_RRF_FUSION=true
USE_CHAR_NGRAM_SPARSE=true
USE_GAP_TRIGGERED_RERANK=true
USE_NATIVE_CITATION_PROMPT=true
RERANK_GAP_THRESHOLD=0.15
```

## 10. Phased Rollout

| Phase | 변경 | 토글 | 핵심 검증 메트릭 |
|---|---|---|---|
| **P1** | Synthesis 프롬프트 native `[N]` + mixed fallback bug fix + citation_validator 보존 | `USE_NATIVE_CITATION_PROMPT` | citation_alignment / pass rate |
| **P2** | QueryRouter 통합 + retrieval 병렬화 | `USE_QUERY_ROUTER` | per-turn latency |
| **P3** | RRF fusion + MMR diversity | `USE_RRF_FUSION` | section_match |
| **P4** | Char n-gram sparse + acronym dictionary + asymmetric query prompt + 인덱스 재빌드 | `USE_CHAR_NGRAM_SPARSE` | acronym/follow-up hit |
| **P5** | Gap-triggered LLM rerank | `USE_GAP_TRIGGERED_RERANK` | tight-gap 케이스 정확도 |

P1을 첫 번째로 두는 이유: synthesis 프롬프트 한 줄 변경만으로 citation_alignment fail이 가장 많이 풀려야 한다는 가설을 가장 빨리 검증할 수 있음.

각 phase 진행 기준:
- 진행 전 `latest.json`을 `baseline-pre-{phase}.json`으로 백업
- phase 후 비교: 좋아진 케이스 ≥ 4 AND 나빠진 케이스 ≤ 1일 때만 다음 phase 진행
- 그렇지 않으면 토글로 롤백 후 root cause 분석

## 11. Testing

### Unit
- `tests/unit/test_query_router.py` — JSON 파싱, fallback rule, follow-up 처리 (mock LLM)
- `tests/unit/test_hybrid_fusion.py` — RRF 점수, MMR diversity
- `tests/unit/test_char_ngram_tokenizer.py` — bigram/trigram, acronym expansion
- `tests/unit/test_rerank_decider.py` — gap 계산, threshold edge
- `tests/unit/test_citation_grounding.py` — native `[N]` 보존, fallback attach

### Integration
- `tests/integration/test_pipeline_flow.py` — router → parallel retrieval → fusion → synthesis end-to-end (mock LLM/embedder)
- `tests/integration/test_lane_routing.py` — 한국어 follow-up 케이스 5개

### Eval-driven
- 매 phase 끝에 `python scripts/run_service_eval.py` 실행
- 결과 `tests/results/service-eval/phase-{N}.json` 저장 → diff 분석
- 최종 목표: pass rate ≥ 7/10, avg latency ≤ 70초/케이스

### Eval dataset 보강 (선택, 별도 PR)
- acronym single-turn 5케이스 (RBAC, OAuth, OLM, MTU, etcd)
- deep-followup 5케이스
- Korean-only 5케이스
- 총 25 케이스로 확장 → phase별 변화 추적 신뢰도 향상

## 12. Files Affected (예상)

신규:
- `apps/api/rag/query/query_router.py`
- `apps/api/rag/retrieval/hybrid_fusion.py` (RRF + MMR)
- `apps/api/rag/retrieval/rerank_decider.py`
- `apps/api/rag/retrieval/korean_tokenizer.py` (char n-gram)
- `apps/api/rag/query/synonym_expansion.py` (도메인 dictionary)

수정:
- `apps/api/rag/generation/unified_copilot_service.py` — router 호출, 병렬 retrieval, native citation prompt, mixed fallback fix
- `apps/api/rag/generation/citation_grounding.py` — validation 실패 시 답변 보존
- `apps/api/rag/retrieval/embedding_clients.py` — query asymmetric prompt
- `apps/api/rag/retrieval/document_retriever.py` — char n-gram tokenize, RRF, MMR
- `apps/api/rag/retrieval/pgvector_bridge.py` — query asymmetric prompt
- `apps/api/rag/query/query_features.py` — synonym expansion helper
- `apps/api/core/llm_settings.py` — phase 토글 settings

Deprecated (보존):
- `apps/api/rag/query/question_normalizer.py`
- `apps/api/rag/query/intent_agent.py`
- `apps/api/rag/query/query_rewrite_agent.py`

운영:
- 인덱스 재빌드 1회 (Phase 4 진입 시)

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Router LLM이 lane 분류를 잘못해서 doc 질문이 live로 가는 등 라우팅 회귀 | rule-based fallback + integration test 5케이스 + 프로덕션 토글 롤백 |
| Native citation prompt가 LLM마다 instruction-following 편차 | post-validation은 보존(removal-only), fallback attach 유지 |
| Char n-gram이 sparse vocabulary 폭증으로 인덱스/메모리 부담 | bigram+trigram 한정, doc_frequency cutoff 적용 |
| 인덱스 재빌드 중 서비스 중단 | 별도 인덱스 키로 빌드 후 atomic swap, 또는 점검 윈도우 |
| Acronym dictionary가 노이즈를 만드는 케이스 | search_query에 단독 acronym일 때만 발동, expansion은 추가만 (대체 X) |
| Phase별 회귀 감지 누락 | regression guard (좋아진 ≥4 AND 나빠진 ≤1) 위반 시 자동 롤백 정책 |

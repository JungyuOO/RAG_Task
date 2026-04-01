# RAG Multiturn Hardening v1.0.3 Plan

작성일: 2026-04-01

기준 문서:

- `docs/specs/v1.0.3/specs.md`
- `tests/results/multiturn_cases.v103c.json`
- `tests/results/long_multiturn_cases.v103e.json`
- `tests/results/cache_behavior_cases.v103d.json`
- `tests/results/extracted_markdown_coverage_cases.v1.json`
- `tests/results/corpus_general_cases.v1.json`
- `tests/results/corpus_structured_cases.v1.json`

## 목표

v1.0.3의 실행 목표는 남은 장기 멀티턴 실패를 generic state 기반으로 줄이고, cache를 완전 통과 수준으로 올리는 것이다.

핵심 방향:

- 키워드 하드코딩 확장 금지
- answer-shape / example-anchor / procedure-pointer state 일반화
- follow-up retrieval anchor 강화
- clarification 개입 축소

## 진행 현황

최신 기준:

- `multiturn_cases.v103c.json`: 7 / 10
- `long_multiturn_cases.v103e.json`: 7 / 10
- `cache_behavior_cases.v103d.json`: 4 / 5
- `extracted_markdown_coverage_cases.v1.json`: 4 / 8
- `corpus_general_cases.v1.json`: 생성 및 실행 완료
- `corpus_structured_cases.v1.json`: 생성 및 실행 완료
- `corpus_session_cases.v1.json`: 실행 중단, 재실행 필요

이번 라운드까지 반영된 작업:

- [x] 초기 절차 설명 응답은 public payload에서 `procedure_state`를 숨기고 내부 메모리에만 저장
- [x] 절차 상태가 없는데 `2단계` 같은 질문이 오면 모호 응답으로 안전하게 정리
- [x] topic-local example/source page anchor를 retrieval 확장에 반영
- [x] explicit resource follow-up에서 judge 개입을 일부 축소
- [x] code/YAML 예시 선택에서 explicit `kind:` 일치 우선
- [x] cache key에서 불필요한 topic rebinding 영향을 제거
- [x] LLM 실패 fallback을 raw context dump보다 더 압축된 grounded summary로 변경
- [x] resource marker를 substring이 아닌 경계 기반으로 해석하여 `PVC -> PV` 오탐 제거
- [x] selected source 기반 fallback code candidate search 추가
- [x] self-contained explicit query에 대해 canonical cache query 도입
- [x] `last_example_anchor` 저장 및 field name 추출
- [x] non-code route에서 fenced code block 제거
- [x] multi-doc YAML block을 requested `kind` 기준으로 분할 추출
- [x] `data/extracted_markdown` 실제 문서 기반 coverage manual-eval dataset 추가
- [x] corpus general dataset 추가
- [x] corpus structured dataset 추가
- [x] corpus session dataset 초안 추가

현재 남은 주요 실패:

- [ ] `storage-five-turn-thread`
  - `그 yaml에서 selector는 왜 쓰는거야?`
- [ ] `networking-five-turn-thread`
  - `그럼 Service yaml 예시 보여줘`
  - `그 중 edge 방식만 다시 설명해줘`
- [ ] `cache-same-topic-different-intent`
  - 동일 설명 질의 재호출의 `observed_cached` 기대 불일치 유지
- [ ] `extracted_markdown_coverage_cases`
  - `argocd-application-yaml`
  - `tekton-task-yaml`
  - `rbac-scc-table`
  - `deployment-strategy-compare`
- [ ] `corpus_session_cases`
  - 전체 재실행 및 baseline 확보
- [ ] `corpus_general/structured`
  - 결과 요약 정리 및 실패 유형 묶음화

## 실패 묶음

현재 남은 실패는 크게 세 묶음이다.

### A. Procedure flow 잔여 실패

- 해당 묶음은 현재 핵심 실패에서 제외됨

문제:

- 초기 목표 대비 대부분 해소됨. 후속 라운드는 procedure보다 example/field 흐름에 집중

### B. Long example follow-up 잔여 실패

- `그 yaml에서 selector는 왜 쓰는거야?`
- `그럼 Service yaml 예시 보여줘`
- `그 중 edge 방식만 다시 설명해줘`

문제:

- 직전 example anchor는 저장되지만 field-level route override가 아직 약함
- Service-only YAML 정밀도는 개선됐지만 block-level 분할/정제는 추가 필요
- `edge` 설명은 retrieval route는 맞지만 fallback answer 품질이 아직 약함

### C. Cache 잔여 실패

- `ConfigMap을 설명해줘` 반복 시 expectation mismatch

문제:

- semantic request identity와 observed_cached 기대가 완전히 맞물리지 않음

## 작업 원칙

### 1. 하드코딩 금지

다음 방식은 사용하지 않는다.

- 리소스 이름별 예외 분기 추가
- 특정 테스트 문장 맞춤 처리
- 특정 문서 파일명/페이지 의존 처리

### 2. 일반화 우선

모든 수정은 아래 일반화 축 중 하나여야 한다.

- state schema 확장
- anchor selection 일반화
- retrieval query composition 일반화
- answer route decision 일반화
- judge/clarification gating 일반화

### 3. 작은 루프 유지

각 phase마다:

1. 실패 케이스 한 묶음 선택
2. 관련 단위 테스트 추가 또는 수정
3. 코드 수정
4. 해당 manual eval 재측정

## Phase 1. Answer Shape State 정교화

### 목표

설명, 절차, 표, 코드 예시를 단일 follow-up state가 아니라 분리된 상태로 다룬다.

### 작업

- [x] `app/rag/memory.py`
  - route / intent / response shape / format / resource focus에 해당하는 topic-local state 확장
- [x] `app/rag/pipeline.py`
  - final payload 저장 시 `query_interpretation`, `answer_route`를 일관되게 기록
  - 기존 `procedure_state`만 따로 보는 구조를 축소
- [ ] 남은 작업
  - `last_answer_shape_state`를 단일 묶음 객체로 재정리
  - explanation / extractive_table / extractive_code / grounded_generation 전환을 한 객체에서 추적

### 기대 효과

- 설명 확대 질문이 code route로 튀는 현상 감소
- 절차 navigation과 일반 document follow-up 충돌 감소

## Phase 2. Procedure Pointer / Topic Pointer 분리

### 목표

절차형 흐름과 topic 흐름을 따로 관리한다.

### 작업

- [x] `app/rag/pipeline.py`
  - `다음 단계`, `2단계`, `1단계부터`는 절차 상태가 있을 때만 procedure pointer 기준
  - 절차 상태 없이 step navigation이 오면 모호 응답으로 짧게 정리
  - `다시 RBAC 차이점만`은 procedure shortcut보다 topic switch를 우선
- [ ] `app/rag/memory.py`
  - `procedure_state`에 `source_route`, `source_chunk_ids`, `source_section_path` 추가
  - topic state에 `active_procedure_id` 또는 동등 개념 추가
- [ ] 남은 작업
  - procedure pointer와 topic pointer를 완전히 분리
  - 부분집합 설명 확대 요청이 procedure/code route로 오염되지 않도록 일반화

### 기대 효과

- `ConfigMap 생성 절차 -> 2단계` 흐름 안정화
- `RBAC 주제 전환`과 절차 follow-up 충돌 감소

## Phase 3. Example Anchor Memory 도입

### 목표

yaml/code 예시 후속 질문에서 직전 anchor를 잃지 않게 한다.

### 작업

- [x] `app/rag/memory.py`
  - `last_example_source_pages`, `last_grounded_section_paths`, `last_code_resource_kind` 저장
- [x] `app/rag/pipeline.py`
  - topic anchor 기반 candidate 확장 추가
  - extractive code/table 응답 후 anchor에 해당하는 state 저장
- [x] `app/rag/pipeline.py`
  - selected source 기반 fallback code candidate search 추가
- [ ] `app/rag/memory.py`
  - `last_example_anchor`를 명시 객체로 승격
  - 최소 필드:
    - `resource_kind`
    - `context_ids`
    - `source_path`
    - `page_numbers`
    - `section_paths`
    - `fields`
- [ ] `app/rag/pipeline.py`
  - field follow-up이면 anchor-first retrieval를 더 강하게 적용
- [ ] `app/services/answer_service.py`
  - code answer에서 field 후보 추출 가능하게 보조 메타데이터 정리
- [ ] 남은 작업
  - field follow-up에서 anchor field match가 있으면 clarification보다 설명 route 우선
  - Service YAML처럼 multi-manifest block에서 resource-only snippet만 더 정밀하게 남기기

### 기대 효과

- `예시 yaml도 보여줘`
- `그 yaml에서 selector는 왜 쓰는거야?`
- `Service yaml 예시`

같은 흐름이 유지된다.

## Phase 4. Retrieval Query Composition 일반화

### 목표

short follow-up도 prior state를 조합해 retrieval이 가능하게 만든다.

### 작업

- [x] `app/services/query_interpreter.py`
  - current-turn explicit signal과 inherited resource state를 조합
- [x] `app/services/query_interpreter.py`
  - resource marker boundary matching으로 substring 오탐 제거
- [x] `app/rag/pipeline.py`
  - retrieval query에 resource alias와 prior state 일부를 조합
  - topic anchor 기반 candidate expansion 도입
- [ ] `app/rag/pipeline.py`
  - request frame 객체를 별도 함수/구조로 명시화
  - 입력:
    - current message
    - current explicit resource
    - previous focus entity
    - previous example anchor
    - requested format
  - 출력:
    - retrieval query
    - rerank hint metadata
- [ ] `app/services/retrieval_service.py`
  - anchor-first candidate selection 또는 section-local boost 추가
- [ ] 남은 작업
  - `Service yaml 예시`
  - `selector field follow-up`
  - `edge 방식`
  같은 short follow-up recall을 더 끌어올릴 request-frame 일반화

### 기대 효과

- `Service yaml 예시`
- `PVC만 더 자세히`
- `edge 방식`

같은 short follow-up recall 개선

## Phase 5. Judge / Clarification Gating 축소

### 목표

이미 충분한 prior context가 있는 long follow-up에서는 judge/clarification이 과하게 개입하지 않게 한다.

### 작업

- [x] `app/rag/pipeline.py`
  - judge 실행 조건에 prior resource / top_score를 반영
- [x] `app/services/turn_policy_service.py`
  - explicit resource code follow-up에서 clarification 회피
- [ ] 남은 작업
  - prior anchor / prior focus / prior example shape까지 judge gating에 반영
  - long follow-up에서는 clarification보다 anchor-based retrieval을 우선

### 기대 효과

- `PVC만 더 자세히`가 불필요한 clarification으로 빠지는 문제 감소
- `Service yaml 예시`가 general fallback으로 가는 문제 감소

## Phase 6. Cache Finalization

### 목표

cache_behavior 세트를 완전 통과로 만든다.

### 작업

- [x] `app/rag/pipeline.py`
  - semantic request identity에서 topic rebinding 영향을 축소
  - `context_ids` 정렬로 cache key 안정화
- [x] `tests/test_pipeline_unit.py`
  - repeated question
  - same topic different intent
  - same response route same context replay
- [ ] 남은 작업
  - 실제 앱 환경에서 동일 설명 질의 재호출의 `observed_cached` 기대를 맞추는 정책 정리
  - no-result / clarification / grounded_generation / extractive_code 간 캐시 정책 최종 일관화

### 기대 효과

- `cache_behavior_cases`: 5 / 5

## Phase 7. 검증 순서

매 phase 후 아래 순서로 검증한다.

1. `tests.test_query_interpreter`
2. `tests.test_turn_policy`
3. `tests.test_turn_context_resolver`
4. `tests.test_pipeline_unit`
5. `tests.test_multiturn` / `tests.test_memory` 가능 시
6. `tests/run_manual_eval.py --dataset tests/datasets/cache_behavior_cases.json`
7. `tests/run_manual_eval.py --dataset tests/datasets/multiturn_cases.json`
8. `tests/run_manual_eval.py --dataset tests/datasets/long_multiturn_cases.json`
9. `tests/run_manual_eval.py --dataset tests/datasets/extracted_markdown_coverage_cases.json`
10. `tests/run_manual_eval.py --dataset tests/datasets/corpus_general_cases.json`
11. `tests/run_manual_eval.py --dataset tests/datasets/corpus_structured_cases.json`
12. `tests/run_manual_eval.py --dataset tests/datasets/corpus_session_cases.json`

## 우선순위

### P0

- [x] Phase 1
- [~] Phase 2
- [~] Phase 3

### P1

- [~] Phase 4
- [~] Phase 5

### P2

- [~] Phase 6
- [ ] Phase 7 반복

## 완료 기준

다음을 만족하면 v1.0.3 완료로 본다.

- `multiturn_cases` 9 / 10 이상
- `long_multiturn_cases` 8 / 10 이상
- `cache_behavior_cases` 5 / 5
- 하드코딩 기반 해결 없이 generic state/anchor 기반으로 동작
- 기존 grounding / out-of-domain / language guard 품질 유지

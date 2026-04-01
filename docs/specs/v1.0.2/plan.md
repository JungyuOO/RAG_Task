# RAG Multiturn Hardening v1.0.2 Plan

작성일: 2026-04-01

기준 문서:

- `docs/specs/v1.0.2/specs.md`
- `docs/specs/v1.0.2/2026-04-01-rag-multiturn-improvement-analysis.md`

## 목표

v1.0.2 작업의 목표는 현재 강한 grounded retrieval과 guardrail 품질은 유지하면서, 멀티턴 실패의 주 원인인 state 오염, follow-up 해석 충돌, code extraction precision 부족을 단계적으로 제거하는 것이다.

## 작업 원칙

모든 작업은 아래 순서를 따른다.

1. 테스트 또는 수동 평가 기준선 확인
2. 작은 범위의 상태/정책 수정
3. 회귀 검증

추가 원칙:

- 새 프레임워크를 도입하지 않는다.
- 현재 직접 구현 구조를 최대한 재사용한다.
- 멀티턴 개선 때문에 out-of-domain / language guard가 약해지면 안 된다.
- 캐시는 키 설계보다 선행 answer quality를 먼저 안정화한다.

## 우선순위

### P0. 즉시 처리

- 절차형 상태 추출 축소
- explicit resource override 도입
- follow-up intent 분리
- code example hard filter 강화

### P1. 바로 뒤따라야 하는 작업

- topic-local intent/format/code state 추가
- field follow-up anchor state 도입
- abbreviation retrieval expansion 보강
- route-aware fallback 정리

### P2. 문서/검증 정리

- README와 실제 구현 정렬
- manual eval 리포트 해석성 강화
- 실패 패턴 로그 정리

## Phase 1. Procedure State 오염 제거

### 목표

비교/설명 응답이 잘못 절차형 상태로 저장되는 문제를 먼저 제거한다.

### 작업

- `app/rag/pipeline.py`
  - `_extract_procedure_state()`를 answer text 패턴 기반에서 intent/route 기반으로 축소
  - 숫자 리스트만으로는 `procedure_state`를 만들지 않도록 변경
  - LLM fallback 경로에서 procedure state가 잘못 생성되지 않도록 방지

- `app/rag/memory.py`
  - assistant metadata의 `procedure_state` 반영 조건을 더 엄격히 적용

### 검증

- `PV, PVC, StorageClass 차이 설명`
- `ConfigMap과 Secret 차이 설명`
- `Service, Ingress, Route 차이 설명`

위 시나리오가 `procedure_state_followup`가 아니라 `grounded_generation`으로 남아야 한다.

## Phase 2. Follow-up 해석 우선순위 재정렬

### 목표

현재 턴 explicit signal을 세션 ambiguity보다 우선시하도록 바꾼다.

### 작업

- `app/services/query_interpreter.py`
  - explicit resource, action, format signal을 더 강하게 추출
  - `resource_focus_followup`, `format_switch_followup`, `field_followup` 분리를 위한 최소 신호 정의

- `app/services/turn_context_resolver.py`
  - candidate topic scoring은 유지하되 explicit resource가 있는 경우 resolution을 직접 강화
  - ambiguity prompt는 truly ambiguous case에만 남김

- `app/services/turn_policy_service.py`
  - clarification 조건 축소
  - explicit resource + code request 조합이면 clarification보다 direct routing 우선

### 검증

- `그럼 Service yaml 예시 보여줘`
- `그중 PVC만 더 자세히 설명해줘`
- `StorageClass 예시로도 바꿔줘`
- `Ingress는 어떤 점이 달라?`

## Phase 3. Topic State를 Intent State까지 확장

### 목표

토픽 유지뿐 아니라 직전 응답 의도와 형태를 같이 이어받게 만든다.

### 작업

- `app/rag/memory.py`
  - `topic_state` 또는 topic summary에 아래 필드 추가
    - `last_explicit_resource`
    - `last_explicit_resources`
    - `last_intent`
    - `last_response_shape`
    - `last_answer_route`
    - `last_format_constraints`
    - `last_code_resource_kind`
    - `last_grounded_chunk_ids`
    - `last_example_source_pages`

- `app/rag/pipeline.py`
  - retrieval 직후와 final answer 직후에 위 상태를 일관되게 저장
  - follow-up 해석 시 topic-local state를 우선 재사용

### 검증

- `설명 -> yaml 예시 -> yaml 필드 질문`
- `비교 -> 특정 리소스만 drill-down`
- `같은 토픽 내 리소스 전환`

## Phase 4. Code / YAML Precision 강화

### 목표

요청 리소스와 직접 관련된 코드만 우선 선택되도록 만든다.

### 작업

- `app/rag/pipeline.py`
  - `_select_code_example_context_items()`에서 explicit resource kind가 있으면 hard filter 우선 적용
  - positive resource match가 있는 경우 다른 kind는 과감히 제외
  - 직전 code block anchor가 있으면 field follow-up은 anchor 중심으로 재검색

- `app/services/answer_service.py`
  - extractive code answer 구성 시 unrelated snippet mixing을 줄임
  - 같은 문서/같은 페이지라도 resource kind mismatch면 배제 가능하도록 metadata 활용

### 검증

- `ConfigMap YAML 예시`에서 `kind: Pod` 혼입 제거
- `Route 예시`에서 Route 관련 manifest 우선
- `PVC YAML 예시`에서 PVC 자체 예시 우선

## Phase 5. Retrieval Recall 보강

### 목표

후속 문맥 없는 첫 질의 abbreviation 실패를 줄인다.

### 작업

- `app/services/query_interpreter.py`
  - alias/abbreviation 사전을 retrieval-friendly 형태로 정리

- `app/rag/pipeline.py`
  - rewrite query 또는 retrieval query에 alias expansion 주입

- `app/rag/retrieval.py`
  - heading/keyword overlap이 abbreviation 확장 결과를 반영하도록 정렬

### 검증

- `PV와 PVC 차이`
- `RBAC 설명`
- `ArgoCD가 뭐야`
- `Tekton이 뭐야`

## Phase 6. Route-Aware Fallback 정리

### 목표

LLM 실패 시에도 answer shape가 유지되도록 한다.

### 작업

- `app/rag/pipeline.py`
  - grounded generation 실패 시 deterministic summary fallback
  - code route 실패 시 extractive fallback 우선
  - fallback answer가 후속 state를 오염시키지 않도록 metadata 정리

### 검증

- LLM 실패 상황에서도
  - 비교 응답은 비교 형태 유지
  - code 응답은 code 형태 유지
  - procedure state는 필요한 경우에만 생성

## Phase 7. 회귀 검증 강화

### 목표

개선이 실제 평가 품질로 이어지는지 빠르게 확인할 수 있게 한다.

### 작업

- 단위 테스트 보강
  - `tests/test_pipeline_unit.py`
  - `tests/test_multiturn.py`
  - `tests/test_turn_context_resolver.py`
  - `tests/test_turn_policy.py`
  - `tests/test_cache.py`

- 수동 평가 우선 세트
  - `tests/datasets/multiturn_cases.json`
  - `tests/datasets/long_multiturn_cases.json`
  - `tests/datasets/cache_behavior_cases.json`

### 필수 확인 항목

- route mismatch 감소
- forbidden term 혼입 감소
- cached 여부는 유지하면서 answer quality도 개선
- grounding/guardrail 회귀 없음

## 추천 실행 순서

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7

이 순서를 권장하는 이유는 현재 가장 큰 손실이 retrieval 알고리즘 자체보다 state 해석 충돌이기 때문이다. 먼저 route와 state를 바로잡아야 이후 retrieval recall 개선과 캐시 개선 효과가 제대로 드러난다.

## 완료 기준

다음 조건이 만족되면 v1.0.2 계획은 완료된 것으로 본다.

- 비교/설명 응답이 더 이상 잘못 절차형 상태로 저장되지 않는다.
- explicit resource follow-up이 clarification에 과도하게 빠지지 않는다.
- YAML/code 응답에서 무관한 리소스 코드 혼입이 줄어든다.
- `multiturn_cases`, `long_multiturn_cases`, `cache_behavior_cases`가 기준선보다 유의미하게 개선된다.
- `document_grounding`, `out_of_domain`, `language_guard`는 유지된다.

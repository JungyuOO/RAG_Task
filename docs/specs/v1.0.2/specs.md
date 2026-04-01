# RAG Multiturn Hardening v1.0.2 Specs

작성일: 2026-04-01

기준 문서:

- `README.md`
- `docss/specs/v1.0.2/2026-04-01-rag-multiturn-improvement-analysis.md`
- `docs/specs/v1.0.1/2026-03-31-rag-quality-hardening-specs.md`

## 목적

v1.0.2의 목적은 현재 리포지토리의 강점인 문서 grounded retrieval, 차단 정책, 직접 구현 구조는 유지하면서, 과제 핵심 평가 항목인 멀티턴 대응과 응답 정밀도를 실제 검증 통과 수준까지 끌어올리는 것이다.

이번 스펙은 특히 다음 문제를 해결하는 데 집중한다.

- 비교/설명 답변이 잘못 `procedure_state`로 저장되는 문제
- 후속 질문에서 주제는 이어지지만 의도, 리소스, 응답 포맷이 끊기는 문제
- 명시 리소스가 있는 follow-up도 과도하게 clarification으로 빠지는 문제
- YAML/코드 예시 요청에서 관련 없는 리소스 코드가 섞이는 문제
- abbreviation 기반 1턴 질의가 recall 부족으로 실패하는 문제

## 과제 요구사항 준수 원칙

이번 개선은 아래 제약을 유지해야 한다.

- 사용자 -> RAG -> LLM 전체 파이프라인은 현재 코드베이스 안에서 직접 구현 구조를 유지한다.
- 오픈소스 RAG 프레임워크를 새로 도입하지 않는다.
- LLM은 과제에서 지정한 엔드포인트/모델 제약을 유지한다.
- SSE 스트리밍 동작은 유지한다.
- 세션 단위 메모리 구조는 유지하되 더 정교하게 만든다.
- Vector index, retrieval, rerank, cache는 직접 구현 구조를 유지한다.
- 기존 out-of-domain 차단과 language guard 성능은 약화시키지 않는다.

## 현재 기준선

저장된 수동 평가 결과 기준:

- `document_grounding_cases.json`: 6/6 통과
- `out_of_domain_cases.json`: 4/4 통과
- `language_guard_cases.json`: 5/5 통과
- `multiturn_cases.json`: 10턴 중 1턴 통과
- `long_multiturn_cases.json`: 10턴 중 1턴 통과
- `cache_behavior_cases.json`: 5턴 중 0턴 통과

v1.0.2는 앞의 세 항목을 유지하면서 뒤의 세 항목을 개선해야 한다.

## 핵심 요구사항

### 1. 설명형 답변과 절차형 답변을 명확히 구분해야 한다

아래와 같은 응답은 번호가 있더라도 절차형 상태로 저장되면 안 된다.

- `ConfigMap과 Secret 차이 설명`
- `PV, PVC, StorageClass 차이 설명`
- `Service, Ingress, Route 비교`

`procedure_state`는 아래 조건에서만 생성되어야 한다.

- 사용자가 `단계`, `절차`, `순서`, `step` 등 절차형 intent를 명시한 경우
- 또는 retrieval context가 절차/실습 단계 중심 섹션으로 강하게 판별된 경우
- 또는 최종 answer route가 명시적인 절차 응답인 경우

### 2. 멀티턴 상태는 topic continuity뿐 아니라 intent continuity를 유지해야 한다

세션/토픽 메모리는 아래 상태를 유지해야 한다.

- 직전 명시 리소스
- 직전 intent
- 직전 response shape
- 직전 answer route
- 직전 format constraint
- 직전 code resource kind
- 직전 grounded chunk/page 후보

즉 후속 질문은 단순히 "무슨 문서 이야기였는가"가 아니라 "무엇을 어떤 형태로 답하고 있었는가"를 이어받아야 한다.

### 3. explicit resource가 있으면 ambiguity보다 우선해야 한다

다음 유형은 clarification보다 바로 해석되어야 한다.

- `그럼 Service yaml 예시 보여줘`
- `그중 PVC만 더 자세히 설명해줘`
- `StorageClass 예시로도 바꿔줘`
- `Ingress는 어떤 점이 달라?`

현재 턴에 리소스가 명시되어 있으면, 이전 턴의 복수 후보가 있더라도 해당 리소스를 우선 해석해야 한다.

### 4. follow-up intent를 더 세분화해야 한다

v1.0.2에서는 아래 follow-up을 분리해 다뤄야 한다.

- `resource_focus_followup`
- `format_switch_followup`
- `field_followup`
- `procedure_navigation_followup`

예시:

- `그중 PVC만 더 자세히` -> `resource_focus_followup`
- `예시 yaml도 보여줘` -> `format_switch_followup`
- `그 yaml에서 selector는 왜 쓰는거야?` -> `field_followup`
- `다음 단계 보여줘` -> `procedure_navigation_followup`

### 5. 코드/YAML 추출은 resource precision을 우선해야 한다

요청에 explicit resource kind가 있으면 그 kind와 맞는 코드 블록이 우선 선택되어야 한다.

예:

- `ConfigMap YAML 예시`
  - `kind: ConfigMap` 우선
  - `kind: Pod`는 보조 맥락으로도 과도하게 섞이면 안 됨

- `Route 예시`
  - `kind: Route`
  - `route.openshift.io/v1` 또는 동등한 Route 맥락이 직접 보여야 함

### 6. 첫 턴 abbreviation recall을 보강해야 한다

다음 질의는 후속 문맥 없이도 안정적으로 grounded retrieval이 되어야 한다.

- `PV와 PVC 차이 설명`
- `RBAC 구성 요소 설명`
- `Route 예시`
- `ArgoCD가 뭐야?`
- `Tekton이 뭐야?`

이를 위해 alias/abbreviation 확장을 retrieval query 단계까지 반영해야 한다.

### 7. LLM 실패 시에도 route와 answer shape가 무너지면 안 된다

LLM 생성 실패 시 단순 raw context dump 대신 route-aware fallback이 필요하다.

- 비교 질문이면 grounded bullet summary
- YAML 요청이면 extractive code fallback
- 표 요청이면 extractive table fallback
- 일반 grounded generation이면 deterministic summary fallback

이 fallback은 이후 follow-up state도 오염시키지 않아야 한다.

## 성공 기준

### 1. 수동 평가 기준

필수 목표:

- `document_grounding_cases.json`: 전부 통과 유지
- `out_of_domain_cases.json`: 전부 통과 유지
- `language_guard_cases.json`: 전부 통과 유지
- `multiturn_cases.json`: 10턴 중 8턴 이상 통과
- `long_multiturn_cases.json`: 10턴 중 8턴 이상 통과
- `cache_behavior_cases.json`: 5턴 중 4턴 이상 통과

권장 목표:

- `multiturn_cases.json`: 10/10
- `long_multiturn_cases.json`: 9/10 이상
- `cache_behavior_cases.json`: 5/5

### 2. 응답 품질 기준

- 설명형 비교 답변이 `procedure_state_followup`로 잘못 분류되지 않는다.
- explicit resource가 있는 follow-up은 clarification보다 직접 해석이 우선된다.
- YAML 요청에서 무관한 리소스 코드 혼입이 크게 줄어든다.
- abbreviation 기반 첫 질의에서 `관련 내용을 찾기 어렵습니다` 실패율이 낮아진다.
- 동일 세션 동일 질문 반복 시 캐시는 유지되며, 설명 요청과 YAML 요청은 분리된 캐시로 동작한다.

### 3. 사용자 경험 기준

- 최소 5턴 이상 대화에서 주제, 리소스, 답변 형태가 자연스럽게 이어진다.
- `설명 -> 예시 -> 필드 질문 -> 다른 리소스로 전환` 흐름이 끊기지 않는다.
- 차단 정책 메시지는 유지되지만 문서 관련 기술 질문은 과잉 차단되지 않는다.
- LLM 실패 시에도 "문서 기반 답변"이라는 감각이 유지된다.

## 비포함 범위

이번 스펙에 포함되지 않는 것:

- UI 전면 개편
- 새로운 인증 체계 도입
- 외부 RAG 프레임워크 교체
- 일반 지식 답변 허용 범위 확대
- 문서에 없는 내용을 LLM 상식으로 보완하는 정책
- 새로운 제품 기능 추가

## 주요 영향 파일

핵심 구현 후보:

- `app/rag/pipeline.py`
- `app/rag/memory.py`
- `app/services/turn_context_resolver.py`
- `app/services/turn_policy_service.py`
- `app/services/query_interpreter.py`
- `app/services/answer_service.py`
- `app/rag/retrieval.py`

핵심 검증 후보:

- `tests/test_pipeline_unit.py`
- `tests/test_multiturn.py`
- `tests/test_turn_context_resolver.py`
- `tests/test_turn_policy.py`
- `tests/test_cache.py`
- `tests/datasets/multiturn_cases.json`
- `tests/datasets/long_multiturn_cases.json`
- `tests/datasets/cache_behavior_cases.json`

## 완료 정의

v1.0.2는 아래를 만족할 때 완료로 본다.

- README 과제 요구사항을 해치지 않는 범위에서 개선이 구현됨
- 멀티턴 route 오염의 핵심 원인이 제거됨
- YAML/code precision이 명확히 개선됨
- 저장된 manual-eval 기준선보다 멀티턴과 캐시 성능이 유의미하게 상승함
- 기존 grounding / guardrail 품질이 유지됨

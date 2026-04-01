# RAG Multiturn Hardening v1.0.3 Specs

작성일: 2026-04-01

기준 문서:

- `README.md`
- `docs/specs/v1.0.2/specs.md`
- `docs/specs/v1.0.2/plan.md`
- `tests/results/multiturn_cases.v102e.json`
- `tests/results/long_multiturn_cases.v102e.json`
- `tests/results/cache_behavior_cases.v102e.json`

## 목적

v1.0.3의 목적은 v1.0.2에서 개선된 멀티턴 품질을 장기 멀티턴과 절차형 follow-up까지 안정적으로 확장하는 것이다.

이번 버전은 특히 다음 남은 실패를 해결해야 한다.

- 절차형 첫 질문이 `procedure_state_followup`로 관측되어 route가 어긋나는 문제
- 비교/설명 후 부분집합 질문이 `extractive_code` 또는 `clarification`으로 잘못 기우는 문제
- `설명 -> yaml 예시 -> yaml field -> 다른 리소스 예시` 흐름이 장기 멀티턴에서 끊기는 문제
- 명시 리소스가 있는 follow-up도 실제 retrieval 결과가 비어 `general_chat`으로 떨어지는 문제
- 동일 세션 동일 질문 반복 시 캐시 기대가 흔들리는 문제

## 과제 규칙

이번 스펙은 아래 제약을 반드시 지킨다.

- 오픈소스 RAG 프레임워크를 새로 도입하지 않는다.
- 사용자 -> RAG -> LLM 전체 흐름은 현재 코드베이스 내 직접 구현 구조를 유지한다.
- 지정된 LLM 제약은 유지한다.
- SSE 스트리밍을 유지한다.
- 세션 단위 memory와 직접 구현 retrieval/cache 구조를 유지한다.
- 문서에 없는 내용을 LLM 일반 상식으로 보완하지 않는다.
- out-of-domain 차단과 language guard는 약화시키지 않는다.

## 추가 금지 규칙

v1.0.3에서는 아래 방식으로 문제를 해결하면 안 된다.

- 특정 키워드 목록을 더 많이 박아 넣는 식의 확장
- `PVC면 이렇게`, `Service면 저렇게` 식의 리소스별 하드코딩 분기 추가
- 특정 데이터셋 문장에 맞춘 예외처리
- 특정 PDF 파일명/페이지 번호를 기준으로 한 예외처리

즉, 해결 방식은 키워드 하드코딩이 아니라 다음 축을 따라야 한다.

- state model 개선
- retrieval / rerank / selection 일반화
- route decision 일반화
- answer-shape state 일반화

## 현재 기준선

최신 수동 평가 결과:

| 세트 | 결과 |
| --- | --- |
| `multiturn_cases.v102e.json` | 7 / 10 |
| `long_multiturn_cases.v102e.json` | 4 / 10 |
| `cache_behavior_cases.v102e.json` | 4 / 5 |

남은 대표 실패:

- `ConfigMap 생성 절차를 단계별로 설명해줘`
- `그럼 2단계만 알려줘`
- `그중 PVC만 더 자세히 설명해줘`
- `예시 yaml도 보여줘`
- `그 yaml에서 selector는 왜 쓰는거야?`
- `StorageClass 예시로도 바꿔줘`
- `그럼 Service yaml 예시 보여줘`
- `그 중 edge 방식만 다시 설명해줘`
- `ConfigMap을 설명해줘` 반복 시 캐시 기대 불일치

## 핵심 요구사항

### 1. Route는 answer shape와 query intent를 분리해 추적해야 한다

현재 실패 중 일부는 질문 의도와 최종 answer shape가 섞여서 발생한다.

v1.0.3에서는 아래를 별도 상태로 유지해야 한다.

- current turn intent
- current turn answer route
- current turn answer shape
- previous turn answer shape
- previous turn example anchor
- previous turn focus entity

예:

- `PVC만 더 자세히`는 설명 확대이지 code route가 아니다
- `예시 yaml도 보여줘`는 설명에서 code로 전환된 것이다
- `그 yaml에서 selector는 왜 써?`는 직전 code example의 field follow-up이다

### 2. Procedure navigation은 topic navigation과 별도 관리해야 한다

절차형 흐름은 아래 상태를 독립적으로 가져야 한다.

- step list
- current step pointer
- step source block ids
- step source section path
- step source route

그리고 아래가 가능해야 한다.

- `1단계부터`
- `2단계만`
- `다음 단계`
- `다시 RBAC 차이점만`

즉 절차 navigation과 토픽 전환이 충돌하지 않아야 한다.

### 3. Example anchor memory가 필요하다

장기 멀티턴 실패의 공통점은 yaml/code 예시를 한 번 보여준 뒤 그 anchor를 잃는 것이다.

v1.0.3에서는 topic-local state에 아래를 저장해야 한다.

- last example block ids
- last example resource kind
- last example format
- last example source pages
- last example fields

이 상태를 기반으로 다음을 처리해야 한다.

- `예시 yaml도 보여줘`
- `그 yaml에서 selector는 왜 쓰는거야?`
- `StorageClass 예시로도 바꿔줘`

### 4. Retrieval는 short follow-up에서도 prior anchor를 재사용해야 한다

장기 멀티턴에서는 현재 턴 텍스트만으로는 retrieval recall이 약한 경우가 많다.

v1.0.3에서는 아래 전략이 필요하다.

- short follow-up이면 prior focused entity를 retrieval query에 주입
- example follow-up이면 last example anchor를 먼저 재검색
- format switch follow-up이면 prior entity + requested format을 같이 사용
- field follow-up이면 prior example block / section path를 우선 anchor로 사용

이 방식은 generic state 기반이어야 하며, 특정 키워드 목록 추가로 해결하면 안 된다.

### 5. Judge / clarification은 long follow-up에서 더 보수적으로 개입해야 한다

아래 상황에서는 clarification이나 general fallback으로 너무 빨리 떨어지면 안 된다.

- 이전 턴에 명확한 설명이 있었고 현재 턴이 그 부분집합을 요청하는 경우
- 이전 턴에 example이 있었고 현재 턴이 그 필드를 묻는 경우
- 현재 턴에 explicit resource가 있는 경우

즉 clarification은 ambiguity가 정말 높은 경우에만 남겨야 한다.

### 6. Cache는 topic id보다 semantic request identity를 우선해야 한다

동일 세션 동일 질문 재호출이 topic rebinding 때문에 miss나 expectation mismatch가 나면 안 된다.

v1.0.3 cache key는 아래를 반영해야 한다.

- session scope
- rewritten query
- answer route
- intent
- response shape
- resource / format constraint
- selected context ids

반대로 topic id 변화만으로 cache miss가 나면 안 된다.

## 성공 기준

### 1. 수동 평가 기준

필수 목표:

- `multiturn_cases`: 9 / 10 이상
- `long_multiturn_cases`: 8 / 10 이상
- `cache_behavior_cases`: 5 / 5

유지 목표:

- `document_grounding`: 전부 통과 유지
- `out_of_domain`: 전부 통과 유지
- `language_guard`: 전부 통과 유지

### 2. 품질 기준

- `ConfigMap 생성 절차` 질의는 procedure flow로 정상 유지된다.
- `그럼 2단계만 알려줘`는 직전 절차 flow 안에서 해결된다.
- `PVC만 더 자세히 -> 예시 yaml -> selector -> StorageClass 예시` 흐름이 이어진다.
- `Service yaml 예시`는 explicit resource follow-up으로 retrieval된다.
- `edge 방식` 질문은 직전 Route/Ingress 문맥 안에서 처리된다.
- 동일 질문 반복 캐시는 기대대로 hit된다.

### 3. 구현 기준

- 새로운 리소스/문서가 들어와도 동작하는 일반화 설계여야 한다.
- 특정 키워드 하드코딩 추가 없이 테스트를 통과해야 한다.
- 실패를 줄이는 방식은 metadata, state, anchor, route logic의 일반화여야 한다.

## 비포함 범위

이번 스펙에 포함되지 않는 것:

- UI 전면 개편
- 새로운 외부 의존성 추가
- 일반 지식 허용 범위 확대
- 문서 없는 정답 생성 정책
- 파일별 예외 처리

## 영향 파일

핵심 구현 후보:

- `app/rag/pipeline.py`
- `app/rag/memory.py`
- `app/services/query_interpreter.py`
- `app/services/turn_policy_service.py`
- `app/services/turn_context_resolver.py`
- `app/services/answer_service.py`
- `app/services/retrieval_service.py`

핵심 검증 후보:

- `tests/test_pipeline_unit.py`
- `tests/test_multiturn.py`
- `tests/test_turn_policy.py`
- `tests/test_turn_context_resolver.py`
- `tests/test_cache.py`
- `tests/datasets/multiturn_cases.json`
- `tests/datasets/long_multiturn_cases.json`
- `tests/datasets/cache_behavior_cases.json`

## 완료 정의

v1.0.3은 아래를 만족할 때 완료로 본다.

- 장기 멀티턴 실패가 구조적으로 줄어든다.
- cache scenario가 전부 통과한다.
- 설명/절차/code follow-up이 서로 다른 route로 안정적으로 분리된다.
- 키워드 하드코딩 확장 없이 generic state 기반으로 해결된다.

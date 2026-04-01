# RAG Task v1.0.2 분석

작성일: 2026-04-01

## 목적

최상위 `README.md`에 적힌 과제 목적, 과제 내용, 추가 요구사항을 기준으로 현재 리포지토리의 준수 상태를 다시 점검하고, 특히 멀티턴 대응과 RAG 품질을 어떻게 더 강화할지 구조적으로 정리한다.

이번 문서는 다음 네 가지에 집중한다.

- 과제 요구사항 기준 현재 강점과 리스크를 분리해서 본다.
- `v1.0.1`에서 다룬 품질 하드닝 이후에도 남아 있는 병목을 명확히 잡는다.
- 멀티턴 대응을 "토픽 유지" 수준에서 "의도/리소스/포맷 유지" 수준으로 끌어올리는 설계를 제안한다.
- 실제 코드 구조와 최근 수동 평가 결과를 근거로 우선순위를 제시한다.

## 참고 기준

주요 기준 문서:

- `README.md`
- `docs/specs/v1.0.1/2026-03-31-rag-quality-hardening-specs.md`
- `docs/2026-03-30/multiturn-redesign-plan.md`
- `docs/2026-03-31/session-isolation-multiturn-hardening-plan.md`
- `docs/2026-03-31/intent-aware-rag-hardening-implementation-plan.md`

주요 분석 대상 코드:

- `app/rag/pipeline.py`
- `app/rag/memory.py`
- `app/services/turn_context_resolver.py`
- `app/services/turn_policy_service.py`
- `app/services/query_interpreter.py`
- `app/services/answer_service.py`
- `app/rag/retrieval.py`

주요 검증 근거:

- `tests/results/document_grounding_cases.json`
- `tests/results/out_of_domain_cases.json`
- `tests/results/language_guard_cases.json`
- `tests/results/multiturn_cases.json`
- `tests/results/long_multiturn_cases.json`
- `tests/results/cache_behavior_cases.json`

## 과제 요구사항 기준 현재 평가

### 1. 과제 목적 및 과제 내용

`README.md` 기준으로 이 프로젝트는 아래 요구를 충족해야 한다.

- 사용자 -> RAG -> LLM 전체 흐름을 직접 구현
- 오픈소스 RAG 프레임워크에 기대지 않고 자체 구현
- 멀티턴 대화 최소 5턴 이상 지원
- 지정된 LLM 엔드포인트와 모델 사용

현재 구현은 이 방향 자체는 맞다.

- FastAPI + 직접 구현한 파이프라인, 직접 구현한 hybrid retrieval, 직접 구현한 session memory 구조를 유지하고 있다.
- `QueryAgent`, `JudgeAgent`, `TurnPolicyService`, `TurnContextResolver`, `SessionStore`가 분리되어 있어 설계 의도는 분명하다.
- 멀티턴을 단순 chat history 전달이 아니라 앱 레벨 state로 관리하고 있다.

다만 평가 리스크도 분명하다.

- `README.md`는 Hashing/E5 기반 설명이 중심인데 실제 런타임은 `BGEOllamaEmbedder`와 `BGEReranker`를 사용한다.
- 문서가 설명하는 구조와 실제 코드가 어긋나면 과제 발표나 평가에서 "무엇을 직접 구현했는지"가 흐려질 수 있다.
- 멀티턴 5턴 이상 지원은 "구조는 있음" 상태이지 "검증 통과" 상태는 아니다.

### 2. 추가 요구사항

요구사항별 현재 상태는 아래와 같다.

| 항목 | 현재 상태 | 판단 |
| --- | --- | --- |
| Streaming 응답 처리 | SSE 토큰/컨텍스트/완료 이벤트 구현 | 충족 |
| Vector Index 직접 설계/구현 | `VectorIndex`, `HybridRetriever`, `RetrievalService` 직접 구현 | 충족 |
| 세션 단위 memory 구조 | `SessionStore`, `session_topics`, `turn_topic_links`, `topic_state_json` 존재 | 충족 |
| RAG 성능 개선 전략 | hybrid retrieval, rerank, metadata-aware rerank, local expansion 존재 | 충족 |
| 캐싱 전략 | embedding/answer cache 존재 | 충족 |
| 멀티턴 5턴 이상 | 구조는 존재하나 수동 평가 결과는 미달 | 미충족 |

정리하면, 아키텍처 항목은 대부분 갖춰져 있지만 "품질과 안정성" 항목에서 멀티턴과 일부 캐시 시나리오가 막히고 있다.

## 현재 강점

### 1. 직접 구현 구조가 분명하다

- `app/rag/retrieval.py`는 dense + BM25 + RRF + overlap rerank를 직접 구현한다.
- `app/rag/memory.py`는 세션 요약, topic state, topic thread를 DB에 직접 저장한다.
- `app/rag/pipeline.py`는 retrieval, rewrite, policy, cache, answer routing을 한 오케스트레이터로 묶는다.

즉, 과제에서 강조한 "직접 설계/직접 구현" 포인트는 충분히 설명 가능한 상태다.

### 2. 차단 정책과 grounded retrieval은 이미 강하다

최근 저장된 수동 평가 결과 기준:

| 평가 세트 | 결과 |
| --- | --- |
| `document_grounding_cases.json` | 6/6 통과 |
| `out_of_domain_cases.json` | 4/4 통과 |
| `language_guard_cases.json` | 5/5 통과 |

이 부분은 이미 `v1.0.1` 하드닝의 성과로 볼 수 있다.

### 3. 멀티턴을 위한 기반 데이터 모델은 이미 있다

- `session_topics`
- `turn_topic_links`
- `topic_state_json`
- `procedure_state`

즉, 지금 필요한 것은 "새로운 큰 프레임워크"가 아니라, 이미 있는 state를 더 정밀하게 해석하고 사용하는 일이다.

## 현재 핵심 병목

### 1. 절차형 상태 추출이 너무 넓어서 정상 설명 답변까지 procedure로 오염된다

현재 `app/rag/pipeline.py`의 `_extract_procedure_state()`는 번호가 붙은 답변을 단계형 상태로 저장한다. 문제는 비교/설명 답변도 `1.`, `2.`, `3.` 구조를 자주 사용한다는 점이다.

실제 결과:

- `multiturn_cases.json` 전체: 10턴 중 1턴만 통과
- `long_multiturn_cases.json` 전체: 10턴 중 1턴만 통과
- `storage-five-turn-thread` 1턴차 `PV, PVC, StorageClass 차이를 설명해줘`가 `grounded_generation`이 아니라 `procedure_state_followup`로 관측됨
- `multiturn-configmap-compare-to-yaml` 1턴차 `ConfigMap과 Secret의 차이를 설명해줘`도 동일하게 `procedure_state_followup`로 관측됨

이건 멀티턴 설계의 핵심 오류다.

- 비교 답변의 번호 매김
- 실제 절차 안내의 단계 매김

이 둘을 같은 상태로 저장하면 이후 follow-up에서 route가 붕괴한다.

### 2. topic continuity는 있지만 intent continuity가 약하다

현재 구조는 "무슨 주제였는가"는 어느 정도 이어가지만, 아래 상태는 안정적으로 이어가지 못한다.

- 직전 리소스가 무엇이었는가
- 직전 응답 포맷이 설명이었는가 YAML이었는가 표였는가
- 직전 코드 블록이 어떤 리소스 kind였는가
- 직전 답변에서 사용한 grounded chunk/page가 무엇이었는가

예를 들어 아래 흐름이 깨진다.

- `PV, PVC, StorageClass 차이를 설명해줘`
- `그중 PVC만 더 자세히 설명해줘`
- `예시 yaml도 보여줘`
- `그 yaml에서 selector는 왜 쓰는거야?`

현재는 첫 턴 이후 `PVC`라는 좁혀진 focus와 `yaml example`이라는 포맷 전환 상태가 안정적으로 이어지지 않는다.

### 3. explicit resource가 있는 follow-up에도 clarification이 과하게 발생한다

`TurnContextResolver`와 `TurnPolicyService`는 ambiguity를 피하려는 방향으로 설계되어 있다. 방향 자체는 맞지만 지금은 보수성이 너무 높다.

실제 결과:

- `networking-five-turn-thread` 2턴차 `그럼 Service yaml 예시 보여줘`가 바로 `extractive_code`로 가야 하는데 `clarification`으로 빠진다.

이 질문에는 이미 `Service`라는 명시 리소스가 들어 있다. 이런 경우는 "후보가 여러 개"여도 clarification보다 explicit override가 우선이어야 한다.

### 4. 코드 예시 선택이 block-level precision보다 page-level recall에 더 치우쳐 있다

`ConfigMap YAML` 요청에서 실제로는 `kind: ConfigMap`만 우선되어야 하는데, 현재는 인접 페이지/섹션 확장과 generic code block 선택 때문에 `kind: Pod`가 같이 섞인다.

실제 결과:

- `multiturn-configmap-compare-to-yaml` 2턴차는 route는 `extractive_code`로 맞지만 `kind: Pod`가 혼입되어 실패한다.

즉 현재 문제는 "예시를 못 찾는 것"보다 "맞는 예시만 정확히 고르지 못하는 것"에 가깝다.

### 5. 캐시는 존재하지만 캐시 성공보다 선행 답변 품질이 더 큰 병목이다

`cache_behavior_cases.json`은 전체 pass rate 0.0이지만, 세부적으로 보면 캐시 자체는 일부 맞게 동작한다.

예:

- `cache-repeat-same-question` 2턴차는 `observed_cached=true`

그런데도 전체 시나리오는 실패한다. 이유는 첫 응답이 이미 기대 품질에 못 미치기 때문이다.

즉 현재 캐시의 1차 문제는 "키가 틀렸다"보다 "캐싱할 결과 자체가 불안정하다"이다.

### 6. LLM 실패 시 fallback이 raw context dump에 가까워 route 품질을 더 망친다

`LLM 응답 생성에 실패했습니다. 현재는 검색된 문맥만 보여드릴게요.` 경로는 운영상 안전장치로는 유용하지만, 평가 기준에서는 오히려 다음 문제를 만든다.

- 답변 route가 사실상 설명/비교가 아니라 raw dump로 보인다.
- 번호 매김이 있는 문맥을 그대로 보여주면서 `procedure_state`까지 잘못 추출된다.
- 후속 질문의 출발점이 정제된 답변이 아니라 문맥 덩어리가 되어 referential follow-up 품질이 떨어진다.

## 멀티턴 대응 강화를 위한 핵심 설계안

### 1. `procedure_state`를 "번호 있는 답변"이 아니라 "절차형 intent"에서만 생성하도록 좁혀야 한다

권장 원칙:

- 절차 상태는 answer text가 아니라 `query_interpretation.intent`, `response_shape`, assistant metadata로 생성한다.
- 아래 조건을 모두 만족할 때만 procedure state를 만든다.
  - 사용자가 단계/순서/절차를 명시적으로 요청
  - 또는 retrieval context 자체가 procedure section으로 강하게 분류됨
  - 또는 assistant final route가 명시적 절차 응답으로 선택됨

즉 다음은 procedure가 아니다.

- `A/B/C 차이 설명`
- `Service, Ingress, Route 비교`
- `PV/PVC/StorageClass 역할 정리`

이 변경 하나만으로도 현재 route 오염의 상당 부분이 줄어든다.

### 2. Topic state를 `topic continuity`에서 `intent continuity`로 확장해야 한다

`topic_state` 또는 topic thread summary에 아래 필드를 추가하는 것이 좋다.

- `last_explicit_resource`
- `last_explicit_resources`
- `last_intent`
- `last_response_shape`
- `last_answer_route`
- `last_format_constraints`
- `last_code_resource_kind`
- `last_grounded_chunk_ids`
- `last_grounded_section_paths`
- `last_example_source_pages`

이 상태가 있어야 다음 전환이 안정적으로 처리된다.

- 설명 -> YAML 예시
- YAML 예시 -> 특정 필드 질문
- 비교 설명 -> 특정 항목만 확대
- 같은 토픽 내 리소스 전환

### 3. follow-up 해석 우선순위를 다시 정의해야 한다

권장 우선순위:

1. 현재 턴 explicit signal
2. 직전 topic-local state
3. session-global topic continuity
4. LLM rewrite

즉 현재 턴에 `Service yaml`, `PVC만`, `StorageClass 예시`처럼 명시 리소스가 있으면 ambiguity보다 explicit signal이 먼저 이겨야 한다.

이 우선순위를 지키면 아래가 자연스러워진다.

- `그럼 Service yaml 예시 보여줘` -> clarification 아님
- `그중 PVC만 더 자세히` -> 이전 비교 답변의 하위 리소스 선택
- `StorageClass 예시로도 바꿔줘` -> 같은 토픽 내 resource switch

### 4. "resource focus 전환"을 독립된 턴 타입으로 다뤄야 한다

현재 follow-up은 대체로 다음 두 축 사이에 끼어 있다.

- document_followup
- procedure_followup

하지만 실제 실패 케이스는 그 사이에 있는 전환이다.

- compare -> single resource drill-down
- explain -> yaml example
- yaml example -> field question

따라서 아래 intent를 독립적으로 두는 편이 좋다.

- `resource_focus_followup`
- `format_switch_followup`
- `field_followup`
- `procedure_navigation_followup`

이렇게 나누면 route 결정이 훨씬 안정적이다.

### 5. code example selection은 "관련 코드가 있는가"가 아니라 "요청 리소스와 kind가 맞는가"까지 보장해야 한다

현재는 metadata-aware rerank가 존재하지만, explicit resource가 있을 때 hard filter가 아직 약하다.

권장 규칙:

- 요청에 explicit resource kind가 있으면 그 kind와 맞는 block만 1차 통과
- 같은 페이지의 주변 code block은 "보조 후보"로만 사용
- `kind: Pod`와 같이 요청 resource와 다른 kind는 positive evidence가 없는 한 제외
- `selector`, `ports`, `storageClassName` 같은 field follow-up은 직전 선택 code block을 anchor로 재검색

즉 "ConfigMap 관련 문서에 있는 코드"가 아니라 "ConfigMap 자체 YAML"을 뽑는 쪽으로 바꿔야 한다.

### 6. first-turn abbreviation retrieval을 더 강하게 만들어야 한다

`PVC와 PV 차이를 설명해줘`가 낮은 score로 실패한 것은 멀티턴 이전에 retrieval recall 문제다.

개선 방향:

- resource alias lexicon을 retrieval query 확장에도 직접 반영
- `pv -> persistentvolume`, `pvc -> persistentvolumeclaim` 같은 별칭을 sparse token과 rewrite query 둘 다에 주입
- 문서 heading alias index를 별도로 둬서 abbreviation first-turn도 높은 lexical signal을 갖게 함

현재는 `QueryInterpreter`는 자원명을 이해하지만, retrieval recall까지 충분히 밀어주지 못하고 있다.

### 7. LLM 실패 fallback을 "문맥 덤프" 대신 "deterministic grounded summary"로 바꿔야 한다

권장 fallback:

- 비교 질문이면 top grounded chunks를 표/불릿으로 요약
- YAML 요청이면 extractive block만 반환
- 일반 grounded generation 실패면 source-grounded bullet summary 생성

즉 LLM 실패 시에도 route와 answer shape가 무너지지 않게 해야 한다.

## 과제 평가 관점에서 꼭 먼저 손봐야 할 항목

### P0. 바로 손봐야 하는 것

1. README와 실제 구현의 기술 설명을 동기화
2. `procedure_state` 생성 조건 축소
3. explicit resource override 우선순위 도입
4. `resource_focus_followup` / `format_switch_followup` 분리
5. extractive code hard filter 강화

이 다섯 개는 과제 발표, 시연, 수동 평가에 직접적인 영향을 준다.

### P1. 다음 단계

1. alias-aware retrieval expansion
2. field-level follow-up anchor state 추가
3. LLM failure deterministic fallback
4. topic thread 요약에 intent/format/code state 포함

### P2. 정리 및 운영성

1. 최신 아키텍처 기준 문서 정리
2. manual eval 결과를 release gate처럼 사용
3. route별 실패 로그를 더 잘 보이게 정리

## 권장 구현 순서

### Phase 1. 상태 오염 제거

- `procedure_state` 생성 축소
- `last_intent`, `last_response_shape`, `last_answer_route` 저장
- 비교 답변과 절차 답변 분리

### Phase 2. follow-up 해석 강화

- explicit resource override
- `resource_focus_followup`, `format_switch_followup`, `field_followup` 추가
- `TurnContextResolver`와 `TurnPolicyService` 역할 경계 단순화

### Phase 3. code/example precision 강화

- `kind` 기준 hard filter
- 직전 code block anchor state 사용
- local context expansion 이후 precision pruning 강화

### Phase 4. retrieval recall 보강

- abbreviation/alias query expansion
- heading alias signal 강화
- `PV/PVC/StorageClass` 같은 자주 실패하는 1턴 질문 우선 보정

### Phase 5. 검증 게이트 정착

- `multiturn_cases.json`, `long_multiturn_cases.json`, `cache_behavior_cases.json`를 필수 회귀 세트로 운영
- route mismatch와 forbidden content 혼입을 별도 리포트

## v1.0.2에서 목표로 삼아야 할 성공 기준

### 1. 수동 평가 기준

최소 목표:

- `document_grounding_cases.json`: 현 수준 유지
- `out_of_domain_cases.json`: 현 수준 유지
- `language_guard_cases.json`: 현 수준 유지
- `multiturn_cases.json`: 10턴 중 8턴 이상 통과
- `long_multiturn_cases.json`: 10턴 중 8턴 이상 통과
- `cache_behavior_cases.json`: 5턴 중 4턴 이상 통과

권장 목표:

- `multiturn_cases.json`: 10/10
- `long_multiturn_cases.json`: 9/10 이상
- `cache_behavior_cases.json`: 5/5

### 2. 사용자 경험 기준

- 비교 설명은 설명 route로 유지된다.
- 설명 후 `yaml 보여줘`는 자연스럽게 code route로 바뀐다.
- `그 yaml에서 selector는 왜 써?` 같은 질문은 직전 code block을 기준으로 답한다.
- 명시 리소스가 있는 질문은 clarification보다 우선한다.
- LLM 실패 시에도 답변 형태와 grounded quality가 크게 무너지지 않는다.

## 문서/구조 측면 추가 제안

### 1. `docs`와 `docss` 이중 경로는 의도적으로 관리해야 한다

현재 기존 스펙은 `docs/specs/v1.0.1`에 있고, 이번 요청 경로는 `docss/specs/v1.0.2`다.

평가나 협업 관점에서는 이중 경로가 혼란을 만들 수 있다. 이번 분석 문서는 요청대로 `docss`에 두되, 이후에는 아래 둘 중 하나로 정리하는 것이 좋다.

- `docs/specs`로 일원화
- `docss`를 임시 draft 공간으로 명시

### 2. README는 "현재 코드 기준"으로 다시 맞춰야 한다

특히 아래는 바로 맞추는 편이 좋다.

- 실제 임베딩/리랭커 스택
- 현재 멀티턴 구조
- 최신 manual eval 기준 품질 상태
- 과제 평가 포인트와 연결되는 설명

README가 최신 구조를 정확히 반영해야 과제 제출물 설득력이 올라간다.

## 결론

현재 리포지토리는 과제의 구조적 요구사항은 상당 부분 충족하고 있다. 문제는 "기능이 없다"가 아니라 "멀티턴 상태 해석이 서로 다른 답변 형태를 구분하지 못해서 경로가 충돌한다"는 점이다.

가장 먼저 해결해야 할 것은 retrieval 엔진 교체나 큰 프레임워크 도입이 아니다.

- 절차형 상태 생성 축소
- explicit resource override
- intent/format/code state의 topic-local 저장
- code example hard filtering

이 네 축을 먼저 바로잡으면, 현재 이미 강한 document grounding/out-of-domain/language guard를 유지하면서도 과제 핵심인 멀티턴 대응과 캐시 실효성을 같이 끌어올릴 수 있다.

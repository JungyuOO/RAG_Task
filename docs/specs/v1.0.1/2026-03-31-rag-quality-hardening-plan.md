# RAG Quality Hardening Plan

기준 스펙:

- [`2026-03-31-rag-quality-hardening.md`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\2026-03-31-rag-quality-hardening.md)

## 고정 작업 원칙

모든 태스크는 아래 순서를 고정한다.

1. 테스트 작성 또는 기존 테스트 확장
2. 구현
3. 테스트 통과 확인

추가 원칙:

- 각 태스크는 아래에 명시된 파일 경로 안에서만 수정한다.
- 명시되지 않은 파일은 필요성이 확인되기 전에는 수정하지 않는다.
- 기존 수동 평가 데이터셋과 결과 JSON 구조는 유지한다.
- `how`는 구현 시점의 코드베이스 분석으로 결정하되, `what`과 수정 범위는 이 계획서로 고정한다.

## Phase 1. 문서 존재 토픽 미탐지 해소

### Task 1-1. 기술 주제 인식 범위 확장

목표:

- ArgoCD, Tekton, RBAC, Route, StorageClass 같은 문서 존재 토픽을 문서 관련 질문으로 안정적으로 인식한다.

수정 파일:

- [`app/services/query_interpreter.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\services\query_interpreter.py)
- [`app/services/turn_policy_service.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\services\turn_policy_service.py)
- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)

테스트 파일:

- [`tests/test_query_interpreter.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_query_interpreter.py)
- [`tests/test_turn_policy.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_turn_policy.py)
- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)

완료 조건:

- `ArgoCD가 뭐야?`, `Tekton이 뭐야?`, `RBAC 구성 요소를 표로 정리해줘`가 general chat 거절로 빠지지 않는다.

### Task 1-2. 문서 존재 예시 코드 미탐지 해소

목표:

- 문서에 실제 존재하는 Route YAML, StorageClass YAML, RBAC 관련 표/설명을 찾지 못하는 문제를 줄인다.

수정 파일:

- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)
- [`app/services/answer_service.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\services\answer_service.py)

테스트 파일:

- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)

수동 검증 대상:

- [`tests/datasets/document_grounding_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\document_grounding_cases.json)

완료 조건:

- `OpenShift Route 예시도 보여줘` 류 질문이 문서 부재로 잘못 응답하지 않는다.

## Phase 2. 멀티턴 문맥 상속 강화

### Task 2-1. referential follow-up 해석 강화

목표:

- `그중`, `그 yaml`, `그럼 ~ 예시`, `다시 ~만` 같은 후속질문이 직전 리소스와 답변 형태를 이어받는다.

수정 파일:

- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)
- [`app/rag/memory.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\memory.py)
- [`app/services/turn_context_resolver.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\services\turn_context_resolver.py)

테스트 파일:

- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)
- [`tests/test_multiturn.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_multiturn.py)
- [`tests/test_turn_context_resolver.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_turn_context_resolver.py)

수동 검증 대상:

- [`tests/datasets/multiturn_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\multiturn_cases.json)
- [`tests/datasets/long_multiturn_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\long_multiturn_cases.json)

완료 조건:

- 5턴 이상 시나리오에서 문맥 단절 응답이 줄어든다.

### Task 2-2. procedure follow-up 과 entity follow-up 분리 안정화

목표:

- 절차형 후속질문과 일반 referential follow-up을 서로 다른 상태로 관리하되, 충돌하지 않게 한다.

수정 파일:

- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)
- [`app/rag/memory.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\memory.py)

테스트 파일:

- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)

완료 조건:

- `1단계`, `다음 단계`, `그 yaml`, `StorageClass 예시로도 바꿔줘`가 서로 엉키지 않는다.

## Phase 3. 코드 추출 정밀도 강화

### Task 3-1. 요청 리소스와 직접 관련된 코드만 우선 선택

목표:

- ConfigMap YAML 요청에서 Pod YAML이 과도하게 섞이는 문제를 줄인다.

수정 파일:

- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)
- [`app/services/answer_service.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\services\answer_service.py)

테스트 파일:

- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)

수동 검증 대상:

- [`tests/datasets/document_grounding_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\document_grounding_cases.json)
- [`tests/datasets/multiturn_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\multiturn_cases.json)

완료 조건:

- `ConfigMap YAML 예시` 응답에서 ConfigMap과 직접 무관한 Pod 코드 비중이 줄어든다.

### Task 3-2. 페이지 주변 확장과 최종 추출 필터의 역할 분리

목표:

- local expansion은 recall을 위해 유지하되, 최종 응답에는 precision 기준을 더 강하게 적용한다.

수정 파일:

- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)

테스트 파일:

- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)

완료 조건:

- “문서에 있는 관련 코드 탐지”와 “최종 응답 코드 정밀도”가 동시에 개선된다.

## Phase 4. 캐시 실효성 강화

### Task 4-1. 동일 질의 캐시 히트 보장

목표:

- 같은 세션에서 같은 질문을 반복했을 때 실제 결과 JSON에서 `observed_cached=true`가 잡힌다.

수정 파일:

- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)
- [`app/repositories/answer_cache_repository.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\repositories\answer_cache_repository.py)

테스트 파일:

- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)
- [`tests/test_cache.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_cache.py)

수동 검증 대상:

- [`tests/datasets/cache_behavior_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\cache_behavior_cases.json)

완료 조건:

- 반복 질의 시 캐시 히트가 실제 결과 JSON에 반영된다.

### Task 4-2. 의도별 캐시 분리 유지

목표:

- 같은 주제라도 설명 응답과 YAML 응답이 서로 다른 캐시로 유지된다.

수정 파일:

- [`app/rag/pipeline.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\app\rag\pipeline.py)

테스트 파일:

- [`tests/test_pipeline_unit.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\test_pipeline_unit.py)

완료 조건:

- `ConfigMap 설명` 캐시가 `ConfigMap YAML 예시` 응답에 재사용되지 않는다.

## Phase 5. 평가 데이터셋과 결과 판정 강화

### Task 5-1. 수동 평가 데이터셋 판정 기준 강화

목표:

- 단순 `must_include_any` 통과가 실제 품질을 과대평가하지 않도록 한다.

수정 파일:

- [`tests/datasets/cache_behavior_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\cache_behavior_cases.json)
- [`tests/datasets/document_grounding_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\document_grounding_cases.json)
- [`tests/datasets/multiturn_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\multiturn_cases.json)
- [`tests/datasets/long_multiturn_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\long_multiturn_cases.json)

테스트/실행 파일:

- [`tests/run_manual_eval.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\run_manual_eval.py)
- [`tests/run_all_manual_eval.ps1`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\run_all_manual_eval.ps1)

완료 조건:

- 부분 정답과 정밀 정답을 더 명확히 구분할 수 있다.

### Task 5-2. 결과 JSON 가독성 유지

목표:

- 결과 JSON은 `final_answer`, `answer_preview`, `cached` 중심으로 빠르게 읽히고, raw event는 옵션으로만 유지한다.

수정 파일:

- [`tests/run_manual_eval.py`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\run_manual_eval.py)
- [`tests/README.md`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\README.md)

완료 조건:

- 품질 검토 시 토큰 단위 SSE 이벤트를 기본적으로 뒤지지 않아도 된다.

## 권장 실행 순서

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5

## 최종 검증 묶음

아래 파일 기준으로 최종 수동 검증을 다시 수행한다.

- [`tests/datasets/document_grounding_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\document_grounding_cases.json)
- [`tests/datasets/multiturn_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\multiturn_cases.json)
- [`tests/datasets/long_multiturn_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\long_multiturn_cases.json)
- [`tests/datasets/cache_behavior_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\cache_behavior_cases.json)
- [`tests/datasets/language_guard_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\language_guard_cases.json)
- [`tests/datasets/out_of_domain_cases.json`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\datasets\out_of_domain_cases.json)

실행 파일:

- [`tests/run_all_manual_eval.ps1`](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\tests\run_all_manual_eval.ps1)

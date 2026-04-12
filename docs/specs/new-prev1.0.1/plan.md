# RAG Task new-prev1.0.1 Plan

작성일: 2026-04-10  
상태: In Progress

관련 문서:

- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\spec.md)
- [inventory.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\inventory.md)
- [decisions.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\decisions.md)
- [dependencies.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\new-prev1.0.1\dependencies.md)
- [spec.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\newv1.0.1\spec.md)
- [plan.md](C:\Users\KJungyu\OneDrive\Desktop\Company\과제\RAG_Task\docs\specs\newv1.0.1\plan.md)

## 1. 목적

`new-prev1.0.1` plan은 `newv1.0.1` 재설계에 들어가기 전에, 현재 리포지토리 구조를 인벤토리하고 정리 대상을 문서화하기 위한 실행 계획이다.

핵심은 구현이 아니라 “현재 구조 해체 계획 수립”이다.

## 2. 실행 원칙

1. 현재 구조를 읽기 전에 함부로 삭제하지 않는다.
2. 먼저 유지 / 삭제 후보 / 통합 후보 / 분리 후보를 문서화한다.
3. 구조 인벤토리 작업도 Phase 단위로 진행한다.
4. 각 Phase 후 검증 및 코드 리뷰를 수행한다.
5. 필요하면 파생 Phase(`1-1`, `2-1`)를 추가한다.
6. 파생 Phase가 있으면 상위 Phase는 미완료 상태로 유지한다.
7. 결과는 `newv1.0.1` plan으로 넘긴다.

## 3. 완료 표시 규칙

- 미완료: `[ ]`
- 진행 중: `[-]`
- 완료: `[x]`

## 4. Phase 계획

### [x] Phase 0. 현재 구조 인벤토리

목표:

- 현재 코드 구조를 읽고 책임 단위를 정리한다.

산출물:

- `inventory.md`

검증:

- 인벤토리 문서와 실제 파일 구조 비교
- 누락 모듈 여부 확인

### [x] Phase 1. 유지 / 삭제 후보 분류

목표:

- 현재 구조에서 무엇을 남기고 무엇을 정리할지 분류한다.

산출물:

- `decisions.md`

검증:

- 분류 결과가 `newv1.0.1` spec과 충돌하지 않는지 검토
- 과도한 삭제 후보 지정 여부 검토

### [x] Phase 2. 의존성 및 라이브러리 정리 후보 도출

목표:

- 새 구조에서 필요한 라이브러리와 제거 가능한 라이브러리를 정리한다.

산출물:

- `dependencies.md`

핵심 결정:

- `pgvector` 도입 방향 확정
- 프론트 스타일링/렌더링 라이브러리 후보 유지
- 즉시 삭제가 아닌 조건부 보류 의존성 구분

검증:

- 현재 `requirements.txt`와 실제 코드 참조 비교
- `newv1.0.1` spec 요구사항과 일치 여부 검토

### [x] Phase 3. newv1.0.1 이관 메모 작성

목표:

- `new-prev1.0.1` 분석 결과를 `newv1.0.1` 실행 계획에 넘길 수 있는 형태로 정리한다.

작업:

- 우선 삭제 대상
- 우선 유지 대상
- 우선 리팩토링 대상
- 1차 구현 착수 대상

검증:

- 이관 메모가 실제 다음 작업 시작에 충분한지 검토

## 5. 현재 상태

- `new-prev1.0.1/spec.md` 작성 완료
- `new-prev1.0.1/plan.md` 작성 완료
- `new-prev1.0.1/inventory.md` 작성 완료
- `new-prev1.0.1/decisions.md` 작성 완료
- `new-prev1.0.1/dependencies.md` 작성 완료
- `new-prev1.0.1/handoff.md` 작성 완료
- Phase 0 완료
- Phase 1 완료
- Phase 2 완료
- Phase 3 완료

## 6. 다음 순서

1. `newv1.0.1` 본계획으로 이관
2. 버전 관련 삭제 후보 정리
3. `newv1.0.1` Phase 1 착수

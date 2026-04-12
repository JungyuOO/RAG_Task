# RAG Task newv1.0.1 Plan

작성일: 2026-04-12
상태: Active

## 완료 표시 규칙

- 미완료: `[ ]`
- 진행 중: `[-]`
- 완료: `[x]`

## 실행 원칙

1. 모든 작업은 Phase 단위로 진행한다.
2. 각 Phase는 구현 후 반드시 검증을 수행한다.
3. 각 Phase는 검증 후 반드시 코드 리뷰를 수행한다.
4. 코드 리뷰는 아키텍처 관점과 코드 리뷰어 관점으로 나눠서 수행한다.
5. 검증 또는 리뷰 중 추가 이슈가 생기면 `1-1`, `2-1` 같은 파생 Phase를 생성한다.
6. 파생 Phase가 열리면 원래 Phase는 완료 처리하지 않는다.
7. 원래 Phase와 파생 Phase가 모두 끝나야 원래 Phase를 완료 처리한다.
8. 불필요한 코드는 즉시 삭제한다.
9. 모든 문서와 코드 자산은 UTF-8 기준으로 유지한다.

## Current Phase Status

- `[-] Phase 1. 답변 품질 및 멀티턴 안정화`
- `[x] Phase 1-1. follow-up helper 정리`
- `[-] Phase 1-2. response-shape validation`
- `[-] Phase 1-3. low-signal answer 보강`
- `[-] Phase 2. 문서 구조 전환`
- `[x] Phase 2-1. block metadata 분리 저장`
- `[x] Phase 2-2. 코드/표/리스트 메타데이터 연결`
- `[-] Phase 2-3. retrieval_text 및 display_text 정제 검증`
- `[x] Phase 3. Vector Index`
- `[x] Phase 3-1. pgvector schema`
- `[x] Phase 3-2. candidate generation`
- `[-] Phase 4. 프론트엔드 렌더링 교체`
- `[x] Phase 4-1. citation UI/preview 고도화`
- `[x] Phase 4-2. chunk viewer 상태 API 연동 보강`
- `[ ] Phase 4-3. 운영 콘솔형 레이아웃 정리`
- `[ ] Phase 4-4. 디자인 polish 및 컴포넌트 정리`
- `[-] Phase 5. OCP API integration`
- `[x] Phase 5-1. Pod/Deployment/Service/Route/Event 조회 안정화`
- `[-] Phase 5-2. OCP Explorer 및 YAML UX 보강`
- `[ ] Phase 5-3. 문서 답변과 OCP 상태 결합 UX`
- `[-] Phase 6. golden dataset`
- `[-] Phase 6-1. golden dataset schema`
- `[-] Phase 6-2. golden dataset seed cases`
- `[-] Phase 6-3. golden dataset scoring/report/compare`
- `[-] Phase 7. golden 데이터 파이프라인 자동화`
- `[ ] Phase 7-1. golden seed 수집/정리`
- `[ ] Phase 7-2. golden run 실행 자동화`
- `[-] Phase 7-3. golden compare/report 자동화`

## 검증 기준

### 공통 검증

- Python 변경은 `py_compile` 또는 관련 단위 테스트로 검증
- 프론트 변경은 `node --check`로 문법 검증
- HTML 변경은 파서 기준으로 구조 검증
- API 변경은 가능하면 실제 HTTP smoke까지 확인

### 코드 리뷰 기준

- 책임 분리가 명확한가
- spec 방향과 실제 구현이 일치하는가
- legacy 경로나 중복 로직이 남아 있지 않은가
- 테스트가 변경 범위를 실제로 보호하는가

## Phase 계획

### Phase 1. 답변 품질 및 멀티턴 안정화

목표:
- 멀티턴 follow-up 안정화
- response-shape 보장
- low-signal answer 제거

진행 상황:
- follow-up helper 정리 완료
- response-shape fallback 일부 적용 완료
- low-signal answer fallback 일부 적용 완료

남은 일:
- golden dataset 실패 케이스 기준으로 answer-shape 품질 재보정
- 실제 앱 기준 멀티턴 재검증

### Phase 2. 문서 구조 전환

목표:
- 공식/고객사 문서를 HTML + metadata + 정제 텍스트 구조로 전환

완료:
- markdown / html / metadata.json artifact 분리
- page/block/table/code metadata 저장
- chunk metadata enrichment
- retrieval_text 정제
- chunk 생성 시 display_text 저장
- chunk viewer에서 display_text 우선 사용

남은 일:
- 재인덱싱 후 retrieval 품질 검증
- 정규화 규칙 추가 보정 여부 판단

### Phase 3. Vector Index

목표:
- pgvector 기반 dense candidate prefilter 도입

완료:
- Postgres `vector` extension 사용
- `embedding` 컬럼 생성 및 legacy backfill
- dense prefilter / supplemented / full fallback 경로 구현
- debug retrieval diagnostics 추가

### Phase 4. 프론트엔드 렌더링 교체

목표:
- source 탐색, preview, chunk viewer를 제품 수준으로 정리

완료:
- inline citation `[1] [2]`
- HTML preview 우선
- raw PDF 분리 버튼
- supporting/source 카드
- chunk viewer source action
- OCP Explorer 화면 추가

남은 일:
- 전체 레이아웃 polish 마감
- 운영 콘솔형 정보 계층 정리
- 실제 브라우저 기준 최종 smoke

### Phase 5. OCP API integration

목표:
- read-only OCP API와 문서 답변 흐름을 연결

완료:
- `/api/ocp/status`
- `/api/ocp/resources`
- `/api/ocp/resource-yaml`
- `/api/ocp/namespaces`
- namespace fallback
- `pods / deployments / services / routes / events` live smoke
- OCP Explorer UI
- namespace suggestions
- resource filter / sort / summary
- YAML viewer
- `Ask in chat`

남은 일:
- 실제 브라우저 기준 OCP 화면 smoke
- 문서 답변과 OCP Explorer 연결 UX 보강

### Phase 6. golden dataset

목표:
- 실제 품질 회귀 검증 기준과 자동 채점 도구 구축

완료:
- `scripts/golden_dataset.py`
- `scripts/validate_golden_dataset.py`
- `tests/data/golden_dataset_v1.json`
- `scripts/run_golden_dataset.py`
- `scripts/compare_golden_runs.py`
- JSON/Markdown report 생성
- run-to-run compare
- single-case live smoke

현재 상태:
- schema 있음
- 4.20 기준 seed cases 9개 있음
- scorer / report / compare 도구 있음
- full run은 응답 시간이 길어 baseline 축적 단계

남은 일:
- 전체 dataset full run 기준 정착
- failure bucket을 품질 개선 루프와 연결

### Phase 7. golden 데이터 파이프라인 자동화

목표:
- golden dataset 실행, 비교, 리포트 생성을 반복 가능한 파이프라인으로 자동화

완료:
- `scripts/golden_pipeline.py`
- pipeline run artifact 생성
- compare artifact 생성

남은 일:
- golden seed 수집/정리 규칙 보강
- full-run 정책 정리
- latest baseline 비교 정책 정리

## 4.20 단일 버전 운영 현황

현재 정책:
- 공식 문서는 4.20만 유지
- 고객사 가이드는 4.20만 생성
- 다른 공식 버전/고객사 가이드는 정리

현재 반영 상태:
- 로컬 공식 문서: `ocp-4.20`만 남김
- 4.20 공식 문서 20종 인덱싱 완료
- 4.20 고객사 가이드 3종 생성 및 인덱싱 완료

현재 4.20 고객사 가이드:
- `ocp-4.20-baremetal-install-network-dns-customer-guide.md`
- `ocp-4.20-cluster-mtu-migration-customer-guide.md`
- `ocp-4.20-oauth-auth-troubleshooting-customer-guide.md`

## spec/code 비교 메모

현재 코드 기준으로 반영된 항목:
- pgvector 기반 vector index
- HTML/metadata 기반 문서 구조화
- source card / preview / chunk viewer 개선
- OCP API read-only 연계
- golden dataset / runner / compare / pipeline

아직 반영이 약하거나 미완료인 항목:
- 전체 UI 디자인 polish 마감
- 실제 브라우저 기준 OCP/채팅/자료실 smoke
- golden full-run 기준 정착
- golden 파이프라인의 운영 정책 정리

## Current Progress Notes

- `new-prev1.0.1` 분석 및 handoff 완료
- Phase 3 pgvector 경로 실제 DB 검증 완료
- Phase 4 source/preview/chunk viewer UX 핵심 반영 완료
- Phase 5 OCP API live smoke 완료
- Phase 6 golden dataset loader / runner / compare 도구 추가 완료
- Phase 7 golden pipeline wrapper 추가 완료
- 4.20 공식 문서 20종 인덱싱 완료
- 4.20 고객사 가이드 3종 생성 및 인덱싱 완료

## 다음 우선순위

1. 앱 기준 브라우저 smoke 후 Phase 4/5를 더 닫는다.
2. 4.20 기준 golden full run baseline을 쌓는다.
3. golden failure를 기준으로 Phase 1 품질 보정을 반복한다.

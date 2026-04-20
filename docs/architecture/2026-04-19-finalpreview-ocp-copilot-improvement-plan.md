# 2026-04-19 FinalPreview OCP Copilot Improvement

## Summary
- Chat 응답 스키마에 `artifacts` 와 `citation_map` 을 추가했다.
- Resources 전용이던 YAML editor 흐름을 공용 컴포넌트로 분리해 Chat에서도 재사용 가능하게 만들었다.
- Chat 세션은 viewport 상태를 저장해 페이지 이동 후에도 마지막 스크롤 위치로 복원된다.
- 전역 loading overlay를 AppShell 수준으로 올리고 카드 내부 로딩 문구는 제거했다.
- Live OCP는 resource list/detail 외에 relation, health, dashboard metrics 확장을 시작했다.
- Dashboard는 Prometheus `query_range` 기반 시계열 그래프를 SVG로 렌더링하도록 바뀌었다.

## Frontend
- `ChatPage` 는 artifact 렌더링과 resource YAML modal open 흐름을 지원한다.
- `ResourceYamlEditorModal` 과 `ResourceList` 를 공용 컴포넌트로 분리했다.
- `AppShell` 헤더에서 우측 상단 route/status badge 를 제거했다.
- Dashboard는 `1h/6h/24h` window 기반 utilization cards 를 렌더링한다.

## Backend
- `CopilotChatResponse` 에 `artifacts`, `citation_map` 을 추가했다.
- live answer composer 는 `resource_list`, `resource_editor` artifact 를 반환한다.
- unified copilot service 는 문서 응답에 citation map 과 command template artifact 를 추가한다.
- unsupported command 가 답변에 섞이면 extractive fallback 으로 내려간다.
- `ConnectedOcpService` 는 dashboard metrics, resource relations, resource health summary helper 를 제공한다.

## Remaining Follow-ups
- command template slot substitution 은 현재 canonical command artifact 까지만 구현되었고, 사용자 값 치환은 다음 단계에서 강화가 필요하다.
- live relation/health intent 는 1차 heuristic 확장 기준이며 질문 분류 coverage 를 더 넓혀야 한다.
- Dashboard Prometheus proxy 는 `openshift-monitoring/thanos-querier` 서비스 프록시를 기준으로 구현되었으므로 실제 클러스터별 검증이 필요하다.
- scenario/golden 수준의 멀티턴 정확성 테스트는 추가 확대가 필요하다.

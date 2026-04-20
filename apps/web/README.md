# apps/web

React + Vite + TypeScript 프런트엔드 신규 작업공간.

현재 단계 목표:
- 기존 `app/web` 정적 UI를 바로 삭제하지 않고 병행 이관
- 채팅 / 대시보드 / 리소스 / 라이브러리 / 액션 화면을 feature 기준으로 재구성
- FastAPI `/api/*` 계약을 그대로 소비하는 프런트 구조 먼저 고정

우선순위:
1. Vite + React + TypeScript 부트스트랩
2. Tailwind + shadcn/ui + TanStack Query 연결
3. Chat / Dashboard 최소 라우트 생성
4. 기존 SSE chat 흐름을 React query/state로 재구성

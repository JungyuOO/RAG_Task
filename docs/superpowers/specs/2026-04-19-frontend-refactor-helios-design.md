# Frontend Refactor + Helios-Inspired Design System

**Date:** 2026-04-19
**Branch:** `dev-ver2`
**Scope:** `apps/web` 전체. 백엔드(`apps/api`) 코드는 건드리지 않음.

## 1. 목표

1. `apps/web`의 과세분화된 FSD 구조(빈 `.gitkeep` 폴더 10+ 개)를 제거하고 feature 단위로 단순화.
2. `shared/styles/partials/*.css`(~2,500줄, BEM 수기 관리)를 Tailwind + shadcn/ui 기반 디자인 시스템으로 전환.
3. DESIGN.md(Helios 참조)에 근사한 다크 전용 UI로 전 페이지 리스킨.
4. 기능 삭제·버튼 비활성화 금지. 백엔드 API URL 전부 보존.

## 2. 확정 결정 사항

| # | 항목 | 결정 |
|---|------|------|
| 1 | 스펙 분할 | 한 스펙 단일화 |
| 2 | 스타일 스택 | shadcn/ui + Tailwind CSS |
| 3 | ActionsPage | 네비 정식 편입 |
| 4 | 테마 | 다크 단일 (토글 없음) |
| 5 | 폴더 구조 | `pages/`, `features/<domain>/`, `components/ui/`, `components/layout/`, `lib/`, `styles/` |
| 6 | 레이아웃 | 좌측 고정 사이드바 + 얇은 topbar |
| 7 | 롤아웃 | 3단계 커밋 체인 (인프라 → 구조 → 스타일) |
| 8 | 아이콘 | Lucide React |
| 9 | 채팅 UX | 스트리밍/세션 rail 동작 보존 — rail 위치만 우측 슬롯으로 이동 |
| 10 | API 경로 | 모든 엔드포인트 URL 문자열 불변 |

## 3. 새 폴더 구조

```
apps/web/src/
  pages/
    ConnectionPage.tsx         # 얇은 래퍼, 실로직은 features/connection
    DashboardPage.tsx
    ResourcesPage.tsx
    LibraryPage.tsx
    ChatPage.tsx
    ActionsPage.tsx
  features/
    connection/
      api/           # ocpConnectionApi.ts
      components/    # OcpConnectionForm 등
      hooks/         # useOcpConnection
      types.ts
    chat/
      api/           # copilotChatApi, ocpLiveChatApi, docsPreviewApi
      components/    # ChatComposer, ChatTranscript, ChatSessionRail, MessagePreview
      hooks/         # useCopilotStream
      types.ts
    actions/
      api/           # actionPreviewApi
      components/    # ActionRequestList, ActionPreviewModal, AuditLogTable
      hooks/
      types.ts
    library/
      api/           # libraryApi, indexingApi
      components/    # BatchReindexPanel, DocumentChunksViewer
      hooks/         # useBatchIndexJob
      types.ts
    resources/
      api/           # ocpLiveApi 中 resource 관련
      components/    # ResourceList, ResourceYamlEditorModal
      hooks/
      types.ts
    dashboard/
      api/           # ocpLiveApi 中 overview/metrics 관련
      components/
      hooks/
      types.ts
  components/
    ui/              # shadcn CLI 결과: button, input, card, dialog, badge, tabs, table, dropdown-menu, skeleton, toast
    layout/
      AppShell.tsx   # 좌측 사이드바 + topbar 래퍼
      Sidebar.tsx
      TopBar.tsx
      PageHeader.tsx
  lib/
    http.ts          # fetch 래퍼 (에러/JSON 처리 공통화)
    cn.ts            # shadcn 표준 classnames merger
    storage.ts       # localStorage 래퍼 (세션/레일 상태)
  styles/
    globals.css      # Tailwind directives + CSS 변수 토큰
  App.tsx            # 라우트 스위치
  main.tsx
```

**제거 대상:**
- `src/entities/**` 전체 (타입은 해당 feature 내부로 흡수)
- 전 `.gitkeep` 파일
- `src/shared/styles/partials/*.css` 6개 파일 (3단계에서 삭제)
- 루트의 `_shared_head.js`, `_failed_v2.txt` (사전 확인 후 판단)

## 4. 레이아웃 아키텍처

### AppShell
```
┌────────────────────────────────────────────────────────────┐
│ TopBar (h-14, bg-background/50 blur border-b)             │
│  Breadcrumb · PageTitle        ·        ConnectionBadge   │
├──────────────┬─────────────────────────────────────────────┤
│  Sidebar     │                                             │
│  (w-64 fix)  │                                             │
│              │                                             │
│  [Logo]      │            <page content>                   │
│              │                                             │
│  • Connection│                                             │
│  • Dashboard │     ┌─ optional page-scoped context ─┐      │
│  • Resources │     │  (chat session rail 등)        │      │
│  • Library   │     └───────────────────────────────┘      │
│  • Chat      │                                             │
│  • Actions   │                                             │
│              │                                             │
│  [Profile]   │                                             │
└──────────────┴─────────────────────────────────────────────┘
```

- Sidebar: `bg-[#0d0e12]`, 아이콘(lucide 20px) + 라벨, 활성 아이템은 `bg-[#15181e]` + 좌측 4px 악센트 바.
- Sidebar 하단 프로필 카드: 아바타 + 이름 + 클러스터 URL + 연결상태 배지, 우측 끝에 Disconnect 버튼(shadcn `Button variant="ghost"`).
- TopBar: `bg-[#15181e]/60 backdrop-blur border-b border-[#616875]/20`, 좌측 `PageHeader`, 우측 `ConnectionStatusPill` + `KeyboardHint`(⌘K 나중에).
- `<page content>`는 max-width 컨테이너 없이 `px-8 py-6`로 full-stretch, 페이지 내부 카드가 자체 너비 관리.

### 페이지 우측 컨텍스트 슬롯
Chat 페이지는 `<ChatSessionRail>`을 본문 우측에 `w-72 sticky` 배치. Resources 페이지는 네임스페이스 선택 드로어를 같은 위치에.

## 5. 디자인 토큰 (Tailwind + CSS 변수)

`src/styles/globals.css` (발췌):

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    /* Helios-inspired dark palette */
    --bg: 220 14% 10%;         /* #15181e */
    --bg-elevated: 222 17% 7%; /* #0d0e12 */
    --fg: 240 3% 94%;          /* #efeff1 */
    --fg-muted: 222 8% 44%;    /* #656a76 */
    --border: 222 8% 44% / 0.25;
    --accent: 218 98% 54%;     /* #1060ff */
    --accent-fg: 0 0% 100%;
    --success: 181 82% 43%;
    --warning: 28 100% 37%;
    --destructive: 354 59% 28%;
    --radius: 8px;

    /* shadcn 표준 매핑 */
    --background: var(--bg);
    --foreground: var(--fg);
    --card: 220 14% 12%;
    --card-foreground: var(--fg);
    --primary: var(--accent);
    --primary-foreground: var(--accent-fg);
    --muted: 220 14% 14%;
    --muted-foreground: var(--fg-muted);
  }
}
```

`tailwind.config.ts`: `darkMode: 'class'` (항상 `<html class="dark">` 적용), font 확장 (`font-sans: system-ui`, `font-display: "HashiCorp Sans", system-ui`).

HashiCorp Sans 웹폰트는 **외부 호스팅 하지 않음** — 대체로 **Inter**(Google Fonts) + fallback `system-ui`. 이유: 상업 폰트 권리 문제 회피, Helios의 "weight 600/700 + tight line-height" 리듬은 Inter에서도 재현 가능.

## 6. 페이지별 재구성 개요

| 페이지 | 주요 변화 |
|--------|-----------|
| Connection | 히어로 풀 섹션(`bg-[#0d0e12]`) + 중앙 shadcn `Card`에 폼, 연결 성공 시 Dashboard 링크 유도 |
| Dashboard | `MetricGrid` 4개 shadcn `Card`, 하단 `MetricLineChart` 보존, 우측 상단 네임스페이스 드롭다운 |
| Resources | 좌측 네임스페이스 리스트 Drawer → 우측 `Table`(shadcn) + 행 클릭 시 YAML 에디터 `Dialog` |
| Library | 상단 `Tabs`(Corpus 요약/카탈로그/배치잡) + 배치잡 진행상황 `Card` + 청크 뷰 `Dialog` |
| Chat | 본문 = 트랜스크립트 + 하단 `ChatComposer`, 우측 `ChatSessionRail` `w-72`, 스트리밍 인디케이터는 shadcn `Skeleton`/`Badge` 조합 |
| Actions | `Tabs`(Requests/Executions/Audit), 각 탭 `Table`, 행 액션은 shadcn `DropdownMenu` + 확인 `Dialog` |

## 7. API 경로 보존 매트릭스

| Feature | 프론트 파일 | 백엔드 엔드포인트 (불변) |
|---------|------------|-------------------------|
| connection | `features/connection/api/ocpConnectionApi.ts` | `POST /api/v1/auth/ocp/connect`, `GET /api/v1/auth/ocp/status/{id}`, `POST /api/v1/auth/ocp/test`, `POST /api/v1/auth/ocp/lease/refresh`, `GET /api/v1/auth/ocp/lease/status`, `POST /api/v1/auth/ocp/disconnect` |
| chat | `features/chat/api/copilotChatApi.ts` | `POST /api/v1/chat/query`, `POST /api/v1/chat/query/stream`, `POST /api/v1/chat/live` |
| chat | `features/chat/api/docsPreviewApi.ts` | `GET /api/v1/docs-preview/snippet` |
| actions | `features/actions/api/actionPreviewApi.ts` | `POST /api/v1/actions/preview`, `POST /api/v1/actions/requests`, `GET /api/v1/actions/requests`, `POST /api/v1/actions/requests/{id}/approve|reject|execute`, `GET /api/v1/actions/executions`, `GET /api/v1/actions/audit` |
| library | `features/library/api/libraryApi.ts` | `GET /api/v1/library/summary|catalog|chunks|document-content|document-file` |
| library | `features/library/api/indexingApi.ts` | `POST /api/v1/index/source|reset|batch/reindex|batch/jobs`, `GET /api/v1/index/batch/jobs[/{id}]`, `POST /api/v1/index/batch/jobs/{id}/retry-failed|cancel` |
| dashboard | `features/dashboard/api/ocpOverviewApi.ts` | `GET /api/v1/ocp/overview/{id}`, `GET /api/v1/ocp/metrics/{id}` |
| resources | `features/resources/api/ocpResourcesApi.ts` | `GET /api/v1/ocp/namespaces/{id}`, `GET /api/v1/ocp/resources/{id}`, `GET /api/v1/ocp/resource-detail/{id}` |

파일은 옮기되 **URL 문자열과 요청/응답 타입은 변경 금지**. 백엔드 `apps/api` 디렉토리는 이 스펙에서 일체 수정하지 않음.

## 8. 롤아웃 단계

### 커밋 1 — 인프라 도입
1. `npm i -D tailwindcss@^3 postcss autoprefixer class-variance-authority clsx tailwind-merge tailwindcss-animate`
2. `npm i lucide-react @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-tabs @radix-ui/react-slot`
3. `npx tailwindcss init -p`, Vite 통합(`postcss.config.js`, `tailwind.config.ts`).
4. `src/styles/globals.css` 작성 — Tailwind directives + 토큰.
5. `src/lib/cn.ts` 추가.
6. shadcn CLI 초기화(또는 수동) → `components/ui/{button,input,card,dialog,badge,tabs,table,dropdown-menu,skeleton,toast}.tsx` 생성 및 다크 토큰에 맞게 variant 튜닝.
7. `main.tsx`에서 `globals.css` import, `<html class="dark">` 보장.
8. 이 시점에 기존 CSS 유지 → 앱 구동 동일.

**검증:** `npm run build` 그린, `npm run dev`로 기존 화면 그대로 뜸.

### 커밋 2 — 폴더 구조 리팩토링
1. `features/<domain>/` 생성하며 `shared/api/*.ts`를 도메인별로 이동.
2. `entities/**`의 타입을 해당 feature의 `types.ts`로 흡수.
3. `app/routes/*/Page.tsx`의 실로직을 `features/<domain>/components/*`로 분해, `pages/*.tsx`는 얇은 래퍼.
4. `ActionsPage`를 `App.tsx` 스위치(`case "actions"`)에 연결, `AppShell`의 `routeMeta`에 Actions 추가.
5. 모든 import 경로 업데이트. 빈 FSD 폴더 삭제.
6. 외관은 기존 CSS 그대로 — 단 클래스명은 기존 partial CSS 클래스를 계속 참조.

**검증:** `npm run build` 그린, 전 페이지 수동 클릭 확인(Connection → Dashboard → Resources → Library → Chat → Actions).

### 커밋 3 — 디자인 시스템 적용
1. `components/layout/AppShell.tsx` 교체 — 좌측 사이드바 + topbar.
2. 각 페이지/피처 컴포넌트를 Tailwind + shadcn 프리미티브로 재구성.
3. 기존 `shared/styles/partials/*.css` 6개 파일 삭제.
4. 루트의 `_shared_head.js`, `_failed_v2.txt` 확인 후 불필요하면 제거.
5. 주요 인터랙션 수동 검증: 연결 플로우, 채팅 스트리밍, 액션 승인/실행, 배치 재인덱스, YAML 에디터.

**검증:** `npm run build` 그린, Playwright 있으면 스모크 테스트, 수동 확인.

## 9. 리스크 & 완화

| 리스크 | 완화 |
|--------|------|
| 커밋 2 이후 외관이 구 CSS인데 구조만 새것 → 중간상태 혼란 | 같은 브랜치에서 커밋 3을 빠르게 쌓아 중간상태 체류시간 최소화. 필요시 커밋 2에서 잠시 멈춤 가능. |
| shadcn 컴포넌트와 기존 CSS 클래스 충돌 | 커밋 1에서 shadcn 파일만 추가(글로벌 스타일 reset 비활성), 실제 적용은 커밋 3에서. |
| 채팅 스트리밍 회귀 | `useCopilotStream` 훅 추출 시 기존 fetch/ReadableStream 로직 그대로 복사, API 응답 형태 변경 없음. |
| Actions 네비 편입 후 RBAC 미체크 경로 | 백엔드가 이미 인증/권한 검증 → 프론트는 그대로 호출, UI는 에러 응답 시 toast로 표시. |

## 10. 완료 기준

- 전 페이지가 다크 테마 + shadcn 프리미티브로 렌더.
- 좌측 사이드바에 6개 라우트(Connection/Dashboard/Resources/Library/Chat/Actions) 모두 노출 및 동작.
- `npm run build` 성공, TypeScript 에러 0.
- 기존에 작동하던 모든 기능(연결, 테스트, 채팅 스트리밍, 프리뷰, 배치 인덱스, YAML 편집, 액션 승인/실행)이 리팩토링 후에도 동작.
- 빈 `.gitkeep` 폴더·사용하지 않는 CSS partial·죽은 entities 슬라이스 제거.
- 백엔드 `apps/api` 변경 없음.

# Frontend Refactor + Helios Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `apps/web`의 FSD 과세분화 구조를 feature 단위로 단순화하고, Tailwind + shadcn/ui 기반 다크 디자인 시스템(Helios 참조)으로 전 페이지를 리스킨한다. 백엔드 API URL은 전부 보존.

**Architecture:** 3단계 커밋 체인. (1) Tailwind/shadcn 인프라 도입(기존 CSS 유지 병존), (2) 폴더 구조를 `pages/` + `features/<domain>/` + `components/{ui,layout}`로 리팩토링(외관 불변), (3) AppShell을 좌측 사이드바 + topbar로 교체하고 페이지를 shadcn 프리미티브로 재구성. 매 단계마다 `npm run build`로 빌드 그린 + 수동 스모크 검증.

**Tech Stack:** React 19, Vite 7, TypeScript 5.8, Tailwind CSS 3, shadcn/ui (Radix UI), Lucide React, class-variance-authority, clsx, tailwind-merge.

**Spec:** `docs/superpowers/specs/2026-04-19-frontend-refactor-helios-design.md`

**Conventions:**
- 모든 신규 파일은 TypeScript + named export.
- Path alias `@/*` → `src/*` (shadcn 관례).
- 백엔드 엔드포인트 URL 문자열 변경 금지.
- 각 Task 종료 시 `npm run build` 그린 확인 후 커밋.

---

## 커밋 1 — 인프라 도입

앱은 기존 그대로 동작해야 한다. 이 커밋 이후 `npm run dev` 시 화면·기능 변경 없음.

### Task 1: 의존성 설치 및 path alias

**Files:**
- Modify: `apps/web/package.json`
- Modify: `apps/web/vite.config.ts`
- Modify: `apps/web/tsconfig.app.json`

- [ ] **Step 1.1: 런타임 의존성 설치**

```bash
cd apps/web
npm i lucide-react class-variance-authority clsx tailwind-merge tailwindcss-animate \
      @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-tabs \
      @radix-ui/react-slot @radix-ui/react-label @radix-ui/react-toast
```

- [ ] **Step 1.2: 개발 의존성 설치**

```bash
cd apps/web
npm i -D tailwindcss@^3 postcss autoprefixer @types/node
```

- [ ] **Step 1.3: `tsconfig.app.json`에 `@/*` path alias 추가**

`compilerOptions`에 추가:
```json
"paths": {
  "@/*": ["./*"]
}
```
기존 `baseUrl: "./src"`는 유지. 즉 `@/components/...`는 `src/components/...`로 해석.

- [ ] **Step 1.4: `vite.config.ts`에 resolve.alias 추가**

파일 상단에:
```ts
import path from "node:path";
```
`defineConfig` 리턴 객체에 추가:
```ts
resolve: {
  alias: {
    "@": path.resolve(__dirname, "./src"),
  },
},
```

- [ ] **Step 1.5: 빌드 검증**

```bash
cd apps/web && npm run build
```
Expected: PASS. 변경이 alias뿐이라 기존 코드 영향 없음.

- [ ] **Step 1.6: 커밋**

```bash
git add apps/web/package.json apps/web/package-lock.json apps/web/vite.config.ts apps/web/tsconfig.app.json
git commit -m "chore(web): Tailwind/shadcn 사전 의존성 및 path alias 추가"
```

---

### Task 2: Tailwind + PostCSS 설정

**Files:**
- Create: `apps/web/tailwind.config.ts`
- Create: `apps/web/postcss.config.js`

- [ ] **Step 2.1: `postcss.config.js` 생성**

```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

- [ ] **Step 2.2: `tailwind.config.ts` 생성**

```ts
import type { Config } from "tailwindcss";
import animate from "tailwindcss-animate";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: { "2xl": "1400px" },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        success: { DEFAULT: "hsl(var(--success))" },
        warning: { DEFAULT: "hsl(var(--warning))" },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        display: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [animate],
};

export default config;
```

- [ ] **Step 2.3: 커밋**

```bash
git add apps/web/tailwind.config.ts apps/web/postcss.config.js
git commit -m "chore(web): Tailwind/PostCSS 설정 추가 (다크 모드 class 전략)"
```

---

### Task 3: globals.css 및 cn 유틸

**Files:**
- Create: `apps/web/src/styles/globals.css`
- Create: `apps/web/src/lib/cn.ts`

- [ ] **Step 3.1: `src/lib/cn.ts` 생성**

```ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

- [ ] **Step 3.2: `src/styles/globals.css` 생성**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 220 14% 10%;       /* #15181e */
    --foreground: 240 3% 94%;        /* #efeff1 */
    --card: 220 14% 12%;
    --card-foreground: 240 3% 94%;
    --popover: 222 17% 9%;
    --popover-foreground: 240 3% 94%;
    --primary: 218 98% 54%;          /* #1060ff */
    --primary-foreground: 0 0% 100%;
    --secondary: 220 14% 16%;
    --secondary-foreground: 240 3% 94%;
    --muted: 220 14% 14%;
    --muted-foreground: 222 8% 58%;  /* #8a8f99 lighter helper */
    --accent: 218 98% 54%;
    --accent-foreground: 0 0% 100%;
    --destructive: 354 59% 38%;
    --destructive-foreground: 0 0% 98%;
    --success: 181 82% 43%;          /* #14c6cb teal */
    --warning: 28 100% 45%;
    --border: 222 8% 44% / 0.25;
    --input: 222 17% 9%;
    --ring: 218 98% 54%;
    --radius: 0.5rem;
  }

  html {
    @apply bg-background text-foreground;
    color-scheme: dark;
  }
  body {
    @apply font-sans antialiased;
    font-feature-settings: "kern" 1, "ss01" 1;
  }
  *:focus-visible {
    @apply outline-none ring-2 ring-ring ring-offset-2 ring-offset-background;
  }
}
```

- [ ] **Step 3.3: 빌드 검증 (globals.css는 아직 import 안 함)**

```bash
cd apps/web && npm run build
```
Expected: PASS.

- [ ] **Step 3.4: 커밋**

```bash
git add apps/web/src/styles/globals.css apps/web/src/lib/cn.ts
git commit -m "chore(web): globals.css 다크 토큰 + cn 유틸 추가"
```

---

### Task 4: shadcn 프리미티브 스캐폴딩 (수동)

shadcn CLI는 네트워크 의존이라 수동 생성. 프로젝트에 10개 프리미티브를 직접 복사한다.

**Files:**
- Create: `apps/web/src/components/ui/button.tsx`
- Create: `apps/web/src/components/ui/input.tsx`
- Create: `apps/web/src/components/ui/card.tsx`
- Create: `apps/web/src/components/ui/dialog.tsx`
- Create: `apps/web/src/components/ui/badge.tsx`
- Create: `apps/web/src/components/ui/tabs.tsx`
- Create: `apps/web/src/components/ui/table.tsx`
- Create: `apps/web/src/components/ui/dropdown-menu.tsx`
- Create: `apps/web/src/components/ui/skeleton.tsx`
- Create: `apps/web/src/components/ui/label.tsx`

- [ ] **Step 4.1: `button.tsx`**

```tsx
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-border bg-transparent hover:bg-secondary hover:text-secondary-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-secondary hover:text-secondary-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-6",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { buttonVariants };
```

- [ ] **Step 4.2: `input.tsx`**

```tsx
import * as React from "react";
import { cn } from "@/lib/cn";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => (
    <input
      type={type}
      ref={ref}
      className={cn(
        "flex h-9 w-full rounded-md border border-border bg-input px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";
```

- [ ] **Step 4.3: `card.tsx`**

```tsx
import * as React from "react";
import { cn } from "@/lib/cn";

export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("rounded-lg border border-border bg-card text-card-foreground shadow-sm", className)} {...props} />
  ),
);
Card.displayName = "Card";

export const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn("flex flex-col space-y-1.5 p-5", className)} {...props} />,
);
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => <h3 ref={ref} className={cn("text-lg font-semibold leading-tight tracking-tight", className)} {...props} />,
);
CardTitle.displayName = "CardTitle";

export const CardDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />,
);
CardDescription.displayName = "CardDescription";

export const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn("p-5 pt-0", className)} {...props} />,
);
CardContent.displayName = "CardContent";

export const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn("flex items-center p-5 pt-0", className)} {...props} />,
);
CardFooter.displayName = "CardFooter";
```

- [ ] **Step 4.4: `dialog.tsx`** (shadcn 표준 구현)

```tsx
import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogPortal = DialogPrimitive.Portal;
export const DialogClose = DialogPrimitive.Close;

export const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      "fixed inset-0 z-50 bg-black/70 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
      className,
    )}
    {...props}
  />
));
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName;

export const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-1/2 top-1/2 z-50 grid w-full max-w-lg -translate-x-1/2 -translate-y-1/2 gap-4 rounded-lg border border-border bg-card p-6 shadow-lg data-[state=open]:animate-in data-[state=closed]:animate-out",
        className,
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring">
        <X className="h-4 w-4" />
        <span className="sr-only">Close</span>
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPortal>
));
DialogContent.displayName = DialogPrimitive.Content.displayName;

export const DialogHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col space-y-1.5 text-left", className)} {...props} />
);
export const DialogFooter = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-row justify-end space-x-2", className)} {...props} />
);
export const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title ref={ref} className={cn("text-lg font-semibold leading-tight", className)} {...props} />
));
DialogTitle.displayName = DialogPrimitive.Title.displayName;
export const DialogDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
));
DialogDescription.displayName = DialogPrimitive.Description.displayName;
```

- [ ] **Step 4.5: `badge.tsx`**

```tsx
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        outline: "border-border text-foreground",
        success: "border-transparent bg-success/15 text-success",
        warning: "border-transparent bg-warning/15 text-warning",
        destructive: "border-transparent bg-destructive text-destructive-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}
export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
export { badgeVariants };
```

- [ ] **Step 4.6: `tabs.tsx`**

```tsx
import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/cn";

export const Tabs = TabsPrimitive.Root;
export const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn("inline-flex h-9 items-center justify-center rounded-md bg-secondary p-1 text-muted-foreground", className)}
    {...props}
  />
));
TabsList.displayName = TabsPrimitive.List.displayName;

export const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow",
      className,
    )}
    {...props}
  />
));
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

export const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content ref={ref} className={cn("mt-4 focus-visible:outline-none", className)} {...props} />
));
TabsContent.displayName = TabsPrimitive.Content.displayName;
```

- [ ] **Step 4.7: `table.tsx`**

```tsx
import * as React from "react";
import { cn } from "@/lib/cn";

export const Table = React.forwardRef<HTMLTableElement, React.HTMLAttributes<HTMLTableElement>>(
  ({ className, ...props }, ref) => (
    <div className="relative w-full overflow-auto">
      <table ref={ref} className={cn("w-full caption-bottom text-sm", className)} {...props} />
    </div>
  ),
);
Table.displayName = "Table";

export const TableHeader = React.forwardRef<HTMLTableSectionElement, React.HTMLAttributes<HTMLTableSectionElement>>(
  ({ className, ...props }, ref) => <thead ref={ref} className={cn("[&_tr]:border-b [&_tr]:border-border", className)} {...props} />,
);
TableHeader.displayName = "TableHeader";

export const TableBody = React.forwardRef<HTMLTableSectionElement, React.HTMLAttributes<HTMLTableSectionElement>>(
  ({ className, ...props }, ref) => <tbody ref={ref} className={cn("[&_tr:last-child]:border-0", className)} {...props} />,
);
TableBody.displayName = "TableBody";

export const TableRow = React.forwardRef<HTMLTableRowElement, React.HTMLAttributes<HTMLTableRowElement>>(
  ({ className, ...props }, ref) => (
    <tr ref={ref} className={cn("border-b border-border transition-colors hover:bg-secondary/40 data-[state=selected]:bg-secondary", className)} {...props} />
  ),
);
TableRow.displayName = "TableRow";

export const TableHead = React.forwardRef<HTMLTableCellElement, React.ThHTMLAttributes<HTMLTableCellElement>>(
  ({ className, ...props }, ref) => (
    <th ref={ref} className={cn("h-10 px-3 text-left align-middle text-xs font-semibold uppercase tracking-wider text-muted-foreground", className)} {...props} />
  ),
);
TableHead.displayName = "TableHead";

export const TableCell = React.forwardRef<HTMLTableCellElement, React.TdHTMLAttributes<HTMLTableCellElement>>(
  ({ className, ...props }, ref) => <td ref={ref} className={cn("p-3 align-middle", className)} {...props} />,
);
TableCell.displayName = "TableCell";
```

- [ ] **Step 4.8: `dropdown-menu.tsx`**

```tsx
import * as React from "react";
import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu";
import { Check, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";

export const DropdownMenu = DropdownMenuPrimitive.Root;
export const DropdownMenuTrigger = DropdownMenuPrimitive.Trigger;
export const DropdownMenuGroup = DropdownMenuPrimitive.Group;
export const DropdownMenuPortal = DropdownMenuPrimitive.Portal;
export const DropdownMenuSeparator = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.Separator ref={ref} className={cn("-mx-1 my-1 h-px bg-border", className)} {...props} />
));
DropdownMenuSeparator.displayName = DropdownMenuPrimitive.Separator.displayName;

export const DropdownMenuContent = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <DropdownMenuPrimitive.Portal>
    <DropdownMenuPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "z-50 min-w-[8rem] overflow-hidden rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-md",
        className,
      )}
      {...props}
    />
  </DropdownMenuPrimitive.Portal>
));
DropdownMenuContent.displayName = DropdownMenuPrimitive.Content.displayName;

export const DropdownMenuItem = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Item> & { inset?: boolean }
>(({ className, inset, ...props }, ref) => (
  <DropdownMenuPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-secondary data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      inset && "pl-8",
      className,
    )}
    {...props}
  />
));
DropdownMenuItem.displayName = DropdownMenuPrimitive.Item.displayName;

export const DropdownMenuLabel = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Label>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.Label ref={ref} className={cn("px-2 py-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground", className)} {...props} />
));
DropdownMenuLabel.displayName = DropdownMenuPrimitive.Label.displayName;
```

- [ ] **Step 4.9: `skeleton.tsx`**

```tsx
import { cn } from "@/lib/cn";

export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("animate-pulse rounded-md bg-secondary", className)} {...props} />;
}
```

- [ ] **Step 4.10: `label.tsx`**

```tsx
import * as React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

const labelVariants = cva("text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70");

export const Label = React.forwardRef<
  React.ElementRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root> & VariantProps<typeof labelVariants>
>(({ className, ...props }, ref) => <LabelPrimitive.Root ref={ref} className={cn(labelVariants(), className)} {...props} />);
Label.displayName = LabelPrimitive.Root.displayName;
```

- [ ] **Step 4.11: 빌드 검증**

```bash
cd apps/web && npm run build
```
Expected: PASS. UI 파일은 아직 import되지 않아 트리쉐이킹됨.

- [ ] **Step 4.12: 커밋**

```bash
git add apps/web/src/components/ui/
git commit -m "feat(web): shadcn/ui 프리미티브 10종 스캐폴딩 (button/input/card/dialog/badge/tabs/table/dropdown/skeleton/label)"
```

---

### Task 5: main.tsx에 globals.css 연결 + Inter 폰트 로드

**Files:**
- Modify: `apps/web/src/main.tsx`
- Modify: `apps/web/index.html`

- [ ] **Step 5.1: 현재 `main.tsx` 확인**

```bash
cat apps/web/src/main.tsx
```
기존 CSS import 구문 확인.

- [ ] **Step 5.2: `main.tsx` 수정 — globals.css import 추가**

기존 `import "./shared/styles/global.css"`(또는 유사) 바로 **아래**에 추가:
```ts
import "./styles/globals.css";
```
기존 import는 삭제하지 않음(커밋 3에서 제거). 두 스타일이 공존하지만 Tailwind는 아직 매칭될 클래스가 없어 영향 없음.

- [ ] **Step 5.3: `index.html`에 Inter 폰트 preconnect + link 추가**

`<head>` 안에 추가:
```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
```
`<html>` 태그에 `class="dark"` 추가:
```html
<html lang="ko" class="dark">
```

- [ ] **Step 5.4: dev 서버로 수동 검증**

```bash
cd apps/web && npm run dev
```
브라우저에서 기존 화면이 **그대로** 렌더되는지 확인(Tailwind는 아직 클래스 매칭 없음, `<body>`에 `bg-background text-foreground`는 적용되지만 기존 CSS가 덮어씀). 콘솔 에러 없음 확인. Ctrl+C로 종료.

- [ ] **Step 5.5: 커밋**

```bash
git add apps/web/src/main.tsx apps/web/index.html
git commit -m "feat(web): globals.css 연결 + Inter 폰트 로드 + dark class"
```

---

## 커밋 2 — 폴더 구조 리팩토링

FSD → feature 기반 구조로 이동. 외관/기능 불변. 기존 `.css` partials는 클래스명 매칭 유지.

### Task 6: 신규 폴더 스켈레톤 + pages 래퍼 전환

**Files:**
- Create: `apps/web/src/pages/` (6 파일)
- Create: `apps/web/src/features/{connection,chat,actions,library,resources,dashboard}/` 하위 스켈레톤

- [ ] **Step 6.1: feature 폴더 생성 (빈 types.ts 파일로 유효한 디렉터리 확보)**

```bash
cd apps/web/src
mkdir -p features/connection/{api,components,hooks} \
         features/chat/{api,components,hooks} \
         features/actions/{api,components,hooks} \
         features/library/{api,components,hooks} \
         features/resources/{api,components,hooks} \
         features/dashboard/{api,components,hooks} \
         pages components/layout
```

- [ ] **Step 6.2: 아무 파일 생성 없이 커밋은 아직 하지 않음 — 다음 Task부터 실제 이동**

---

### Task 7: `connection` feature 이동

**Files:**
- Move: `apps/web/src/features/auth/api/ocpConnectionApi.ts` → `apps/web/src/features/connection/api/ocpConnectionApi.ts`
- Move: `apps/web/src/features/auth/components/OcpConnectionForm.tsx` → `apps/web/src/features/connection/components/OcpConnectionForm.tsx`
- Move: `apps/web/src/features/auth/hooks/useOcpConnection.ts` → `apps/web/src/features/connection/hooks/useOcpConnection.ts`
- Move: `apps/web/src/entities/ocp/connectionTypes.ts` → `apps/web/src/features/connection/types.ts` (내용 merge)
- Move: `apps/web/src/entities/ocp/types.ts` → `apps/web/src/features/connection/types.ts` (append)
- Create: `apps/web/src/pages/ConnectionPage.tsx`
- Delete: `apps/web/src/app/routes/connection/ConnectionPage.tsx`

- [ ] **Step 7.1: 파일 이동 (git mv로)**

```bash
cd apps/web
git mv src/features/auth/api/ocpConnectionApi.ts src/features/connection/api/ocpConnectionApi.ts
git mv src/features/auth/components/OcpConnectionForm.tsx src/features/connection/components/OcpConnectionForm.tsx
git mv src/features/auth/hooks/useOcpConnection.ts src/features/connection/hooks/useOcpConnection.ts
git mv src/features/auth/README.md src/features/connection/README.md 2>/dev/null || true
```

- [ ] **Step 7.2: 타입 통합**

`src/entities/ocp/connectionTypes.ts`와 `src/entities/ocp/types.ts` 내용을 읽고 `src/features/connection/types.ts`에 합쳐 생성. export 이름 유지.

- [ ] **Step 7.3: import 경로 일괄 치환**

grep으로 레퍼런스를 모두 찾는다:
```bash
cd apps/web/src
grep -rn "features/auth/" . | grep -v node_modules
grep -rn "entities/ocp/" . | grep -v node_modules
```
각 참조처에서 경로를 아래로 치환:
- `features/auth/api/ocpConnectionApi` → `@/features/connection/api/ocpConnectionApi`
- `features/auth/components/OcpConnectionForm` → `@/features/connection/components/OcpConnectionForm`
- `features/auth/hooks/useOcpConnection` → `@/features/connection/hooks/useOcpConnection`
- `entities/ocp/connectionTypes` → `@/features/connection/types`
- `entities/ocp/types` → `@/features/connection/types`

- [ ] **Step 7.4: pages/ConnectionPage.tsx 생성 (기존 `app/routes/connection/ConnectionPage.tsx`를 옮김)**

기존 파일 내용 복사 후 import 경로 업데이트. 저장 후 원본 삭제:
```bash
git rm src/app/routes/connection/ConnectionPage.tsx
```

- [ ] **Step 7.5: `app/App.tsx`의 import 업데이트**

```ts
// before
import { ConnectionPage } from "./routes";
// after
import { ConnectionPage } from "@/pages/ConnectionPage";
```
(라우트 index.ts도 함께 정리; `src/app/routes/index.ts`는 아직 존재 — 다음 Task들에서 차차 삭제)

- [ ] **Step 7.6: 빌드 검증**

```bash
cd apps/web && npm run build
```
Expected: PASS. TypeScript 에러 없음.

- [ ] **Step 7.7: 커밋**

```bash
git add -A
git commit -m "refactor(web): auth feature를 connection으로 리네임 + pages로 얇은 래퍼 분리"
```

---

### Task 8: `chat` feature 이동

**Files:**
- Move: `apps/web/src/shared/api/copilotChatApi.ts` → `apps/web/src/features/chat/api/copilotChatApi.ts`
- Move: `apps/web/src/shared/api/ocpLiveChatApi.ts` → `apps/web/src/features/chat/api/ocpLiveChatApi.ts`
- Move: `apps/web/src/shared/api/docsPreviewApi.ts` → `apps/web/src/features/chat/api/docsPreviewApi.ts`
- Move: `apps/web/src/features/chat/components/ChatSessionRail.tsx` — 경로 동일, 내용 유지
- Move: `apps/web/src/features/chat/types.ts` — 경로 동일
- Merge: `apps/web/src/entities/chat/types.ts`, `copilotTypes.ts`, `previewTypes.ts` → `apps/web/src/features/chat/types.ts`
- Create: `apps/web/src/pages/ChatPage.tsx`
- Delete: `apps/web/src/app/routes/chat/ChatPage.tsx`

- [ ] **Step 8.1: API 모듈 이동**

```bash
cd apps/web
git mv src/shared/api/copilotChatApi.ts src/features/chat/api/copilotChatApi.ts
git mv src/shared/api/ocpLiveChatApi.ts src/features/chat/api/ocpLiveChatApi.ts
git mv src/shared/api/docsPreviewApi.ts src/features/chat/api/docsPreviewApi.ts
```

- [ ] **Step 8.2: entities/chat 타입 통합**

`src/entities/chat/types.ts`, `copilotTypes.ts`, `previewTypes.ts` 내용을 `src/features/chat/types.ts`에 append. export 이름 충돌 없으면 그대로, 있으면 prefix로 명확화(예: `ChatMessage`는 유지, `PreviewMessage`는 `DocsPreviewMessage`로).

- [ ] **Step 8.3: `ChatPage.tsx` 이동**

```bash
mkdir -p src/features/chat/components
```
`src/app/routes/chat/ChatPage.tsx`(603줄) 내용을 그대로 `src/pages/ChatPage.tsx`로 복사 — **지금은 통째 복사**가 목적(구조 변경만). 내부 내장 컴포넌트 분해는 커밋 3에서.

- [ ] **Step 8.4: import 경로 일괄 치환**

```bash
cd apps/web/src
grep -rn "shared/api/copilotChatApi\|shared/api/ocpLiveChatApi\|shared/api/docsPreviewApi\|entities/chat/\|app/routes/chat" . | grep -v node_modules
```
각 참조를 다음으로 치환:
- `shared/api/copilotChatApi` → `@/features/chat/api/copilotChatApi`
- `shared/api/ocpLiveChatApi` → `@/features/chat/api/ocpLiveChatApi`
- `shared/api/docsPreviewApi` → `@/features/chat/api/docsPreviewApi`
- `entities/chat/types|copilotTypes|previewTypes` → `@/features/chat/types`
- `app/routes/chat/ChatPage`나 `../routes` 간접참조 → `@/pages/ChatPage`

- [ ] **Step 8.5: 원본 및 entities/chat 삭제**

```bash
git rm src/app/routes/chat/ChatPage.tsx
git rm src/entities/chat/types.ts src/entities/chat/copilotTypes.ts src/entities/chat/previewTypes.ts
```

- [ ] **Step 8.6: App.tsx import 업데이트**

```ts
import { ChatPage } from "@/pages/ChatPage";
```

- [ ] **Step 8.7: 빌드 검증**

```bash
cd apps/web && npm run build
```

- [ ] **Step 8.8: 커밋**

```bash
git add -A
git commit -m "refactor(web): chat feature를 feature 슬라이스로 이동 (내용·API 경로 불변)"
```

---

### Task 9: `actions` feature 이동 + 네비 편입

**Files:**
- Move: `apps/web/src/shared/api/actionPreviewApi.ts` → `apps/web/src/features/actions/api/actionPreviewApi.ts`
- Move: `apps/web/src/app/routes/actions/ActionsPage.tsx` → `apps/web/src/pages/ActionsPage.tsx`
- Merge: `apps/web/src/entities/actions/{types,auditTypes,executionTypes,requestTypes}.ts` → `apps/web/src/features/actions/types.ts`
- Modify: `apps/web/src/app/App.tsx` (case "actions" 추가)
- Modify: `apps/web/src/shared/components/AppShell.tsx` (routeMeta에 Actions 추가, AppRoute 유니온 확장)

- [ ] **Step 9.1: 파일 이동 및 타입 통합**

```bash
cd apps/web
git mv src/shared/api/actionPreviewApi.ts src/features/actions/api/actionPreviewApi.ts
```

`src/entities/actions/*.ts` 네 파일 내용을 `src/features/actions/types.ts`로 통합(export 이름 유지).

`src/app/routes/actions/ActionsPage.tsx`(553줄) 내용을 `src/pages/ActionsPage.tsx`로 복사.

- [ ] **Step 9.2: import 경로 치환**

```bash
grep -rn "shared/api/actionPreviewApi\|entities/actions/\|app/routes/actions" src/ | grep -v node_modules
```
- `shared/api/actionPreviewApi` → `@/features/actions/api/actionPreviewApi`
- `entities/actions/(types|auditTypes|executionTypes|requestTypes)` → `@/features/actions/types`

- [ ] **Step 9.3: AppShell에 Actions 편입**

`src/shared/components/AppShell.tsx:7` 수정:
```ts
export type AppRoute = "connection" | "chat" | "dashboard" | "resources" | "library" | "actions";
```
`routeMeta` 배열에 추가:
```ts
{ key: "actions", label: "Actions" },
```
(순서는 Chat 다음)

- [ ] **Step 9.4: App.tsx에 case 추가**

```tsx
case "actions":
  return <ActionsPage controller={connectionController} onLoadingChange={setPageLoadingState} />;
```
(props 시그니처는 `ActionsPage` 원본에 맞춰 확인. `onLoadingChange` 없으면 생략)

상단 import:
```ts
import { ActionsPage } from "@/pages/ActionsPage";
```

- [ ] **Step 9.5: 원본 삭제**

```bash
git rm src/app/routes/actions/ActionsPage.tsx
git rm src/entities/actions/types.ts src/entities/actions/auditTypes.ts src/entities/actions/executionTypes.ts src/entities/actions/requestTypes.ts
```

- [ ] **Step 9.6: 빌드 + 수동 검증**

```bash
cd apps/web && npm run build && npm run dev
```
브라우저에서 상단 네비에 **Actions** 탭이 나타나는지, 클릭 시 ActionsPage가 렌더되는지, 기존 Requests/Executions/Audit 목록이 동작하는지 확인. Ctrl+C.

- [ ] **Step 9.7: 커밋**

```bash
git add -A
git commit -m "refactor(web): actions feature 이동 + 네비 정식 편입"
```

---

### Task 10: `library` feature 이동

**Files:**
- Move: `apps/web/src/shared/api/libraryApi.ts` → `apps/web/src/features/library/api/libraryApi.ts`
- Move: `apps/web/src/shared/api/indexingApi.ts` → `apps/web/src/features/library/api/indexingApi.ts`
- Move: `apps/web/src/features/library/components/BatchReindexPanel.tsx` — 경로 동일
- Move: `apps/web/src/features/library/hooks/useBatchIndexJob.ts` — 경로 동일
- Merge: `apps/web/src/entities/library/types.ts` + `entities/indexing/types.ts` → `apps/web/src/features/library/types.ts`
- Move: `apps/web/src/app/routes/library/LibraryPage.tsx` → `apps/web/src/pages/LibraryPage.tsx`

- [ ] **Step 10.1: 파일 이동 + 타입 통합 + import 치환**

```bash
cd apps/web
git mv src/shared/api/libraryApi.ts src/features/library/api/libraryApi.ts
git mv src/shared/api/indexingApi.ts src/features/library/api/indexingApi.ts
```
타입 통합 및 LibraryPage 복사는 Task 9와 동일한 패턴.

import 치환:
- `shared/api/libraryApi` → `@/features/library/api/libraryApi`
- `shared/api/indexingApi` → `@/features/library/api/indexingApi`
- `entities/library/types` + `entities/indexing/types` → `@/features/library/types`

- [ ] **Step 10.2: 원본 삭제**

```bash
git rm src/app/routes/library/LibraryPage.tsx
git rm src/entities/library/types.ts src/entities/indexing/types.ts
```

- [ ] **Step 10.3: App.tsx import 업데이트**

```ts
import { LibraryPage } from "@/pages/LibraryPage";
```

- [ ] **Step 10.4: 빌드 검증 + 커밋**

```bash
cd apps/web && npm run build
git add -A
git commit -m "refactor(web): library feature 이동"
```

---

### Task 11: `resources` feature 이동

**Files:**
- Create: `apps/web/src/features/resources/api/ocpResourcesApi.ts` (기존 `shared/api/ocpLiveApi.ts`에서 resource 관련 함수 추출)
- Move: `apps/web/src/features/resources/components/ResourceList.tsx` — 경로 동일
- Move: `apps/web/src/features/resources/components/ResourceYamlEditorModal.tsx` — 경로 동일
- Move: `apps/web/src/app/routes/resources/ResourcesPage.tsx` → `apps/web/src/pages/ResourcesPage.tsx`

- [ ] **Step 11.1: `shared/api/ocpLiveApi.ts` 분할**

파일을 읽어 아래 기준으로 2개로 분리:
- **resources 관련** (namespaces, resources, resource-detail 엔드포인트 함수) → `src/features/resources/api/ocpResourcesApi.ts`
- **dashboard 관련** (overview, metrics 엔드포인트 함수) → `src/features/dashboard/api/ocpOverviewApi.ts` (Task 12에서 생성)
- **공용 타입**(있다면) → 임시로 `src/features/resources/api/ocpResourcesApi.ts` 상단에 두고, Task 12에서 필요시 이동

엔드포인트 URL 문자열·함수 시그니처 변경 금지.

- [ ] **Step 11.2: 원본 `ocpLiveApi.ts` 일단 유지 (Task 12 마무리 시 삭제)**

- [ ] **Step 11.3: ResourcesPage 이동 + import 치환**

```bash
cd apps/web
mv src/app/routes/resources/ResourcesPage.tsx src/pages/ResourcesPage.tsx
git add -A
```
`grep -rn "shared/api/ocpLiveApi" src/`로 참조 확인, Resources 관련 부분만 `@/features/resources/api/ocpResourcesApi`로 변경. Dashboard 관련은 Task 12에서 처리.

- [ ] **Step 11.4: App.tsx import 업데이트**

```ts
import { ResourcesPage } from "@/pages/ResourcesPage";
```

- [ ] **Step 11.5: 빌드 검증 + 커밋**

```bash
cd apps/web && npm run build
git add -A
git commit -m "refactor(web): resources feature 이동 (ocpLiveApi 분할)"
```

---

### Task 12: `dashboard` feature 이동 + ocpLiveApi 완전 분할

**Files:**
- Create: `apps/web/src/features/dashboard/api/ocpOverviewApi.ts`
- Move: `apps/web/src/app/routes/dashboard/DashboardPage.tsx` → `apps/web/src/pages/DashboardPage.tsx`
- Delete: `apps/web/src/shared/api/ocpLiveApi.ts`

- [ ] **Step 12.1: dashboard API 추출**

`shared/api/ocpLiveApi.ts`의 dashboard 함수(overview, metrics)를 `src/features/dashboard/api/ocpOverviewApi.ts`로 이동. 그 외 모든 함수는 Task 11에서 이미 resources로 옮겨졌어야 함.

- [ ] **Step 12.2: DashboardPage 이동 + import 치환**

```bash
cd apps/web
mv src/app/routes/dashboard/DashboardPage.tsx src/pages/DashboardPage.tsx
grep -rn "shared/api/ocpLiveApi" src/ | grep -v node_modules
```
남은 참조를 `@/features/dashboard/api/ocpOverviewApi`로 변경.

- [ ] **Step 12.3: 원본 삭제**

```bash
git rm src/shared/api/ocpLiveApi.ts
```

- [ ] **Step 12.4: App.tsx import 업데이트**

```ts
import { DashboardPage } from "@/pages/DashboardPage";
```

- [ ] **Step 12.5: 빌드 검증 + 커밋**

```bash
cd apps/web && npm run build
git add -A
git commit -m "refactor(web): dashboard feature 이동 + ocpLiveApi 완전 분할"
```

---

### Task 13: 죽은 폴더/파일 정리

**Files:**
- Delete: `apps/web/src/entities/` 전체 (남은 `.gitkeep`만)
- Delete: `apps/web/src/features/auth/`, `features/dashboard/.gitkeep`, `features/guided-ask/`, `features/resources/.gitkeep` 등 빈 FSD 잔재
- Delete: `apps/web/src/app/` 전체 (App.tsx는 `src/App.tsx`로 이동)
- Delete: `apps/web/src/shared/api/` (비어있으면)

- [ ] **Step 13.1: App.tsx 이동**

```bash
cd apps/web
git mv src/app/App.tsx src/App.tsx
```
`main.tsx`의 `import { App } from "./app/App"`를 `import { App } from "./App"`로 수정.

- [ ] **Step 13.2: 빈 폴더·잔여 파일 제거**

```bash
cd apps/web/src
# entities 전체 제거 (모든 타입이 feature로 이동 완료)
find entities -type f | xargs git rm 2>/dev/null || true
rm -rf entities

# app 폴더 제거 (App.tsx 이동 후 routes/만 남음)
find app -type f | xargs git rm 2>/dev/null || true
rm -rf app

# features 잔해
for d in auth guided-ask; do
  find "features/$d" -type f 2>/dev/null | xargs git rm 2>/dev/null || true
  rm -rf "features/$d"
done
# 각 feature의 .gitkeep만 남은 경우
find features -name ".gitkeep" | xargs git rm 2>/dev/null || true

# shared/api 비었으면 제거
rmdir shared/api 2>/dev/null || true
```

- [ ] **Step 13.3: 빌드 검증**

```bash
cd apps/web && npm run build
```
Expected: PASS.

- [ ] **Step 13.4: 수동 스모크**

```bash
npm run dev
```
브라우저에서 Connection → Dashboard → Resources → Library → Chat → Actions 6개 라우트 모두 클릭. 기능 동일 확인.

- [ ] **Step 13.5: 커밋**

```bash
git add -A
git commit -m "refactor(web): entities/app/FSD 잔재 제거, App.tsx 루트 승격"
```

---

## 커밋 3 — 디자인 시스템 적용

이 시점부터 외관을 Helios 스타일로 재구성한다. `shared/styles/partials/*.css`는 점진적으로 교체 후 최종 삭제.

### Task 14: AppShell → Sidebar + TopBar 교체

**Files:**
- Create: `apps/web/src/components/layout/AppShell.tsx`
- Create: `apps/web/src/components/layout/Sidebar.tsx`
- Create: `apps/web/src/components/layout/TopBar.tsx`
- Create: `apps/web/src/components/layout/PageHeader.tsx`
- Modify: `apps/web/src/App.tsx` (import 교체)
- Delete: `apps/web/src/shared/components/AppShell.tsx` (마지막 Step에서)

- [ ] **Step 14.1: `Sidebar.tsx` 생성**

```tsx
import { Activity, Boxes, Database, LayoutDashboard, MessageSquare, Plug, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";
import { Badge } from "@/components/ui/badge";
import type { AppRoute } from "@/components/layout/AppShell";

const NAV: Array<{ key: AppRoute; label: string; icon: LucideIcon }> = [
  { key: "connection", label: "Connection", icon: Plug },
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "resources", label: "Resources", icon: Boxes },
  { key: "library", label: "Library", icon: Database },
  { key: "chat", label: "Chat", icon: MessageSquare },
  { key: "actions", label: "Actions", icon: Activity },
];

type Props = {
  active: AppRoute;
  onNavigate: (r: AppRoute) => void;
  profileName: string;
  clusterUrl: string | null;
  status: string;
  onDisconnect?: () => void;
};

export function Sidebar({ active, onNavigate, profileName, clusterUrl, status, onDisconnect }: Props) {
  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col border-r border-border bg-[hsl(222_17%_7%)]">
      <div className="flex h-14 items-center gap-2 border-b border-border px-5">
        <div className="grid h-8 w-8 place-items-center rounded-md bg-primary/15 text-sm font-bold text-primary">K</div>
        <div className="font-display text-base font-semibold tracking-tight">OCPOps</div>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {NAV.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.key;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onNavigate(item.key)}
              className={cn(
                "group relative flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
              )}
            >
              {isActive && <span className="absolute left-0 top-1.5 h-5 w-1 rounded-r bg-primary" />}
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="border-t border-border p-4">
        <div className="flex items-start gap-3">
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-primary/15 text-sm font-semibold text-primary">
            {(profileName[0] ?? "U").toUpperCase()}
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-medium">{profileName}</div>
            <div className="truncate text-xs text-muted-foreground">{clusterUrl ?? "Not connected"}</div>
            <div className="mt-2 flex items-center gap-2">
              <Badge variant={status === "connected" ? "success" : "outline"}>{status}</Badge>
              {onDisconnect && (
                <button type="button" onClick={onDisconnect} className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground">
                  Disconnect
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
```

- [ ] **Step 14.2: `TopBar.tsx` 생성**

```tsx
import { cn } from "@/lib/cn";
import { Badge } from "@/components/ui/badge";

type Props = {
  title: string;
  subtitle?: string;
  connectionState: "connected" | "idle" | "error";
  statusMessage?: string;
};

export function TopBar({ title, subtitle, connectionState, statusMessage }: Props) {
  const pillVariant = connectionState === "connected" ? "success" : connectionState === "error" ? "destructive" : "outline";
  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-[hsl(220_14%_10%)]/80 px-8 backdrop-blur">
      <div className="flex items-baseline gap-3">
        <h1 className="font-display text-lg font-semibold leading-none tracking-tight">{title}</h1>
        {subtitle && <span className="text-sm text-muted-foreground">{subtitle}</span>}
      </div>
      <div className="flex items-center gap-3">
        {statusMessage && <span className="text-xs text-muted-foreground">{statusMessage}</span>}
        <Badge variant={pillVariant} className={cn("px-2 py-0.5 text-xs")}>{connectionState}</Badge>
      </div>
    </header>
  );
}
```

- [ ] **Step 14.3: `PageHeader.tsx` 생성**

```tsx
import { cn } from "@/lib/cn";
import type { ReactNode } from "react";

type Props = {
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
};

export function PageHeader({ title, description, actions, className }: Props) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-4 pb-6", className)}>
      <div>
        <h2 className="font-display text-2xl font-semibold tracking-tight">{title}</h2>
        {description && <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
```

- [ ] **Step 14.4: `AppShell.tsx` 생성 (래퍼)**

```tsx
import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

export type AppRoute = "connection" | "chat" | "dashboard" | "resources" | "library" | "actions";

type Props = {
  active: AppRoute;
  onNavigate: (r: AppRoute) => void;
  title: string;
  subtitle?: string;
  connectionState: "connected" | "idle" | "error";
  statusMessage?: string;
  profileName: string;
  clusterUrl: string | null;
  onDisconnect?: () => void;
  rightRail?: ReactNode;
  children: ReactNode;
};

export function AppShell({ active, onNavigate, title, subtitle, connectionState, statusMessage, profileName, clusterUrl, onDisconnect, rightRail, children }: Props) {
  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar
        active={active}
        onNavigate={onNavigate}
        profileName={profileName}
        clusterUrl={clusterUrl}
        status={connectionState}
        onDisconnect={onDisconnect}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar title={title} subtitle={subtitle} connectionState={connectionState} statusMessage={statusMessage} />
        <div className="flex min-h-0 flex-1">
          <main className="min-w-0 flex-1 overflow-y-auto px-8 py-6">{children}</main>
          {rightRail && <aside className="hidden w-72 shrink-0 overflow-y-auto border-l border-border bg-[hsl(222_17%_7%)] p-4 xl:block">{rightRail}</aside>}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 14.5: `App.tsx` 전면 교체**

기존 `App.tsx`를 아래 구조로 교체. 기존 상태 로직(chatSessions, loadingState 등)은 그대로 유지하되, `AppShell` props를 새 시그니처에 맞춘다. 라우트별로 `pageTitle`/`connectionState` 계산:

```tsx
import { useEffect, useMemo, useState } from "react";
import { AppShell, type AppRoute } from "@/components/layout/AppShell";
import { ConnectionPage } from "@/pages/ConnectionPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { ResourcesPage } from "@/pages/ResourcesPage";
import { LibraryPage } from "@/pages/LibraryPage";
import { ChatPage } from "@/pages/ChatPage";
import { ActionsPage } from "@/pages/ActionsPage";
import { useOcpConnection } from "@/features/connection/hooks/useOcpConnection";
import { ChatSessionRail } from "@/features/chat/components/ChatSessionRail";
import { createChatSessionRecord, type ChatSessionRecord } from "@/features/chat/types";

const CHAT_SESSIONS_STORAGE_KEY = "rag-task.chat.sessions";

const ROUTE_META: Record<AppRoute, { title: string; subtitle?: string }> = {
  connection: { title: "Connection", subtitle: "OCP 클러스터 연결 관리" },
  dashboard: { title: "Dashboard", subtitle: "클러스터 상태 요약" },
  resources: { title: "Resources", subtitle: "네임스페이스 및 리소스 탐색" },
  library: { title: "Library", subtitle: "문서 코퍼스 · 배치 인덱싱" },
  chat: { title: "Chat", subtitle: "RAG 기반 운영 어시스턴트" },
  actions: { title: "Actions", subtitle: "OCP 액션 프리뷰 · 승인 · 감사" },
};

export function App() {
  const controller = useOcpConnection();
  const [route, setRoute] = useState<AppRoute>("connection");
  const [chatSessions, setChatSessions] = useState<ChatSessionRecord[]>(() => loadChatSessions());
  const [activeChatSessionId, setActiveChatSessionId] = useState("");
  const [pageMessage, setPageMessage] = useState<string>("");

  useEffect(() => {
    window.localStorage.setItem(CHAT_SESSIONS_STORAGE_KEY, JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    if (!activeChatSessionId && chatSessions[0]) setActiveChatSessionId(chatSessions[0].id);
  }, [activeChatSessionId, chatSessions]);

  const activeChatSession = useMemo(
    () => chatSessions.find((s) => s.id === activeChatSessionId) ?? chatSessions[0] ?? createChatSessionRecord(),
    [activeChatSessionId, chatSessions],
  );

  const connectionState: "connected" | "idle" | "error" = controller.testResult?.ok
    ? "connected"
    : controller.submitState === "error"
      ? "error"
      : "idle";

  const profileName = controller.testResult?.resolvedUser ?? controller.profile?.displayName ?? "Not connected";

  const pageNode = useMemo(() => {
    switch (route) {
      case "chat":
        return (
          <ChatPage
            controller={controller}
            session={activeChatSession}
            updateSession={(id, u) => setChatSessions((cur) => cur.map((s) => (s.id === id ? u(s) : s)))}
          />
        );
      case "dashboard":
        return <DashboardPage controller={controller} onLoadingChange={(s) => setPageMessage(s.detail ?? "")} />;
      case "resources":
        return <ResourcesPage controller={controller} onLoadingChange={(s) => setPageMessage(s.detail ?? "")} />;
      case "library":
        return <LibraryPage controller={controller} onLoadingChange={(s) => setPageMessage(s.detail ?? "")} />;
      case "actions":
        return <ActionsPage controller={controller} onLoadingChange={(s) => setPageMessage(s.detail ?? "")} />;
      case "connection":
      default:
        return <ConnectionPage controller={controller} />;
    }
  }, [activeChatSession, controller, route]);

  const rightRail = route === "chat" ? (
    <ChatSessionRail
      sessions={chatSessions}
      activeSessionId={activeChatSession.id}
      onSelect={(id) => setActiveChatSessionId(id)}
      onCreate={() => {
        const next = createChatSessionRecord();
        setChatSessions((cur) => [next, ...cur]);
        setActiveChatSessionId(next.id);
      }}
      onRemove={(id) =>
        setChatSessions((cur) => {
          const next = cur.filter((s) => s.id !== id);
          return next.length > 0 ? next : [createChatSessionRecord()];
        })
      }
    />
  ) : null;

  return (
    <AppShell
      active={route}
      onNavigate={setRoute}
      title={ROUTE_META[route].title}
      subtitle={ROUTE_META[route].subtitle}
      connectionState={connectionState}
      statusMessage={controller.message || pageMessage}
      profileName={profileName}
      clusterUrl={controller.profile?.clusterUrl ?? null}
      onDisconnect={controller.testResult?.ok ? () => controller.disconnect?.() : undefined}
      rightRail={rightRail}
    >
      {pageNode}
    </AppShell>
  );
}

function loadChatSessions(): ChatSessionRecord[] {
  if (typeof window === "undefined") return [createChatSessionRecord()];
  try {
    const raw = window.localStorage.getItem(CHAT_SESSIONS_STORAGE_KEY);
    if (!raw) return [createChatSessionRecord()];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) && parsed.length > 0 ? (parsed as ChatSessionRecord[]) : [createChatSessionRecord()];
  } catch {
    return [createChatSessionRecord()];
  }
}
```

**주의:** 페이지 props 시그니처가 기존과 맞지 않으면 페이지 쪽을 이 시그니처에 맞춰 조정(기존에 `AppRoute` 타입을 참조하던 곳은 `@/components/layout/AppShell`에서 import).

- [ ] **Step 14.6: 기존 AppShell 삭제**

```bash
git rm src/shared/components/AppShell.tsx
```

- [ ] **Step 14.7: 빌드 검증 + 수동 확인**

```bash
cd apps/web && npm run build && npm run dev
```
브라우저에서 좌측 사이드바 + 상단 topbar가 뜨고, 6개 라우트 클릭 시 각 페이지가 렌더되며 기존 기능이 동작하는지 확인. 사이드바 하단 프로필 카드, 연결 상태 배지 노출 확인. Ctrl+C.

- [ ] **Step 14.8: 커밋**

```bash
git add -A
git commit -m "feat(web): 좌측 사이드바 + 얇은 topbar AppShell 적용"
```

---

### Task 15: ConnectionPage Helios 재구성

**Files:**
- Modify: `apps/web/src/pages/ConnectionPage.tsx`
- Modify: `apps/web/src/features/connection/components/OcpConnectionForm.tsx`

- [ ] **Step 15.1: `ConnectionPage.tsx` 재작성**

기존 페이지를 히어로 스타일 + 중앙 Card 폼으로 변경:

```tsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { OcpConnectionForm } from "@/features/connection/components/OcpConnectionForm";
import type { useOcpConnection } from "@/features/connection/hooks/useOcpConnection";

type Props = { controller: ReturnType<typeof useOcpConnection> };

export function ConnectionPage({ controller }: Props) {
  return (
    <div className="mx-auto max-w-3xl py-10">
      <div className="mb-10 text-center">
        <div className="mb-3 inline-flex rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold uppercase tracking-widest text-primary">
          OCP Operator Console
        </div>
        <h2 className="font-display text-4xl font-semibold tracking-tight">Connect your OpenShift cluster</h2>
        <p className="mx-auto mt-3 max-w-xl text-sm text-muted-foreground">
          URL과 자격증명만 있으면 3초 안에 상태·RBAC·리스(lease)까지 검증합니다. 모든 작업은 감사 로그에 남고, 채팅과 Actions는 이 연결을 공유합니다.
        </p>
      </div>

      <Card className="border-border/60 shadow-lg">
        <CardHeader>
          <CardTitle>Cluster profile</CardTitle>
          <CardDescription>
            Server URL, 인증 방식, 기본 네임스페이스를 입력하세요. 연결 성공 시 Dashboard로 자동 전환됩니다.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <OcpConnectionForm controller={controller} />
        </CardContent>
      </Card>
    </div>
  );
}
```

- [ ] **Step 15.2: `OcpConnectionForm.tsx` 리스킨**

기존 폼 JSX를 유지하되 모든 native `<input>`을 `<Input>`, `<button>`을 `<Button>`으로, 레이블은 `<Label>`로 교체. `className`은 제거하고 shadcn 기본 스타일 사용. 폼 레이아웃은 `space-y-4`, 버튼 그룹은 `flex gap-2 justify-end`.

핸들러 함수/state/hook 호출은 **절대 변경 금지** — 시각만 교체.

- [ ] **Step 15.3: 빌드 + 수동 확인**

```bash
cd apps/web && npm run build && npm run dev
```
Connection 페이지에서 폼 렌더, Connect 버튼 클릭 시 기존과 동일하게 동작 확인. Ctrl+C.

- [ ] **Step 15.4: 커밋**

```bash
git add -A
git commit -m "feat(web): ConnectionPage Helios 히어로 + shadcn 폼 적용"
```

---

### Task 16: DashboardPage 재구성

**Files:**
- Modify: `apps/web/src/pages/DashboardPage.tsx`

- [ ] **Step 16.1: DashboardPage를 `MetricGrid` → shadcn Card 4개로 교체**

기존 페이지의 데이터 fetch 로직(useEffect + ocpOverviewApi 호출)은 유지. 렌더 부분만 다음 패턴으로 변경:

```tsx
import { PageHeader } from "@/components/layout/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

// ... 기존 state + fetch 유지

return (
  <>
    <PageHeader
      title="Cluster overview"
      description="연결된 클러스터의 노드·네임스페이스·리소스 밀도와 시계열 메트릭."
    />

    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
      {[
        { label: "Nodes", value: overview?.nodeCount, unit: "" },
        { label: "Namespaces", value: overview?.namespaceCount, unit: "" },
        { label: "Pods", value: overview?.podCount, unit: "" },
        { label: "Services", value: overview?.serviceCount, unit: "" },
      ].map((m) => (
        <Card key={m.label}>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{m.label}</CardTitle>
          </CardHeader>
          <CardContent>
            {m.value == null ? <Skeleton className="h-8 w-16" /> : <div className="font-display text-3xl font-semibold tabular-nums">{m.value}{m.unit}</div>}
          </CardContent>
        </Card>
      ))}
    </div>

    <Card className="mt-6">
      <CardHeader>
        <CardTitle>Activity</CardTitle>
      </CardHeader>
      <CardContent>
        {/* 기존 MetricLineChart 그대로 — className만 제거 */}
      </CardContent>
    </Card>
  </>
);
```

`overview` 객체 키(`nodeCount` 등)는 **실제 API 응답 필드명**에 맞춰 조정. 원본 DashboardPage를 확인하고 존재하지 않는 필드는 교체.

- [ ] **Step 16.2: 빌드 + 수동 확인 + 커밋**

```bash
cd apps/web && npm run build
git add -A && git commit -m "feat(web): DashboardPage shadcn Card 그리드 + 메트릭 정리"
```

---

### Task 17: ResourcesPage 재구성

**Files:**
- Modify: `apps/web/src/pages/ResourcesPage.tsx`
- Modify: `apps/web/src/features/resources/components/ResourceList.tsx`
- Modify: `apps/web/src/features/resources/components/ResourceYamlEditorModal.tsx`

- [ ] **Step 17.1: 리소스 테이블을 shadcn Table로 교체**

`ResourceList.tsx`에서 `<table>`을 `<Table>` / `<TableHeader>` / `<TableBody>` / `<TableRow>` / `<TableHead>` / `<TableCell>`로 치환. 기존 데이터/필터 로직 유지.

네임스페이스 선택은 상단 `<DropdownMenu>`로 — 기존 `<select>`가 있다면 교체:
```tsx
<DropdownMenu>
  <DropdownMenuTrigger asChild>
    <Button variant="outline">Namespace: {current}</Button>
  </DropdownMenuTrigger>
  <DropdownMenuContent>
    {namespaces.map((ns) => (
      <DropdownMenuItem key={ns} onSelect={() => selectNamespace(ns)}>{ns}</DropdownMenuItem>
    ))}
  </DropdownMenuContent>
</DropdownMenu>
```

- [ ] **Step 17.2: YAML 에디터 모달을 shadcn Dialog로 교체**

`ResourceYamlEditorModal.tsx`의 최상위 div/포털을 `<Dialog>` + `<DialogContent>`로 교체. textarea 영역은 Tailwind 클래스로 `font-mono text-xs` + 고정 높이 `h-[60vh]`. 저장/취소 버튼은 `<DialogFooter>`에 `<Button>` 2개.

```tsx
<Dialog open={open} onOpenChange={setOpen}>
  <DialogContent className="max-w-3xl">
    <DialogHeader>
      <DialogTitle>{resourceKind} / {resourceName}</DialogTitle>
      <DialogDescription>YAML을 편집하면 변경 사항은 Dry-Run 후 적용됩니다.</DialogDescription>
    </DialogHeader>
    <textarea
      value={yaml}
      onChange={(e) => setYaml(e.target.value)}
      className="h-[60vh] w-full resize-none rounded-md border border-border bg-input p-3 font-mono text-xs"
    />
    <DialogFooter>
      <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
      <Button onClick={applyYaml}>Apply</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
```

- [ ] **Step 17.3: `ResourcesPage.tsx` 헤더 + 레이아웃 정리**

```tsx
<PageHeader title="Resources" description="네임스페이스별 리소스를 탐색하고 YAML을 편집합니다." actions={namespaceDropdown} />
<ResourceList ... />
<ResourceYamlEditorModal ... />
```

- [ ] **Step 17.4: 빌드 + 수동 확인 + 커밋**

```bash
cd apps/web && npm run build
git add -A && git commit -m "feat(web): ResourcesPage shadcn Table + Dialog 적용"
```

---

### Task 18: LibraryPage 재구성

**Files:**
- Modify: `apps/web/src/pages/LibraryPage.tsx`
- Modify: `apps/web/src/features/library/components/BatchReindexPanel.tsx`

- [ ] **Step 18.1: 최상단을 Tabs로 분할**

```tsx
<PageHeader title="Library" description="문서 코퍼스와 배치 인덱싱 운영." />
<Tabs defaultValue="summary" className="w-full">
  <TabsList>
    <TabsTrigger value="summary">Summary</TabsTrigger>
    <TabsTrigger value="catalog">Catalog</TabsTrigger>
    <TabsTrigger value="batch">Batch Jobs</TabsTrigger>
  </TabsList>
  <TabsContent value="summary">{/* 기존 summary 영역 */}</TabsContent>
  <TabsContent value="catalog">{/* 기존 catalog 테이블 → shadcn Table */}</TabsContent>
  <TabsContent value="batch"><BatchReindexPanel ... /></TabsContent>
</Tabs>
```

- [ ] **Step 18.2: `BatchReindexPanel` 내부를 Card + Progress 유사 UI로 정리**

기존 상태 업데이트 로직(`useBatchIndexJob`) 유지. 각 진행 중 잡을 `<Card>`로 감싸고, progress는 Tailwind bar:
```tsx
<div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
  <div className="h-full bg-primary transition-all" style={{ width: `${(done / total) * 100}%` }} />
</div>
```

- [ ] **Step 18.3: 카탈로그 테이블 + 청크 Dialog**

Catalog 테이블 행 클릭 시 청크 뷰 Dialog 열기. 기존 `docsPreviewApi` 호출 로직 유지.

- [ ] **Step 18.4: 빌드 + 수동 확인 + 커밋**

```bash
cd apps/web && npm run build
git add -A && git commit -m "feat(web): LibraryPage Tabs + 배치잡 Card + 청크 Dialog 적용"
```

---

### Task 19: ChatPage 재구성

**Files:**
- Modify: `apps/web/src/pages/ChatPage.tsx`
- Split: `ChatPage` 내부 대형 JSX를 `features/chat/components/ChatTranscript.tsx`, `features/chat/components/ChatComposer.tsx`로 분해

- [ ] **Step 19.1: ChatTranscript 추출**

`ChatPage.tsx`의 메시지 목록 렌더 부분(약 200줄)을 `src/features/chat/components/ChatTranscript.tsx`로 이동. props는 `messages`, `onCitationClick`, `isStreaming`. JSX는 shadcn `Card` 외곽 + 내부 스크롤 영역.

각 메시지 버블:
```tsx
<div className={cn("flex gap-3", message.role === "user" ? "justify-end" : "")}>
  {message.role !== "user" && <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-primary/15 text-xs font-semibold text-primary">AI</div>}
  <div className={cn("max-w-[85%] rounded-lg border border-border px-4 py-3 text-sm leading-relaxed", message.role === "user" ? "bg-primary text-primary-foreground" : "bg-card")}>
    {/* markdown 렌더는 기존 MarkdownArticle 그대로 */}
  </div>
</div>
```

스트리밍 표시는 마지막 AI 메시지 아래에 `<div className="flex gap-1"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted-foreground" /><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted-foreground [animation-delay:150ms]" /><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted-foreground [animation-delay:300ms]" /></div>`.

- [ ] **Step 19.2: ChatComposer 추출**

하단 입력창을 `src/features/chat/components/ChatComposer.tsx`로 분리. props는 `onSubmit`, `disabled`, `placeholder`. textarea는 `<textarea>` + Tailwind, 제출 버튼은 `<Button>` 오른쪽 하단.

```tsx
<form onSubmit={handleSubmit} className="flex items-end gap-2 rounded-lg border border-border bg-card p-3">
  <textarea
    value={value}
    onChange={(e) => setValue(e.target.value)}
    rows={2}
    placeholder={placeholder}
    disabled={disabled}
    className="min-h-[40px] flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground"
  />
  <Button type="submit" size="sm" disabled={disabled || !value.trim()}>Send</Button>
</form>
```

- [ ] **Step 19.3: ChatSessionRail 리스킨**

기존 `ChatSessionRail.tsx`의 JSX를 shadcn 스타일로 교체. 세션 항목은 버튼 리스트 + 활성 상태 좌측 막대, 새 세션 버튼은 `<Button variant="outline" size="sm" className="w-full">+ New session</Button>`.

- [ ] **Step 19.4: ChatPage 최상위 레이아웃**

```tsx
<div className="flex h-[calc(100vh-3.5rem)] flex-col gap-4">
  <PageHeader title="Chat" description="RAG 기반 어시스턴트 — 인용과 OCP 액션 제안 포함." />
  <ChatTranscript messages={session.messages} isStreaming={isStreaming} onCitationClick={openPreview} />
  <ChatComposer onSubmit={sendMessage} disabled={isStreaming} placeholder="무엇을 도와드릴까요?" />
</div>
```

스트리밍 로직(`copilotChatApi.stream()` 호출, `ReadableStream` 소비)은 **변경 없이 유지**.

- [ ] **Step 19.5: 빌드 + 수동 확인**

채팅 전송 → 스트리밍 응답 → 인용 프리뷰 Dialog까지 전부 동작하는지 확인.

- [ ] **Step 19.6: 커밋**

```bash
git add -A
git commit -m "feat(web): ChatPage를 Transcript/Composer/Rail로 분해 + shadcn 적용"
```

---

### Task 20: ActionsPage 재구성

**Files:**
- Modify: `apps/web/src/pages/ActionsPage.tsx`
- Split: 필요 시 `features/actions/components/{RequestsTable,ExecutionsTable,AuditTable,ActionPreviewDialog}.tsx`

- [ ] **Step 20.1: Tabs 구조로 분할**

```tsx
<PageHeader title="Actions" description="OCP 액션 프리뷰 · 요청 · 실행 · 감사." />
<Tabs defaultValue="requests">
  <TabsList>
    <TabsTrigger value="requests">Requests</TabsTrigger>
    <TabsTrigger value="executions">Executions</TabsTrigger>
    <TabsTrigger value="audit">Audit</TabsTrigger>
  </TabsList>
  <TabsContent value="requests"><RequestsTable ... /></TabsContent>
  <TabsContent value="executions"><ExecutionsTable ... /></TabsContent>
  <TabsContent value="audit"><AuditTable ... /></TabsContent>
</Tabs>
```

기존 553줄 ActionsPage 내부를 3개의 테이블 컴포넌트로 분해. 각 테이블은 shadcn `Table`. 행 액션(approve/reject/execute)은 `<DropdownMenu>` 트리거 + 확인 `<Dialog>`.

- [ ] **Step 20.2: ActionPreviewDialog**

프리뷰 모달을 shadcn Dialog로:
```tsx
<Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
  <DialogContent className="max-w-2xl">
    <DialogHeader>
      <DialogTitle>Action preview</DialogTitle>
      <DialogDescription>YAML 변경 사항과 예상 영향을 확인한 뒤 Request를 생성합니다.</DialogDescription>
    </DialogHeader>
    <pre className="max-h-[50vh] overflow-auto rounded-md border border-border bg-input p-3 text-xs font-mono">{previewYaml}</pre>
    <DialogFooter>
      <Button variant="outline" onClick={() => setPreviewOpen(false)}>Close</Button>
      <Button onClick={submitRequest}>Create request</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
```

API 호출(`actionPreviewApi.*`)은 시그니처 변경 없음.

- [ ] **Step 20.3: 빌드 + 수동 확인**

Requests 탭에서 Preview → Create request → Approve → Execute 한 사이클 동작 확인.

- [ ] **Step 20.4: 커밋**

```bash
git add -A
git commit -m "feat(web): ActionsPage Tabs + shadcn Table/Dialog/DropdownMenu 적용"
```

---

### Task 21: 기존 CSS partial 제거 및 최종 정리

**Files:**
- Delete: `apps/web/src/shared/styles/global.css`
- Delete: `apps/web/src/shared/styles/partials/*.css` (6개)
- Delete: `apps/web/src/shared/components/{AppLoadingOverlay,MarkdownArticle,MetricGrid,MetricLineChart,PageHero,StatusNotice,SurfaceCard}.tsx` 중 더 이상 참조되지 않는 것
- Delete: 루트 `_shared_head.js`, `_failed_v2.txt` (파일 내용 확인 후)

- [ ] **Step 21.1: 참조 없는 `shared/components` 파일 식별**

```bash
cd apps/web/src
for f in shared/components/*.tsx; do
  name=$(basename "$f" .tsx)
  count=$(grep -rn "$name" . --include="*.tsx" --include="*.ts" | grep -v "shared/components/$name.tsx" | wc -l)
  echo "$name: $count refs"
done
```
카운트 0인 컴포넌트는 `git rm`.

- [ ] **Step 21.2: MarkdownArticle은 ChatTranscript에서 계속 사용 → `features/chat/components/MarkdownArticle.tsx`로 이동**

```bash
git mv src/shared/components/MarkdownArticle.tsx src/features/chat/components/MarkdownArticle.tsx
```
import 경로 업데이트.

- [ ] **Step 21.3: 나머지 shared 컴포넌트 제거**

```bash
git rm src/shared/components/{AppLoadingOverlay,MetricGrid,MetricLineChart,PageHero,StatusNotice,SurfaceCard}.tsx 2>/dev/null || true
```
(존재 여부에 따라)

- [ ] **Step 21.4: 기존 CSS partial 전부 제거**

```bash
git rm src/shared/styles/global.css
git rm src/shared/styles/partials/*.css
rmdir src/shared/styles/partials src/shared/styles 2>/dev/null || true
```
`main.tsx`에서 기존 global.css import 라인 제거.

- [ ] **Step 21.5: `shared/lib/useCompactLayout.ts` 사용처 확인**

```bash
grep -rn "useCompactLayout" src/
```
새 AppShell은 반응형을 Tailwind `xl:block` 등으로 처리하므로 미사용이면:
```bash
git rm src/shared/lib/useCompactLayout.ts
rmdir src/shared/lib 2>/dev/null || true
```

- [ ] **Step 21.6: `shared/` 폴더 전체 제거 확인**

```bash
ls src/shared/ 2>/dev/null
rmdir src/shared 2>/dev/null || true
```
비었으면 자동 삭제됨.

- [ ] **Step 21.7: 루트 스크래치 파일 확인**

```bash
head -5 _shared_head.js _failed_v2.txt
```
내용이 이전 버전의 스크래치/실패 로그라면 삭제:
```bash
git rm _shared_head.js _failed_v2.txt
```
(실제 의미 있는 파일이면 이 Step 건너뛰기)

- [ ] **Step 21.8: 빌드 + 수동 전체 검증**

```bash
cd apps/web && npm run build && npm run dev
```
6개 페이지 모두 정상 렌더, 모든 인터랙션(연결·테스트·채팅 스트리밍·리소스 탐색·YAML 편집·배치 인덱스·액션 승인/실행) 수동 검증.

- [ ] **Step 21.9: 최종 커밋**

```bash
git add -A
git commit -m "refactor(web): 구 CSS partials 및 사용처 없는 shared 컴포넌트 제거"
```

---

### Task 22: Playwright 스모크 테스트 (선택)

**Files:**
- Check: `apps/web/tests/e2e/` — 기존 테스트 확인, 있으면 실행

- [ ] **Step 22.1: 기존 e2e 확인**

```bash
ls apps/web/tests/e2e/
```
비어있으면 이 Task 건너뛰기. 테스트가 있으면:

- [ ] **Step 22.2: 실행 + 실패 시 셀렉터 업데이트**

```bash
cd apps/web && npx playwright test
```
새 디자인의 셀렉터에 맞춰 테스트 수정(텍스트 기반 쿼리 `getByText("Connect")` 등은 대부분 유지됨).

- [ ] **Step 22.3: 통과 시 커밋 (변경 있는 경우)**

```bash
git add -A
git commit -m "test(web): e2e 셀렉터를 신규 AppShell에 맞춰 업데이트"
```

---

## 최종 검증 체크리스트

3단계 커밋 완료 후:

- [ ] `npm run build` 그린, TypeScript 에러 0
- [ ] 6개 라우트(Connection, Dashboard, Resources, Library, Chat, Actions) 모두 사이드바에서 접근 가능
- [ ] Connection → Test → Disconnect 플로우 동작
- [ ] Chat 스트리밍 응답 + 인용 Dialog 동작
- [ ] Actions Preview → Request → Approve → Execute 동작
- [ ] Library 배치 재인덱스 진행률 UI 동작
- [ ] Resources 네임스페이스 필터 + YAML 편집 Dialog 동작
- [ ] `apps/web/src/` 디렉터리에 `entities/`, `app/`, `shared/styles/partials` 없음
- [ ] `.gitkeep` 파일 0개 (`find apps/web/src -name ".gitkeep"`)
- [ ] 백엔드(`apps/api`) 변경 없음 (`git diff main..HEAD -- apps/api`로 확인)
- [ ] 엔드포인트 URL 문자열 불변 — 스펙 §7 매트릭스 대조

---

## 롤백 전략

각 커밋이 독립적으로 돌아가도록 설계됨. 문제 발생 시:
- 커밋 3 롤백: `git reset --hard <커밋2 해시>` → 구조는 새것, 외관은 구것으로 복귀
- 전체 롤백: `git reset --hard <커밋1 이전>` → 원상 복귀

각 Task 종료 시 빌드 그린을 확인하므로 중간 커밋도 모두 bisect 가능한 상태.

import { useState, type ReactNode } from "react";

import { LoaderCircle, Menu } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type {
  OcpConnectionProfile,
  OcpConnectionTestResult,
  OcpLeaseSchedulerStatus,
} from "@/features/connection/types";
import { Sidebar } from "./Sidebar";

export type AppRoute = "connection" | "chat" | "dashboard" | "resources" | "library" | "actions";

type AppShellProps = {
  activeRoute: AppRoute;
  onNavigate: (route: AppRoute) => void;
  profile: OcpConnectionProfile | null;
  testResult: OcpConnectionTestResult | null;
  schedulerStatus: OcpLeaseSchedulerStatus | null;
  message: string;
  railContent?: ReactNode;
  children: ReactNode;
  loadingState?: {
    active: boolean;
    title: string;
    detail?: string;
  };
};

export function AppShell({
  activeRoute,
  onNavigate,
  profile,
  testResult,
  schedulerStatus,
  message,
  railContent,
  children,
  loadingState,
}: AppShellProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex min-h-screen">
        <Sidebar
          open={sidebarOpen}
          activeRoute={activeRoute}
          onNavigate={onNavigate}
          onClose={() => setSidebarOpen(false)}
          railContent={railContent}
        />

        <div className="flex min-w-0 flex-1">
          <main className="min-w-0 flex-1 px-4 py-6 md:px-8">
            <div className="mx-auto min-h-full w-full max-w-[1600px]">
              <div className="mb-4 lg:hidden">
                <Button variant="outline" size="sm" onClick={() => setSidebarOpen(true)}>
                  <Menu className="mr-2 h-4 w-4" />
                  Menu
                </Button>
              </div>
              {children}
            </div>
          </main>
        </div>
      </div>

      {loadingState?.active ? (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/65 p-4 backdrop-blur-sm">
          <Card className="w-full max-w-md border-border/70 bg-card/95 shadow-2xl">
            <CardContent className="flex items-start gap-4 p-6">
              <div className="rounded-full bg-primary/10 p-3 text-primary">
                <LoaderCircle className="h-5 w-5 animate-spin" />
              </div>
              <div className="space-y-1">
                <div className="font-semibold text-foreground">{loadingState.title}</div>
                {loadingState.detail ? (
                  <div className="text-sm leading-6 text-muted-foreground">{loadingState.detail}</div>
                ) : null}
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}
    </div>
  );
}

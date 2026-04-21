import { useEffect, useMemo, useState } from "react";

import { PageHeader } from "@/shared/layout/PageHeader";
import { Button } from "@/shared/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/shared/ui/card";
import { Skeleton } from "@/shared/ui/skeleton";
import type { OcpOverviewResponse } from "@/domains/connection/types";
import type { OcpConnectionController } from "@/domains/connection/useOcpConnection";
import { getOcpOverview } from "@/domains/dashboard/ocpOverviewApi";

type DashboardPageProps = {
  controller: OcpConnectionController;
  onLoadingChange?: (state: { active: boolean; title: string; detail?: string }) => void;
};

export function DashboardPage({ controller, onLoadingChange }: DashboardPageProps) {
  const [overview, setOverview] = useState<OcpOverviewResponse | null>(null);
  const [overviewError, setOverviewError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function run() {
      if (!controller.profile) {
        setOverview(null);
        setOverviewError("");
        return;
      }

      setLoading(true);
      setOverviewError("");

      try {
        const overviewResult = await getOcpOverview(controller.profile.connectionId);
        setOverview(overviewResult);
      } catch (nextError) {
        setOverview(null);
        setOverviewError(nextError instanceof Error ? nextError.message : "Failed to load dashboard overview.");
      } finally {
        setLoading(false);
      }
    }

    void run();
  }, [controller.profile]);

  useEffect(() => {
    onLoadingChange?.({
      active: loading,
      title: "대시보드 불러오는 중",
      detail: "클러스터 개요와 리소스 상태를 가져오는 중입니다.",
    });
    return () => onLoadingChange?.({ active: false, title: "" });
  }, [loading, onLoadingChange]);

  const resourceEntries = useMemo(() => {
    if (!overview) return [];
    return Object.entries(overview.resourceCounts)
      .filter(([, value]) => value >= 0)
      .sort((left, right) => right[1] - left[1]);
  }, [overview]);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Cluster Overview"
        title="Operations dashboard"
        description="연결된 클러스터의 규모, namespace 가시성, RBAC posture, 시계열 활용량을 한 화면에서 확인합니다."
      />

      {!controller.profile ? (
        <Card className="border-border/70 bg-background/50">
          <CardContent className="p-6 text-sm text-muted-foreground">
            Connection 화면에서 클러스터 연결을 먼저 구성해야 합니다.
          </CardContent>
        </Card>
      ) : null}

      {overviewError ? (
        <Card className="border-destructive/40 bg-destructive/10">
          <CardContent className="p-6 text-sm text-destructive">{overviewError}</CardContent>
        </Card>
      ) : null}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[
          { label: "Nodes", value: overview ? overview.resourceCounts.nodes ?? 0 : null },
          { label: "Namespaces", value: overview ? overview.namespaceCount : null },
          { label: "Pods", value: overview ? overview.resourceCounts.pods ?? 0 : null },
          { label: "Services", value: overview ? overview.resourceCounts.services ?? 0 : null },
        ].map((item) => (
          <Card key={item.label} className="border-border/70 bg-card/95">
            <CardHeader className="pb-3">
              <CardTitle className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                {item.label}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              {item.value === null ? (
                <Skeleton className="h-10 w-20" />
              ) : (
                <div className="font-display text-4xl font-semibold tracking-tight text-foreground">
                  {item.value}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {overview ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="border-border/70 bg-card/95">
            <CardHeader>
              <CardTitle>Access posture</CardTitle>
              <CardDescription>검증된 identity와 현재 연결 컨텍스트의 핵심 상태입니다.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm text-muted-foreground md:grid-cols-2">
              <div><span className="font-medium text-foreground">Resolved User:</span> {controller.testResult?.resolvedUser || controller.profile?.displayName || "-"}</div>
              <div><span className="font-medium text-foreground">Default Namespace:</span> {controller.testResult?.resolvedNamespace || overview.defaultNamespace || "-"}</div>
              <div><span className="font-medium text-foreground">Roles:</span> {controller.testResult?.resolvedRoles.join(", ") || "-"}</div>
              <div><span className="font-medium text-foreground">Secret Backend:</span> {controller.testResult?.secretBackend || "-"}</div>
              <div><span className="font-medium text-foreground">Lease:</span> {controller.schedulerStatus?.running ? "running" : "idle"}</div>
              <div><span className="font-medium text-foreground">Message:</span> {controller.testResult?.message || overview.message}</div>
            </CardContent>
          </Card>

          <Card className="border-border/70 bg-card/95">
            <CardHeader>
              <CardTitle>Namespace sample</CardTitle>
              <CardDescription>현재 토큰으로 조회 가능한 namespace 일부입니다.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {overview.namespaceSample.length > 0 ? (
                  overview.namespaceSample.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-border/70 bg-background/60 px-3 py-1 text-xs text-muted-foreground"
                    >
                      {item}
                    </span>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground">No namespace sample</span>
                )}
              </div>
              <div className="text-sm text-muted-foreground">
                {overview.namespaceCount} namespaces visible
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}

      {resourceEntries.length > 0 ? (
        <Card className="border-border/70 bg-card/95">
          <CardHeader>
            <CardTitle>Resource density</CardTitle>
            <CardDescription>가장 큰 리소스 군집을 빠르게 파악하기 위한 분포입니다.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {resourceEntries.map(([label, value]) => {
              const maxValue = resourceEntries[0]?.[1] ?? 1;
              return (
                <div key={label} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-foreground">{label}</span>
                    <span className="text-muted-foreground">{value}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${Math.max(8, (value / maxValue) * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}



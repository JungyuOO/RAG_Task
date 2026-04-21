import { FormEvent, useEffect, useState } from "react";

import { PageHeader } from "@/shared/layout/PageHeader";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import type { WorkspaceCreateRequest, WorkspaceRecord } from "@/domains/workspaces/types";
import { createWorkspace, listWorkspaces } from "@/domains/workspaces/workspacesApi";

type WorkspacesPageProps = {
  selectedWorkspaceId: string;
  onSelectWorkspace: (workspaceId: string) => void;
  onLoadingChange?: (state: { active: boolean; title: string; detail?: string }) => void;
};

export function WorkspacesPage({
  selectedWorkspaceId,
  onSelectWorkspace,
  onLoadingChange,
}: WorkspacesPageProps) {
  const [items, setItems] = useState<WorkspaceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState<WorkspaceCreateRequest>({
    name: "",
    slug: "",
    industry: "",
    environment: "",
  });

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const next = await listWorkspaces();
      setItems(next);
      if (!selectedWorkspaceId && next[0]) {
        onSelectWorkspace(next[0].workspaceId);
      }
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to load workspaces.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    onLoadingChange?.({
      active: loading,
      title: "워크스페이스 불러오는 중",
      detail: "고객사별 작업 공간 목록을 가져오는 중입니다.",
    });
    return () => onLoadingChange?.({ active: false, title: "" });
  }, [loading, onLoadingChange]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const created = await createWorkspace(form);
      setForm({ name: "", slug: "", industry: "", environment: "" });
      await refresh();
      onSelectWorkspace(created.workspaceId);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to create workspace.");
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Workspaces"
        title="Workspace settings"
        description="고객사별 작업 공간을 만들고 현재 활성 workspace를 선택합니다."
      />

      {error ? (
        <Card className="border-destructive/40 bg-destructive/10">
          <CardContent className="p-6 text-sm text-destructive">{error}</CardContent>
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <Card className="border-border/70 bg-card/95">
          <CardHeader>
            <CardTitle>Workspace list</CardTitle>
            <CardDescription>선택된 workspace는 이후 모델 설정 및 고객사별 기능의 기준이 됩니다.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {items.length === 0 ? (
              <div className="text-sm text-muted-foreground">등록된 workspace가 없습니다.</div>
            ) : (
              items.map((item) => {
                const active = item.workspaceId === selectedWorkspaceId;
                return (
                  <button
                    key={item.workspaceId}
                    type="button"
                    onClick={() => onSelectWorkspace(item.workspaceId)}
                    className={`block w-full rounded-xl border px-4 py-3 text-left ${
                      active
                        ? "border-primary bg-primary/10"
                        : "border-border/70 bg-background/50 hover:bg-background/80"
                    }`}
                  >
                    <div className="font-medium text-foreground">{item.name}</div>
                    <div className="text-sm text-muted-foreground">
                      slug={item.slug} · {item.industry || "industry -"} · {item.environment || "env -"}
                    </div>
                  </button>
                );
              })
            )}
          </CardContent>
        </Card>

        <Card className="border-border/70 bg-card/95">
          <CardHeader>
            <CardTitle>Create workspace</CardTitle>
            <CardDescription>고객사 또는 운영 환경 단위로 새 workspace를 생성합니다.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="space-y-2">
                <Label htmlFor="workspace-name">Name</Label>
                <Input
                  id="workspace-name"
                  value={form.name ?? ""}
                  onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="workspace-slug">Slug</Label>
                <Input
                  id="workspace-slug"
                  value={form.slug ?? ""}
                  onChange={(event) => setForm((current) => ({ ...current, slug: event.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="workspace-industry">Industry</Label>
                <Input
                  id="workspace-industry"
                  value={form.industry ?? ""}
                  onChange={(event) => setForm((current) => ({ ...current, industry: event.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="workspace-environment">Environment</Label>
                <Input
                  id="workspace-environment"
                  value={form.environment ?? ""}
                  onChange={(event) => setForm((current) => ({ ...current, environment: event.target.value }))}
                />
              </div>
              <Button disabled={loading}>{loading ? "생성 중..." : "Create workspace"}</Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

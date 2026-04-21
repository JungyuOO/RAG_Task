import { FormEvent, useEffect, useState } from "react";

import type { WorkspaceCreateRequest, WorkspaceRecord } from "@/domains/workspaces/types";
import { createWorkspace, listWorkspaces } from "@/domains/workspaces/workspacesApi";
import { PageHeader } from "@/shared/layout/PageHeader";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";

type WorkspacesPageProps = {
  selectedWorkspaceId: string;
  onSelectWorkspace: (workspaceId: string) => void;
  onClearWorkspace: () => void;
  onLoadingChange?: (state: { active: boolean; title: string; detail?: string }) => void;
};

const DEFAULT_FORM: WorkspaceCreateRequest = {
  name: "",
  environment: "dev",
};

export function WorkspacesPage({
  selectedWorkspaceId,
  onSelectWorkspace,
  onClearWorkspace,
  onLoadingChange,
}: WorkspacesPageProps) {
  const [items, setItems] = useState<WorkspaceRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState<WorkspaceCreateRequest>(DEFAULT_FORM);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const next = await listWorkspaces();
      setItems(next);
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
      title: "Loading workspaces",
      detail: "Fetching available customer workspaces.",
    });
    return () => onLoadingChange?.({ active: false, title: "" });
  }, [loading, onLoadingChange]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const created = await createWorkspace(form);
      setForm(DEFAULT_FORM);
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
        title="Workspaces"
        description="Create a customer workspace, choose the active one, or clear the current selection before switching contexts."
        actions={
          selectedWorkspaceId ? (
            <Button type="button" variant="outline" onClick={onClearWorkspace}>
              Clear active workspace
            </Button>
          ) : null
        }
      />

      {error ? (
        <Card className="surface-danger">
          <CardContent className="p-6 text-sm text-destructive">{error}</CardContent>
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.08fr)_minmax(0,0.92fr)]">
        <Card className="surface-soft">
          <CardHeader>
            <CardTitle>Workspace list</CardTitle>
            <CardDescription>The active workspace drives models, connections, library context, and SCM delivery.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {items.length === 0 ? (
              <div className="text-sm text-muted-foreground">No workspace has been created yet.</div>
            ) : (
              items.map((item) => {
                const active = item.workspaceId === selectedWorkspaceId;
                return (
                  <button
                    key={item.workspaceId}
                    type="button"
                    onClick={() => onSelectWorkspace(item.workspaceId)}
                    className={`block w-full rounded-xl border px-4 py-3 text-left ${
                      active ? "surface-brand" : "surface-muted hover:bg-background/80"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="font-medium text-foreground">{item.name}</div>
                        <div className="text-sm text-muted-foreground">{item.environment || "No environment set"}</div>
                        <div className="mt-1 text-xs text-muted-foreground/80">slug={item.slug}</div>
                      </div>
                      {active ? (
                        <div className="rounded-full border border-border/70 px-2.5 py-1 text-xs text-foreground">
                          Active
                        </div>
                      ) : null}
                    </div>
                  </button>
                );
              })
            )}
          </CardContent>
        </Card>

        <Card className="surface-soft">
          <CardHeader>
            <CardTitle>Create workspace</CardTitle>
            <CardDescription>Name is enough for MVP. Slug is generated automatically, and environment stays lightweight.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="space-y-2">
                <Label htmlFor="workspace-name">Name</Label>
                <Input
                  id="workspace-name"
                  value={form.name ?? ""}
                  onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Demo Workspace"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="workspace-environment">Environment</Label>
                <select
                  id="workspace-environment"
                  className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                  value={form.environment ?? "dev"}
                  onChange={(event) => setForm((current) => ({ ...current, environment: event.target.value }))}
                >
                  <option value="dev">dev</option>
                  <option value="staging">staging</option>
                  <option value="prod">prod</option>
                </select>
              </div>
              <Button disabled={loading || !form.name.trim()}>{loading ? "Creating..." : "Create workspace"}</Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

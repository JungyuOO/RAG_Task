import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  buildScmDeploymentPlan,
  createScmConnection,
  createScmRepository,
  listScmConnections,
  listScmRepositories,
  startScmOauth,
  updateScmRepository,
} from "@/domains/scm/scmApi";
import type {
  ScmConnectionCreateRequest,
  ScmConnectionRecord,
  ScmDeploymentPlanResponse,
  ScmProvider,
  ScmRepositoryCreateRequest,
  ScmRepositoryRecord,
} from "@/domains/scm/types";
import type { WorkspaceRecord } from "@/domains/workspaces/types";
import { PageHeader } from "@/shared/layout/PageHeader";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";

type ScmPageProps = {
  selectedWorkspace: WorkspaceRecord | null;
  onLoadingChange?: (state: { active: boolean; title: string; detail?: string }) => void;
};

type RepositoryDraft = {
  defaultBranch: string;
  configPath: string;
  deliveryMode: string;
  manifestKind: string;
  targetClusterUrl: string;
  targetNamespace: string;
  autoDeployEnabled: boolean;
};

type PlanForm = {
  repositoryId: string;
  resourceKind: string;
  resourceName: string;
  targetNamespace: string;
  replicas: string;
  imageTag: string;
  configKey: string;
  reason: string;
};

const EMPTY_CONNECTION_FORM: ScmConnectionCreateRequest = {
  provider: "github",
  hostUrl: "https://github.com",
  authType: "token",
  accountLabel: "",
};

const EMPTY_REPOSITORY_FORM: ScmRepositoryCreateRequest = {
  scmConnectionId: "",
  repoFullName: "",
  defaultBranch: "main",
  configPath: "config.yaml",
  deliveryMode: "gitops_commit",
  manifestKind: "config_yaml",
  targetClusterUrl: "",
  targetNamespace: "default",
  autoDeployEnabled: true,
};

const EMPTY_PLAN_FORM: PlanForm = {
  repositoryId: "",
  resourceKind: "Deployment",
  resourceName: "",
  targetNamespace: "",
  replicas: "3",
  imageTag: "",
  configKey: "replicas",
  reason: "",
};

export function ScmPage({ selectedWorkspace, onLoadingChange }: ScmPageProps) {
  const [connections, setConnections] = useState<ScmConnectionRecord[]>([]);
  const [repositories, setRepositories] = useState<ScmRepositoryRecord[]>([]);
  const [connectionForm, setConnectionForm] = useState<ScmConnectionCreateRequest>(EMPTY_CONNECTION_FORM);
  const [repositoryForm, setRepositoryForm] = useState<ScmRepositoryCreateRequest>(EMPTY_REPOSITORY_FORM);
  const [drafts, setDrafts] = useState<Record<string, RepositoryDraft>>({});
  const [planForm, setPlanForm] = useState<PlanForm>(EMPTY_PLAN_FORM);
  const [deploymentPlan, setDeploymentPlan] = useState<ScmDeploymentPlanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState<
    "" | "connection" | "repository" | "plan" | `repository:${string}`
  >("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const connectionMap = useMemo(
    () => Object.fromEntries(connections.map((item) => [item.scmConnectionId, item])),
    [connections],
  );

  async function refresh(workspaceId: string) {
    setLoading(true);
    setError("");
    try {
      const [nextConnections, nextRepositories] = await Promise.all([
        listScmConnections(workspaceId),
        listScmRepositories(workspaceId),
      ]);
      setConnections(nextConnections);
      setRepositories(nextRepositories);
      setDrafts(
        Object.fromEntries(
          nextRepositories.map((item) => [
            item.repositoryId,
            {
              defaultBranch: item.defaultBranch,
              configPath: item.configPath,
              deliveryMode: item.deliveryMode,
              manifestKind: item.manifestKind,
              targetClusterUrl: item.targetClusterUrl,
              targetNamespace: item.targetNamespace,
              autoDeployEnabled: item.autoDeployEnabled,
            },
          ]),
        ),
      );
      setRepositoryForm((current) => ({
        ...current,
        scmConnectionId: current.scmConnectionId || nextConnections[0]?.scmConnectionId || "",
      }));
      setPlanForm((current) => ({
        ...current,
        repositoryId: current.repositoryId || nextRepositories[0]?.repositoryId || "",
        targetNamespace: current.targetNamespace || nextRepositories[0]?.targetNamespace || "",
      }));
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to load SCM settings.");
      setConnections([]);
      setRepositories([]);
      setDeploymentPlan(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    async function run() {
      if (!selectedWorkspace) {
        setConnections([]);
        setRepositories([]);
        setDrafts({});
        setRepositoryForm(EMPTY_REPOSITORY_FORM);
        setPlanForm(EMPTY_PLAN_FORM);
        setDeploymentPlan(null);
        return;
      }
      await refresh(selectedWorkspace.workspaceId);
    }
    void run();
  }, [selectedWorkspace?.workspaceId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const search = new URLSearchParams(window.location.search);
    const oauthStatus = search.get("oauth_status")?.trim() ?? "";
    const provider = search.get("provider")?.trim() ?? "";
    const oauthMessage = search.get("message")?.trim() ?? "";
    if (!oauthStatus) return;
    if (oauthStatus === "connected") {
      setMessage(provider ? `${provider} account connected.` : "SCM account connected.");
      setError("");
    } else {
      setError(oauthMessage || "OAuth connection failed.");
      setMessage("");
    }
    window.history.replaceState(null, "", "/scm");
  }, []);

  useEffect(() => {
    onLoadingChange?.({
      active: loading,
      title: "Loading SCM settings",
      detail: "Fetching saved GitHub and GitLab connections for the active workspace.",
    });
    return () => onLoadingChange?.({ active: false, title: "" });
  }, [loading, onLoadingChange]);

  function updateProvider(provider: ScmProvider) {
    setConnectionForm((current) => ({
      ...current,
      provider,
      hostUrl: provider === "gitlab" ? "https://gitlab.com" : "https://github.com",
    }));
  }

  async function handleOauthConnect(provider: ScmProvider) {
    if (!selectedWorkspace || typeof window === "undefined") return;
    setError("");
    setMessage("");
    try {
      const started = await startScmOauth(provider, selectedWorkspace.workspaceId);
      window.location.assign(started.authorizeUrl);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : `Failed to start ${provider} OAuth.`);
    }
  }

  async function handleCreateConnection(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedWorkspace) return;
    setSubmitting("connection");
    setError("");
    setMessage("");
    try {
      const created = await createScmConnection(selectedWorkspace.workspaceId, connectionForm);
      await refresh(selectedWorkspace.workspaceId);
      setConnectionForm({
        provider: created.provider,
        hostUrl: created.provider === "gitlab" ? "https://gitlab.com" : "https://github.com",
        authType: created.authType,
        accountLabel: "",
      });
      setRepositoryForm((current) => ({
        ...current,
        scmConnectionId: current.scmConnectionId || created.scmConnectionId,
      }));
      setMessage("SCM connection saved.");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to create SCM connection.");
    } finally {
      setSubmitting("");
    }
  }

  async function handleCreateRepository(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedWorkspace) return;
    setSubmitting("repository");
    setError("");
    setMessage("");
    try {
      const created = await createScmRepository(selectedWorkspace.workspaceId, repositoryForm);
      await refresh(selectedWorkspace.workspaceId);
      setRepositoryForm((current) => ({
        ...EMPTY_REPOSITORY_FORM,
        scmConnectionId: current.scmConnectionId,
      }));
      setPlanForm((current) => ({
        ...current,
        repositoryId: current.repositoryId || created.repositoryId,
        targetNamespace: current.targetNamespace || created.targetNamespace || "",
      }));
      setMessage("Repository delivery profile saved.");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to register repository.");
    } finally {
      setSubmitting("");
    }
  }

  async function handleRepositorySave(repositoryId: string) {
    if (!selectedWorkspace) return;
    const draft = drafts[repositoryId];
    if (!draft) return;
    setSubmitting(`repository:${repositoryId}`);
    setError("");
    setMessage("");
    try {
      await updateScmRepository(selectedWorkspace.workspaceId, repositoryId, draft);
      await refresh(selectedWorkspace.workspaceId);
      setMessage("Repository delivery settings updated.");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to update repository.");
    } finally {
      setSubmitting("");
    }
  }

  async function handleBuildPlan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedWorkspace || !planForm.repositoryId) return;
    setSubmitting("plan");
    setError("");
    setMessage("");
    try {
      const plan = await buildScmDeploymentPlan(selectedWorkspace.workspaceId, planForm.repositoryId, {
        resourceKind: planForm.resourceKind,
        resourceName: planForm.resourceName,
        targetNamespace: planForm.targetNamespace,
        replicas: planForm.replicas ? Number(planForm.replicas) : undefined,
        imageTag: planForm.imageTag,
        configKey: planForm.configKey,
        reason: planForm.reason,
      });
      setDeploymentPlan(plan);
      setMessage("Repo-driven deployment plan generated.");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to build deployment plan.");
    } finally {
      setSubmitting("");
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="SCM"
        title="GitHub and GitLab delivery"
        description="Treat the repository as the deployment source of truth. Register the connected repo, branch, config file, target cluster, and GitOps or CI/CD delivery mode for each workspace."
      />

      {!selectedWorkspace ? (
        <Card className="surface-muted">
          <CardContent className="p-6 text-sm text-muted-foreground">
            Select a workspace first. SCM connections and repository delivery profiles are stored per workspace.
          </CardContent>
        </Card>
      ) : null}

      {selectedWorkspace ? (
        <Card className="surface-brand">
          <CardContent className="flex flex-wrap items-center justify-between gap-3 p-5">
            <div>
              <div className="text-xs uppercase tracking-[0.2em] text-primary-foreground/70">Active workspace</div>
              <div className="mt-1 text-lg font-semibold text-foreground">{selectedWorkspace.name}</div>
            </div>
            <div className="text-sm text-muted-foreground">slug={selectedWorkspace.slug}</div>
          </CardContent>
        </Card>
      ) : null}

      {error ? (
        <Card className="surface-danger">
          <CardContent className="p-6 text-sm text-destructive">{error}</CardContent>
        </Card>
      ) : null}

      {message ? (
        <Card className="surface-brand">
          <CardContent className="p-6 text-sm text-foreground">{message}</CardContent>
        </Card>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
        <Card className="surface-soft">
          <CardHeader>
            <CardTitle>OAuth connect</CardTitle>
            <CardDescription>Connect GitHub or GitLab directly, then keep manual connection only for self-hosted or PAT fallback cases.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="grid gap-3 md:grid-cols-2">
              <button
                type="button"
                onClick={() => void handleOauthConnect("github")}
                disabled={!selectedWorkspace}
                className="surface-muted flex items-center justify-between rounded-xl border border-border/70 px-4 py-4 text-left transition hover:bg-background/80 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <div>
                  <div className="font-medium text-foreground">Connect GitHub</div>
                  <div className="text-sm text-muted-foreground">Sign in and create a workspace connection.</div>
                </div>
                <div className="rounded-full bg-foreground px-3 py-1 text-xs font-semibold text-background">GH</div>
              </button>
              <button
                type="button"
                onClick={() => void handleOauthConnect("gitlab")}
                disabled={!selectedWorkspace}
                className="surface-muted flex items-center justify-between rounded-xl border border-border/70 px-4 py-4 text-left transition hover:bg-background/80 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <div>
                  <div className="font-medium text-foreground">Connect GitLab</div>
                  <div className="text-sm text-muted-foreground">Authorize the workspace against GitLab delivery repos.</div>
                </div>
                <div className="rounded-full bg-[#fc6d26] px-3 py-1 text-xs font-semibold text-white">GL</div>
              </button>
            </div>

            <div className="rounded-xl border border-dashed border-border/70 px-4 py-4 text-sm text-muted-foreground">
              Manual connection is still available below for self-hosted GitLab, GitHub Enterprise, or environments where OAuth apps cannot be used.
            </div>

            <form className="space-y-4" onSubmit={handleCreateConnection}>
              <div className="space-y-2">
                <Label htmlFor="scm-provider">Provider</Label>
                <select
                  id="scm-provider"
                  className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                  value={connectionForm.provider}
                  onChange={(event) => updateProvider(event.target.value as ScmProvider)}
                >
                  <option value="github">GitHub</option>
                  <option value="gitlab">GitLab</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="scm-host-url">Host URL</Label>
                <Input
                  id="scm-host-url"
                  value={connectionForm.hostUrl ?? ""}
                  onChange={(event) => setConnectionForm((current) => ({ ...current, hostUrl: event.target.value }))}
                  placeholder="https://github.com or self-hosted URL"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="scm-auth-type">Auth type</Label>
                <select
                  id="scm-auth-type"
                  className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                  value={connectionForm.authType ?? "token"}
                  onChange={(event) => setConnectionForm((current) => ({ ...current, authType: event.target.value }))}
                >
                  <option value="token">Token</option>
                  <option value="oauth">OAuth (metadata only)</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="scm-account-label">Account label</Label>
                <Input
                  id="scm-account-label"
                  value={connectionForm.accountLabel ?? ""}
                  onChange={(event) => setConnectionForm((current) => ({ ...current, accountLabel: event.target.value }))}
                  placeholder="customer-admin"
                />
              </div>
              <Button disabled={!selectedWorkspace || submitting === "connection"}>
                {submitting === "connection" ? "Saving connection..." : "Add SCM connection"}
              </Button>
            </form>

            <div className="space-y-3">
              {connections.length === 0 ? (
                <div className="rounded-xl border border-dashed border-border/70 px-4 py-5 text-sm text-muted-foreground">
                  No GitHub or GitLab connection is saved for this workspace yet.
                </div>
              ) : (
                connections.map((connection) => (
                  <div key={connection.scmConnectionId} className="surface-muted rounded-xl px-4 py-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="font-medium text-foreground">{connection.accountLabel || `${connection.provider} account`}</div>
                        <div className="text-sm text-muted-foreground">
                          {connection.provider} · {connection.hostUrl || "default host"}
                        </div>
                      </div>
                      <div className="rounded-full border border-border/70 px-2.5 py-1 text-xs text-muted-foreground">
                        {connection.status}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="surface-soft">
          <CardHeader>
            <CardTitle>Repository delivery profile</CardTitle>
            <CardDescription>Register the repository and define how repo changes flow into OpenShift deployment.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <form className="grid gap-4 md:grid-cols-2" onSubmit={handleCreateRepository}>
              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="repo-connection">SCM connection</Label>
                <select
                  id="repo-connection"
                  className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                  value={repositoryForm.scmConnectionId}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, scmConnectionId: event.target.value }))}
                >
                  <option value="">Select a saved connection</option>
                  {connections.map((connection) => (
                    <option key={connection.scmConnectionId} value={connection.scmConnectionId}>
                      {connection.provider} · {connection.accountLabel || connection.hostUrl || connection.scmConnectionId}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="repo-full-name">Repository</Label>
                <Input
                  id="repo-full-name"
                  value={repositoryForm.repoFullName}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, repoFullName: event.target.value }))}
                  placeholder="org/project or group/project"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="repo-branch">Default branch</Label>
                <Input
                  id="repo-branch"
                  value={repositoryForm.defaultBranch ?? "main"}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, defaultBranch: event.target.value }))}
                  placeholder="main"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="repo-config-path">Config path</Label>
                <Input
                  id="repo-config-path"
                  value={repositoryForm.configPath ?? "config.yaml"}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, configPath: event.target.value }))}
                  placeholder="deploy/config.yaml"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="repo-delivery-mode">Delivery mode</Label>
                <select
                  id="repo-delivery-mode"
                  className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                  value={repositoryForm.deliveryMode ?? "gitops_commit"}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, deliveryMode: event.target.value }))}
                >
                  <option value="gitops_commit">GitOps commit</option>
                  <option value="cicd_pipeline">CI/CD pipeline</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="repo-manifest-kind">Manifest kind</Label>
                <select
                  id="repo-manifest-kind"
                  className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                  value={repositoryForm.manifestKind ?? "config_yaml"}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, manifestKind: event.target.value }))}
                >
                  <option value="config_yaml">config.yaml</option>
                  <option value="helm_values">Helm values</option>
                  <option value="kustomize">Kustomize overlay</option>
                </select>
              </div>
              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="repo-target-cluster">Target cluster URL</Label>
                <Input
                  id="repo-target-cluster"
                  value={repositoryForm.targetClusterUrl ?? ""}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, targetClusterUrl: event.target.value }))}
                  placeholder="https://api.cluster.example.com:6443"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="repo-target-namespace">Target namespace</Label>
                <Input
                  id="repo-target-namespace"
                  value={repositoryForm.targetNamespace ?? ""}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, targetNamespace: event.target.value }))}
                  placeholder="payments"
                />
              </div>
              <label className="surface-muted mt-7 flex items-center gap-2 rounded-xl px-4 py-3 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-[#1060ff]"
                  checked={repositoryForm.autoDeployEnabled ?? true}
                  onChange={(event) => setRepositoryForm((current) => ({ ...current, autoDeployEnabled: event.target.checked }))}
                />
                Auto-deploy after repo change
              </label>
              <div className="md:col-span-2">
                <Button disabled={!selectedWorkspace || !repositoryForm.scmConnectionId || submitting === "repository"}>
                  {submitting === "repository" ? "Saving repository..." : "Add delivery profile"}
                </Button>
              </div>
            </form>

            <div className="space-y-3">
              {repositories.length === 0 ? (
                <div className="rounded-xl border border-dashed border-border/70 px-4 py-5 text-sm text-muted-foreground">
                  No delivery profile is registered yet. Add a saved connection first, then register the repo and delivery settings.
                </div>
              ) : (
                repositories.map((repository) => {
                  const draft = drafts[repository.repositoryId] ?? {
                    defaultBranch: repository.defaultBranch,
                    configPath: repository.configPath,
                    deliveryMode: repository.deliveryMode,
                    manifestKind: repository.manifestKind,
                    targetClusterUrl: repository.targetClusterUrl,
                    targetNamespace: repository.targetNamespace,
                    autoDeployEnabled: repository.autoDeployEnabled,
                  };
                  const connection = connectionMap[repository.scmConnectionId];
                  return (
                    <div key={repository.repositoryId} className="surface-muted space-y-4 rounded-xl px-4 py-4">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div>
                          <div className="font-medium text-foreground">{repository.repoFullName}</div>
                          <div className="text-sm text-muted-foreground">
                            {connection?.provider ?? "scm"} · {connection?.accountLabel || connection?.hostUrl || repository.scmConnectionId}
                          </div>
                        </div>
                        <div className="rounded-full border border-border/70 px-2.5 py-1 text-xs text-muted-foreground">
                          {repository.syncStatus}
                        </div>
                      </div>
                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-2">
                          <Label htmlFor={`branch-${repository.repositoryId}`}>Default branch</Label>
                          <Input
                            id={`branch-${repository.repositoryId}`}
                            value={draft.defaultBranch}
                            onChange={(event) => setDrafts((current) => ({ ...current, [repository.repositoryId]: { ...draft, defaultBranch: event.target.value } }))}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor={`config-${repository.repositoryId}`}>Config path</Label>
                          <Input
                            id={`config-${repository.repositoryId}`}
                            value={draft.configPath}
                            onChange={(event) => setDrafts((current) => ({ ...current, [repository.repositoryId]: { ...draft, configPath: event.target.value } }))}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor={`delivery-${repository.repositoryId}`}>Delivery mode</Label>
                          <select
                            id={`delivery-${repository.repositoryId}`}
                            className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                            value={draft.deliveryMode}
                            onChange={(event) => setDrafts((current) => ({ ...current, [repository.repositoryId]: { ...draft, deliveryMode: event.target.value } }))}
                          >
                            <option value="gitops_commit">GitOps commit</option>
                            <option value="cicd_pipeline">CI/CD pipeline</option>
                          </select>
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor={`manifest-${repository.repositoryId}`}>Manifest kind</Label>
                          <select
                            id={`manifest-${repository.repositoryId}`}
                            className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                            value={draft.manifestKind}
                            onChange={(event) => setDrafts((current) => ({ ...current, [repository.repositoryId]: { ...draft, manifestKind: event.target.value } }))}
                          >
                            <option value="config_yaml">config.yaml</option>
                            <option value="helm_values">Helm values</option>
                            <option value="kustomize">Kustomize overlay</option>
                          </select>
                        </div>
                        <div className="space-y-2 md:col-span-2">
                          <Label htmlFor={`cluster-${repository.repositoryId}`}>Target cluster URL</Label>
                          <Input
                            id={`cluster-${repository.repositoryId}`}
                            value={draft.targetClusterUrl}
                            onChange={(event) => setDrafts((current) => ({ ...current, [repository.repositoryId]: { ...draft, targetClusterUrl: event.target.value } }))}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor={`namespace-${repository.repositoryId}`}>Target namespace</Label>
                          <Input
                            id={`namespace-${repository.repositoryId}`}
                            value={draft.targetNamespace}
                            onChange={(event) => setDrafts((current) => ({ ...current, [repository.repositoryId]: { ...draft, targetNamespace: event.target.value } }))}
                          />
                        </div>
                        <label className="surface-soft mt-7 flex items-center gap-2 rounded-xl px-4 py-3 text-sm text-muted-foreground">
                          <input
                            type="checkbox"
                            className="h-4 w-4 accent-[#1060ff]"
                            checked={draft.autoDeployEnabled}
                            onChange={(event) => setDrafts((current) => ({ ...current, [repository.repositoryId]: { ...draft, autoDeployEnabled: event.target.checked } }))}
                          />
                          Auto-deploy after repo change
                        </label>
                      </div>
                      <Button disabled={!selectedWorkspace || submitting === `repository:${repository.repositoryId}`} onClick={() => void handleRepositorySave(repository.repositoryId)}>
                        {submitting === `repository:${repository.repositoryId}` ? "Saving..." : "Update delivery settings"}
                      </Button>
                    </div>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="surface-soft">
        <CardHeader>
          <CardTitle>Repo-driven deployment plan</CardTitle>
          <CardDescription>Preview how an operational change should be translated into a repository change instead of direct cluster YAML edits.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <form className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" onSubmit={handleBuildPlan}>
            <div className="space-y-2 xl:col-span-3">
              <Label htmlFor="plan-repository">Delivery profile</Label>
              <select
                id="plan-repository"
                className="flex h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-primary/30"
                value={planForm.repositoryId}
                onChange={(event) => setPlanForm((current) => ({ ...current, repositoryId: event.target.value }))}
              >
                <option value="">Select repository profile</option>
                {repositories.map((repository) => (
                  <option key={repository.repositoryId} value={repository.repositoryId}>
                    {repository.repoFullName} · {repository.deliveryMode}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="plan-resource-kind">Resource kind</Label>
              <Input id="plan-resource-kind" value={planForm.resourceKind} onChange={(event) => setPlanForm((current) => ({ ...current, resourceKind: event.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="plan-resource-name">Resource name</Label>
              <Input id="plan-resource-name" value={planForm.resourceName} onChange={(event) => setPlanForm((current) => ({ ...current, resourceName: event.target.value }))} placeholder="payments-api" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="plan-namespace">Target namespace</Label>
              <Input id="plan-namespace" value={planForm.targetNamespace} onChange={(event) => setPlanForm((current) => ({ ...current, targetNamespace: event.target.value }))} placeholder="payments" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="plan-replicas">Replicas</Label>
              <Input id="plan-replicas" type="number" min={0} value={planForm.replicas} onChange={(event) => setPlanForm((current) => ({ ...current, replicas: event.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="plan-image-tag">Image tag</Label>
              <Input id="plan-image-tag" value={planForm.imageTag} onChange={(event) => setPlanForm((current) => ({ ...current, imageTag: event.target.value }))} placeholder="v2.4.1" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="plan-config-key">Config key</Label>
              <Input id="plan-config-key" value={planForm.configKey} onChange={(event) => setPlanForm((current) => ({ ...current, configKey: event.target.value }))} placeholder="replicas" />
            </div>
            <div className="space-y-2 xl:col-span-3">
              <Label htmlFor="plan-reason">Reason</Label>
              <textarea
                id="plan-reason"
                className="min-h-[96px] w-full rounded-md border border-border bg-input px-3 py-2 text-sm"
                value={planForm.reason}
                onChange={(event) => setPlanForm((current) => ({ ...current, reason: event.target.value }))}
                placeholder="Explain why the repository-backed delivery change is needed."
              />
            </div>
            <div className="xl:col-span-3">
              <Button disabled={!selectedWorkspace || !planForm.repositoryId || submitting === "plan"}>
                {submitting === "plan" ? "Building plan..." : "Build repo deployment plan"}
              </Button>
            </div>
          </form>

          {deploymentPlan ? (
            <div className="surface-muted space-y-4 rounded-xl p-5">
              <div className="space-y-1">
                <div className="text-sm font-medium text-foreground">{deploymentPlan.summary}</div>
                <div className="text-sm text-muted-foreground">{deploymentPlan.nextStep}</div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Repository</div>
                  <div className="mt-1 text-sm text-foreground">
                    {deploymentPlan.repoFullName} · {deploymentPlan.defaultBranch}
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Trigger</div>
                  <div className="mt-1 text-sm text-foreground">{deploymentPlan.triggerKind}</div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Target</div>
                  <div className="mt-1 text-sm text-foreground">
                    {deploymentPlan.targetClusterUrl || "cluster not set"} · {deploymentPlan.targetNamespace || "namespace not set"}
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Manifest</div>
                  <div className="mt-1 text-sm text-foreground">
                    {deploymentPlan.manifestKind} · {deploymentPlan.configPath}
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Suggested updates</div>
                <ul className="space-y-2 text-sm text-foreground">
                  {deploymentPlan.suggestedUpdates.map((item) => (
                    <li key={item} className="rounded-lg border border-border/60 bg-background/60 px-3 py-2">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Commit title</div>
                  <div className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-sm text-foreground">
                    {deploymentPlan.commitTitle}
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Files to change</div>
                  <div className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-sm text-foreground">
                    {deploymentPlan.filesToChange.join(", ")}
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Commit body</div>
                <pre className="overflow-auto rounded-lg border border-border/60 bg-background/60 px-3 py-3 text-sm text-foreground whitespace-pre-wrap">{deploymentPlan.commitBody}</pre>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}

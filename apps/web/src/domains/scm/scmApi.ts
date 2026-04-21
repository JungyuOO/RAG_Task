import type {
  ScmConnectionCreateRequest,
  ScmConnectionRecord,
  ScmDeploymentPlanRequest,
  ScmDeploymentPlanResponse,
  ScmRepositoryCreateRequest,
  ScmRepositoryRecord,
  ScmRepositoryUpdateRequest,
} from "@/domains/scm/types";

type ApiScmConnectionRecord = {
  scm_connection_id: string;
  workspace_id: string;
  provider: "github" | "gitlab";
  host_url: string;
  auth_type: string;
  account_label: string;
  login_name: string;
  scopes: string[];
  secret_ref: string;
  status: string;
  created_at: string;
  updated_at: string;
};

type ApiScmRepositoryRecord = {
  repository_id: string;
  workspace_id: string;
  scm_connection_id: string;
  repo_full_name: string;
  default_branch: string;
  config_path: string;
  delivery_mode: string;
  manifest_kind: string;
  target_cluster_url: string;
  target_namespace: string;
  auto_deploy_enabled: boolean;
  sync_status: string;
  created_at: string;
  updated_at: string;
};

type ApiScmDeploymentPlanResponse = {
  repository_id: string;
  workspace_id: string;
  repo_full_name: string;
  default_branch: string;
  config_path: string;
  delivery_mode: string;
  manifest_kind: string;
  target_cluster_url: string;
  target_namespace: string;
  auto_deploy_enabled: boolean;
  files_to_change: string[];
  suggested_updates: string[];
  trigger_kind: string;
  summary: string;
  commit_title: string;
  commit_body: string;
  requires_pull_request: boolean;
  next_step: string;
};

type ApiScmConnectionListResponse = {
  items: ApiScmConnectionRecord[];
};

type ApiScmRepositoryListResponse = {
  items: ApiScmRepositoryRecord[];
};

function apiUrl(path: string) {
  if (typeof window === "undefined") return path;
  const { origin } = window.location;
  return `${origin}${path}`;
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

function mapConnection(input: ApiScmConnectionRecord): ScmConnectionRecord {
  return {
    scmConnectionId: input.scm_connection_id,
    workspaceId: input.workspace_id,
    provider: input.provider,
    hostUrl: input.host_url,
    authType: input.auth_type,
    accountLabel: input.account_label,
    loginName: input.login_name,
    scopes: input.scopes ?? [],
    secretRef: input.secret_ref,
    status: input.status,
    createdAt: input.created_at,
    updatedAt: input.updated_at,
  };
}

function mapRepository(input: ApiScmRepositoryRecord): ScmRepositoryRecord {
  return {
    repositoryId: input.repository_id,
    workspaceId: input.workspace_id,
    scmConnectionId: input.scm_connection_id,
    repoFullName: input.repo_full_name,
    defaultBranch: input.default_branch,
    configPath: input.config_path,
    deliveryMode: input.delivery_mode,
    manifestKind: input.manifest_kind,
    targetClusterUrl: input.target_cluster_url,
    targetNamespace: input.target_namespace,
    autoDeployEnabled: input.auto_deploy_enabled,
    syncStatus: input.sync_status,
    createdAt: input.created_at,
    updatedAt: input.updated_at,
  };
}

function mapDeploymentPlan(input: ApiScmDeploymentPlanResponse): ScmDeploymentPlanResponse {
  return {
    repositoryId: input.repository_id,
    workspaceId: input.workspace_id,
    repoFullName: input.repo_full_name,
    defaultBranch: input.default_branch,
    configPath: input.config_path,
    deliveryMode: input.delivery_mode,
    manifestKind: input.manifest_kind,
    targetClusterUrl: input.target_cluster_url,
    targetNamespace: input.target_namespace,
    autoDeployEnabled: input.auto_deploy_enabled,
    filesToChange: input.files_to_change ?? [],
    suggestedUpdates: input.suggested_updates ?? [],
    triggerKind: input.trigger_kind,
    summary: input.summary,
    commitTitle: input.commit_title,
    commitBody: input.commit_body,
    requiresPullRequest: input.requires_pull_request,
    nextStep: input.next_step,
  };
}

export async function listScmConnections(workspaceId: string): Promise<ScmConnectionRecord[]> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/scm/connections`));
  const payload = await readJson<ApiScmConnectionListResponse>(response);
  return (payload.items ?? []).map(mapConnection);
}

export async function createScmConnection(
  workspaceId: string,
  request: ScmConnectionCreateRequest,
): Promise<ScmConnectionRecord> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/scm/connections`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: request.provider,
      host_url: request.hostUrl ?? "",
      auth_type: request.authType ?? "token",
      account_label: request.accountLabel ?? "",
    }),
  });
  return mapConnection(await readJson<ApiScmConnectionRecord>(response));
}

export async function startScmOauth(
  provider: "github" | "gitlab",
  workspaceId: string,
): Promise<{ authorizeUrl: string; state: string }> {
  const response = await fetch(apiUrl(`/api/v1/oauth/${provider}/start?workspace_id=${encodeURIComponent(workspaceId)}`), {
    method: "POST",
  });
  const payload = await readJson<{ authorize_url: string; state: string }>(response);
  return {
    authorizeUrl: payload.authorize_url,
    state: payload.state,
  };
}

export async function listScmRepositories(workspaceId: string): Promise<ScmRepositoryRecord[]> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/scm/repositories`));
  const payload = await readJson<ApiScmRepositoryListResponse>(response);
  return (payload.items ?? []).map(mapRepository);
}

export async function createScmRepository(
  workspaceId: string,
  request: ScmRepositoryCreateRequest,
): Promise<ScmRepositoryRecord> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/scm/repositories`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scm_connection_id: request.scmConnectionId,
      repo_full_name: request.repoFullName,
      default_branch: request.defaultBranch ?? "main",
      config_path: request.configPath ?? "config.yaml",
      delivery_mode: request.deliveryMode ?? "gitops_commit",
      manifest_kind: request.manifestKind ?? "config_yaml",
      target_cluster_url: request.targetClusterUrl ?? "",
      target_namespace: request.targetNamespace ?? "",
      auto_deploy_enabled: request.autoDeployEnabled ?? true,
    }),
  });
  return mapRepository(await readJson<ApiScmRepositoryRecord>(response));
}

export async function updateScmRepository(
  workspaceId: string,
  repositoryId: string,
  request: ScmRepositoryUpdateRequest,
): Promise<ScmRepositoryRecord> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/scm/repositories/${repositoryId}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      default_branch: request.defaultBranch ?? "",
      config_path: request.configPath ?? "",
      delivery_mode: request.deliveryMode ?? "",
      manifest_kind: request.manifestKind ?? "",
      target_cluster_url: request.targetClusterUrl ?? "",
      target_namespace: request.targetNamespace ?? "",
      auto_deploy_enabled:
        typeof request.autoDeployEnabled === "boolean" ? request.autoDeployEnabled : null,
    }),
  });
  return mapRepository(await readJson<ApiScmRepositoryRecord>(response));
}

export async function buildScmDeploymentPlan(
  workspaceId: string,
  repositoryId: string,
  request: ScmDeploymentPlanRequest,
): Promise<ScmDeploymentPlanResponse> {
  const response = await fetch(
    apiUrl(`/api/v1/workspaces/${workspaceId}/scm/repositories/${repositoryId}/deployment-plan`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resource_kind: request.resourceKind ?? "Deployment",
        resource_name: request.resourceName,
        target_namespace: request.targetNamespace ?? "",
        replicas: typeof request.replicas === "number" ? request.replicas : null,
        image_tag: request.imageTag ?? "",
        config_key: request.configKey ?? "replicas",
        reason: request.reason ?? "",
      }),
    },
  );
  return mapDeploymentPlan(await readJson<ApiScmDeploymentPlanResponse>(response));
}

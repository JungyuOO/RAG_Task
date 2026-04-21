export type ScmProvider = "github" | "gitlab";

export type ScmConnectionRecord = {
  scmConnectionId: string;
  workspaceId: string;
  provider: ScmProvider;
  hostUrl: string;
  authType: string;
  accountLabel: string;
  loginName: string;
  scopes: string[];
  secretRef: string;
  status: string;
  createdAt: string;
  updatedAt: string;
};

export type ScmRepositoryRecord = {
  repositoryId: string;
  workspaceId: string;
  scmConnectionId: string;
  repoFullName: string;
  defaultBranch: string;
  configPath: string;
  deliveryMode: string;
  manifestKind: string;
  targetClusterUrl: string;
  targetNamespace: string;
  autoDeployEnabled: boolean;
  syncStatus: string;
  createdAt: string;
  updatedAt: string;
};

export type ScmConnectionCreateRequest = {
  provider: ScmProvider;
  hostUrl?: string;
  authType?: string;
  accountLabel?: string;
};

export type ScmRepositoryCreateRequest = {
  scmConnectionId: string;
  repoFullName: string;
  defaultBranch?: string;
  configPath?: string;
  deliveryMode?: string;
  manifestKind?: string;
  targetClusterUrl?: string;
  targetNamespace?: string;
  autoDeployEnabled?: boolean;
};

export type ScmRepositoryUpdateRequest = {
  defaultBranch?: string;
  configPath?: string;
  deliveryMode?: string;
  manifestKind?: string;
  targetClusterUrl?: string;
  targetNamespace?: string;
  autoDeployEnabled?: boolean;
};

export type ScmDeploymentPlanRequest = {
  resourceKind?: string;
  resourceName: string;
  targetNamespace?: string;
  replicas?: number;
  imageTag?: string;
  configKey?: string;
  reason?: string;
};

export type ScmDeploymentPlanResponse = {
  repositoryId: string;
  workspaceId: string;
  repoFullName: string;
  defaultBranch: string;
  configPath: string;
  deliveryMode: string;
  manifestKind: string;
  targetClusterUrl: string;
  targetNamespace: string;
  autoDeployEnabled: boolean;
  filesToChange: string[];
  suggestedUpdates: string[];
  triggerKind: string;
  summary: string;
  commitTitle: string;
  commitBody: string;
  requiresPullRequest: boolean;
  nextStep: string;
};

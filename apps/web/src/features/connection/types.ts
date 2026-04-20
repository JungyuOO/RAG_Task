export type OcpAuthMode = "token" | "password" | "oauth_future";

export interface OcpConnectionRequest {
  clusterUrl: string;
  authMode: OcpAuthMode;
  verifySsl: boolean;
  defaultNamespace?: string;
  displayName?: string;
  saveProfile?: boolean;
  token?: string;
  username?: string;
  password?: string;
}

export interface OcpConnectionProfile {
  connectionId: string;
  displayName: string;
  clusterUrl: string;
  authMode: OcpAuthMode;
  verifySsl: boolean;
  defaultNamespace: string;
  usernameHint: string;
  secretRef: string;
  saveProfile: boolean;
  status: string;
  lastVerifiedAt: string;
  expiresAt: string;
}

export interface SavedOcpConnectionProfile {
  id: string;
  displayName: string;
  clusterUrl: string;
  authMode: OcpAuthMode;
  verifySsl: boolean;
  defaultNamespace: string;
  usernameHint: string;
  lastResolvedUser: string;
  lastConnectionId: string;
  lastStatus: string;
  lastConnectedAt: string;
  lastVerifiedAt: string;
  request: OcpConnectionRequest;
}

export interface OcpConnectionStatusResponse {
  connected: boolean;
  connection: OcpConnectionProfile | null;
  message: string;
}

export interface OcpConnectionTestResult {
  success: boolean;
  connectionId: string;
  clusterUrl: string;
  authMode: OcpAuthMode;
  resolvedUser: string;
  resolvedGroups: string[];
  resolvedRoles: string[];
  identitySource: string;
  permissionHints: Record<string, boolean>;
  rbacEvidence: string[];
  rbacRulesIncomplete: boolean;
  rbacEvaluationError: string;
  secretBackend: string;
  secretVersion: string;
  secretCreatedAt: string;
  secretLeaseRenewable: boolean;
  secretLeaseTtlSeconds: number;
  secretLeaseExpiresAt: string;
  secretRotationSupported: boolean;
  secretAutoRenewApplied: boolean;
  secretAutoRenewThresholdSeconds: number;
  secretRenewMessage: string;
  resolvedNamespace: string;
  expiresAt: string;
  message: string;
  error: string;
}

export interface OcpLeaseSchedulerStatus {
  enabled: boolean;
  running: boolean;
  intervalSeconds: number;
  lastRunAt: string;
  lastSuccessAt: string;
  lastFailureAt: string;
  lastError: string;
  consecutiveFailures: number;
  nextRunDelaySeconds: number;
  alertLevel: string;
  profilesChecked: number;
  renewalsApplied: number;
  recentFailures: string[];
  alertDeliveryEnabled: boolean;
  alertTarget: string;
  alertDispatchCount: number;
  lastAlertAt: string;
  lastAlertLevel: string;
  lastAlertStatus: string;
  lastAlertError: string;
}

export interface OcpLiveResourceSummary {
  name: string;
  namespace: string;
  kind: string;
  createdAt: string;
  phase: string;
  nodeName: string;
  readyReplicas: number;
  replicas: number;
  type: string;
  clusterIp: string;
  host: string;
  to: string;
}

export interface OcpLiveNamespaceListResponse {
  connectionId: string;
  clusterUrl: string;
  count: number;
  items: string[];
}

export interface OcpLiveResourceListResponse {
  connectionId: string;
  clusterUrl: string;
  resource: string;
  namespace: string;
  count: number;
  items: OcpLiveResourceSummary[];
}

export interface OcpLiveResourceDetailResponse {
  connectionId: string;
  clusterUrl: string;
  resource: string;
  namespace: string;
  name: string;
  kind: string;
  manifestYaml: string;
  manifestJson: Record<string, unknown>;
}

export interface OcpOverviewResponse {
  connectionId: string;
  clusterUrl: string;
  defaultNamespace: string;
  namespaceCount: number;
  namespaceSample: string[];
  resourceCounts: Record<string, number>;
  message: string;
}

export interface OcpMetricPoint {
  timestamp: number;
  value: number;
}

export interface OcpMetricSeries {
  metricId: string;
  label: string;
  unit: string;
  currentValue: number;
  capacityValue: number;
  availableValue: number;
  points: OcpMetricPoint[];
}

export interface OcpDashboardMetricsResponse {
  connectionId: string;
  clusterUrl: string;
  window: string;
  step: string;
  series: OcpMetricSeries[];
}



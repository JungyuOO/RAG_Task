import type {
  OcpConnectionProfile,
  OcpConnectionRequest,
  OcpConnectionStatusResponse,
  OcpConnectionTestResult,
  OcpLeaseSchedulerStatus,
} from "@/features/connection/types";

type ApiConnectionProfile = {
  connection_id: string;
  display_name: string;
  cluster_url: string;
  auth_mode: OcpConnectionProfile["authMode"];
  verify_ssl: boolean;
  default_namespace: string;
  username_hint: string;
  secret_ref: string;
  save_profile: boolean;
  status: string;
  last_verified_at: string;
  expires_at: string;
};

type ApiConnectionStatusResponse = {
  connected: boolean;
  connection: ApiConnectionProfile | null;
  message: string;
};

type ApiConnectionTestResult = {
  success: boolean;
  connection_id: string;
  cluster_url: string;
  auth_mode: OcpConnectionTestResult["authMode"];
  resolved_user: string;
  resolved_groups: string[];
  resolved_roles: string[];
  identity_source: string;
  permission_hints: Record<string, boolean>;
  rbac_evidence: string[];
  rbac_rules_incomplete: boolean;
  rbac_evaluation_error: string;
  secret_backend: string;
  secret_version: string;
  secret_created_at: string;
  secret_lease_renewable: boolean;
  secret_lease_ttl_seconds: number;
  secret_lease_expires_at: string;
  secret_rotation_supported: boolean;
  secret_auto_renew_applied: boolean;
  secret_auto_renew_threshold_seconds: number;
  secret_renew_message: string;
  resolved_namespace: string;
  expires_at: string;
  message: string;
  error: string;
};

type ApiLeaseSchedulerStatus = {
  enabled: boolean;
  running: boolean;
  interval_seconds: number;
  last_run_at: string;
  last_success_at: string;
  last_failure_at: string;
  last_error: string;
  consecutive_failures: number;
  next_run_delay_seconds: number;
  alert_level: string;
  profiles_checked: number;
  renewals_applied: number;
  recent_failures: string[];
  alert_delivery_enabled: boolean;
  alert_target: string;
  alert_dispatch_count: number;
  last_alert_at: string;
  last_alert_level: string;
  last_alert_status: string;
  last_alert_error: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

function mapProfile(input: ApiConnectionProfile): OcpConnectionProfile {
  return {
    connectionId: input.connection_id,
    displayName: input.display_name,
    clusterUrl: input.cluster_url,
    authMode: input.auth_mode,
    verifySsl: input.verify_ssl,
    defaultNamespace: input.default_namespace,
    usernameHint: input.username_hint,
    secretRef: input.secret_ref,
    saveProfile: input.save_profile,
    status: input.status,
    lastVerifiedAt: input.last_verified_at,
    expiresAt: input.expires_at,
  };
}

function mapStatus(input: ApiConnectionStatusResponse): OcpConnectionStatusResponse {
  return {
    connected: input.connected,
    connection: input.connection ? mapProfile(input.connection) : null,
    message: input.message,
  };
}

function mapTestResult(input: ApiConnectionTestResult): OcpConnectionTestResult {
  return {
    success: input.success,
    connectionId: input.connection_id,
    clusterUrl: input.cluster_url,
    authMode: input.auth_mode,
    resolvedUser: input.resolved_user,
    resolvedGroups: input.resolved_groups ?? [],
    resolvedRoles: input.resolved_roles ?? [],
    identitySource: input.identity_source ?? "",
    permissionHints: input.permission_hints ?? {},
    rbacEvidence: input.rbac_evidence ?? [],
    rbacRulesIncomplete: input.rbac_rules_incomplete ?? false,
    rbacEvaluationError: input.rbac_evaluation_error ?? "",
    secretBackend: input.secret_backend ?? "",
    secretVersion: input.secret_version ?? "",
    secretCreatedAt: input.secret_created_at ?? "",
    secretLeaseRenewable: input.secret_lease_renewable ?? false,
    secretLeaseTtlSeconds: input.secret_lease_ttl_seconds ?? 0,
    secretLeaseExpiresAt: input.secret_lease_expires_at ?? "",
    secretRotationSupported: input.secret_rotation_supported ?? false,
    secretAutoRenewApplied: input.secret_auto_renew_applied ?? false,
    secretAutoRenewThresholdSeconds: input.secret_auto_renew_threshold_seconds ?? 0,
    secretRenewMessage: input.secret_renew_message ?? "",
    resolvedNamespace: input.resolved_namespace,
    expiresAt: input.expires_at,
    message: input.message,
    error: input.error,
  };
}

function mapLeaseSchedulerStatus(input: ApiLeaseSchedulerStatus): OcpLeaseSchedulerStatus {
  return {
    enabled: input.enabled,
    running: input.running,
    intervalSeconds: input.interval_seconds,
    lastRunAt: input.last_run_at,
    lastSuccessAt: input.last_success_at,
    lastFailureAt: input.last_failure_at,
    lastError: input.last_error,
    consecutiveFailures: input.consecutive_failures,
    nextRunDelaySeconds: input.next_run_delay_seconds,
    alertLevel: input.alert_level,
    profilesChecked: input.profiles_checked,
    renewalsApplied: input.renewals_applied,
    recentFailures: input.recent_failures ?? [],
    alertDeliveryEnabled: input.alert_delivery_enabled ?? false,
    alertTarget: input.alert_target ?? "",
    alertDispatchCount: input.alert_dispatch_count ?? 0,
    lastAlertAt: input.last_alert_at ?? "",
    lastAlertLevel: input.last_alert_level ?? "none",
    lastAlertStatus: input.last_alert_status ?? "",
    lastAlertError: input.last_alert_error ?? "",
  };
}

function serializeRequest(request: OcpConnectionRequest) {
  return {
    cluster_url: request.clusterUrl,
    auth_mode: request.authMode,
    verify_ssl: request.verifySsl,
    default_namespace: request.defaultNamespace ?? "",
    display_name: request.displayName ?? "",
    save_profile: request.saveProfile ?? false,
    token: request.token ?? null,
    username: request.username ?? "",
    password: request.password ?? null,
  };
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

export async function connectOcp(request: OcpConnectionRequest): Promise<OcpConnectionStatusResponse> {
  const response = await fetch(apiUrl("/api/v1/auth/ocp/connect"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(serializeRequest(request)),
  });
  return mapStatus(await readJson<ApiConnectionStatusResponse>(response));
}

export async function getOcpConnectionStatus(connectionId: string): Promise<OcpConnectionStatusResponse> {
  const response = await fetch(apiUrl(`/api/v1/auth/ocp/status/${connectionId}`));
  return mapStatus(await readJson<ApiConnectionStatusResponse>(response));
}

export async function testOcpConnection(connectionId: string): Promise<OcpConnectionTestResult> {
  const response = await fetch(apiUrl("/api/v1/auth/ocp/test"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connection_id: connectionId }),
  });
  return mapTestResult(await readJson<ApiConnectionTestResult>(response));
}

export async function disconnectOcp(connectionId: string): Promise<OcpConnectionStatusResponse> {
  const response = await fetch(apiUrl("/api/v1/auth/ocp/disconnect"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connection_id: connectionId }),
  });
  return mapStatus(await readJson<ApiConnectionStatusResponse>(response));
}

export async function refreshOcpConnectionLease(connectionId: string): Promise<OcpConnectionTestResult> {
  const response = await fetch(apiUrl("/api/v1/auth/ocp/lease/refresh"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connection_id: connectionId }),
  });
  return mapTestResult(await readJson<ApiConnectionTestResult>(response));
}

export async function getOcpLeaseSchedulerStatus(): Promise<OcpLeaseSchedulerStatus> {
  const response = await fetch(apiUrl("/api/v1/auth/ocp/lease/status"));
  return mapLeaseSchedulerStatus(await readJson<ApiLeaseSchedulerStatus>(response));
}



import type { OcpActionAuditRecord } from "@/features/actions/types";
import type { OcpActionExecutionRecord } from "@/features/actions/types";
import type { OcpActionPreviewRequest, OcpActionPreviewResponse } from "@/features/actions/types";
import type { OcpActionRequestCreateRequest, OcpActionRequestRecord } from "@/features/actions/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

type ApiOcpActionPreviewRequest = {
  connection_id: string;
  actor_id?: string;
  actor_roles?: string[];
  action_type: string;
  namespace?: string;
  resource_name?: string;
  replicas?: number;
  reason?: string;
  break_glass?: boolean;
  break_glass_reason?: string;
  break_glass_ticket?: string;
  manifest_yaml?: string;
  resource_version?: string | null;
};

type ApiOcpActionPreviewResponse = {
  connection_id: string;
  action_type: OcpActionPreviewResponse["actionType"];
  namespace: string;
  resource_name: string;
  allowed: boolean;
  risk_level: string;
  summary: string;
  preview_command: string;
  break_glass: boolean;
  break_glass_reason: string;
  break_glass_ticket: string;
  required_approvals: number;
  approval_strategy: string;
  requester_roles: string[];
  approver_roles: string[];
  executor_roles: string[];
  approval_rules: string[];
  policy_checks: string[];
  blocked_reasons: string[];
  validation_messages: string[];
  diff_unified: string;
  dry_run_status: OcpActionPreviewResponse["dryRunStatus"];
  dry_run_messages: string[];
  next_step: string;
};

type ApiOcpActionRequestRecord = {
  request_id: string;
  status: OcpActionRequestRecord["status"];
  preview: ApiOcpActionPreviewResponse;
  requested_by: string;
  requested_roles: string[];
  required_approvals: number;
  approval_count: number;
  approver_ids: string[];
  approver_role_map: Record<string, string[]>;
  reason: string;
  decision_note: string;
  created_at: string;
  updated_at: string;
};

type ApiOcpActionExecutionRecord = {
  execution_id: string;
  request_id: string;
  status: OcpActionExecutionRecord["status"];
  execution_mode: string;
  simulated: boolean;
  preview: ApiOcpActionPreviewResponse;
  summary: string;
  preflight_checks: string[];
  output_lines: string[];
  error: string;
  created_at: string;
  updated_at: string;
};

type ApiOcpActionAuditRecord = {
  event_id: string;
  event_type: OcpActionAuditRecord["eventType"];
  actor_id: string;
  request_id: string;
  execution_id: string;
  action_type: OcpActionAuditRecord["actionType"];
  namespace: string;
  resource_name: string;
  risk_level: string;
  decision_note: string;
  details: Record<string, unknown>;
  created_at: string;
};

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

function serializePreviewRequest(input: OcpActionPreviewRequest): ApiOcpActionPreviewRequest {
  return {
    connection_id: input.connectionId,
    actor_id: input.actorId ?? "ui-local",
    actor_roles: input.actorRoles ?? [],
    action_type: input.actionType,
    namespace: input.namespace ?? "",
    resource_name: input.resourceName ?? "",
    replicas: input.replicas ?? 1,
    reason: input.reason ?? "",
    break_glass: input.breakGlass ?? false,
    break_glass_reason: input.breakGlassReason ?? "",
    break_glass_ticket: input.breakGlassTicket ?? "",
    manifest_yaml: input.manifestYaml ?? "",
    resource_version: input.resourceVersion ?? null,
  };
}

function mapPreview(input: ApiOcpActionPreviewResponse): OcpActionPreviewResponse {
  return {
    connectionId: input.connection_id,
    actionType: input.action_type,
    namespace: input.namespace,
    resourceName: input.resource_name,
    allowed: input.allowed,
    riskLevel: input.risk_level,
    summary: input.summary,
    previewCommand: input.preview_command,
    breakGlass: input.break_glass ?? false,
    breakGlassReason: input.break_glass_reason ?? "",
    breakGlassTicket: input.break_glass_ticket ?? "",
    requiredApprovals: input.required_approvals,
    approvalStrategy: input.approval_strategy,
    requesterRoles: input.requester_roles,
    approverRoles: input.approver_roles,
    executorRoles: input.executor_roles,
    approvalRules: input.approval_rules,
    policyChecks: input.policy_checks,
    blockedReasons: input.blocked_reasons,
    validationMessages: input.validation_messages,
    diffUnified: input.diff_unified ?? "",
    dryRunStatus: input.dry_run_status ?? "skipped",
    dryRunMessages: input.dry_run_messages ?? [],
    nextStep: input.next_step,
  };
}

function mapRequestRecord(input: ApiOcpActionRequestRecord): OcpActionRequestRecord {
  return {
    requestId: input.request_id,
    status: input.status,
    preview: mapPreview(input.preview),
    requestedBy: input.requested_by,
    requestedRoles: input.requested_roles,
    requiredApprovals: input.required_approvals,
    approvalCount: input.approval_count,
    approverIds: input.approver_ids,
    approverRoleMap: input.approver_role_map,
    reason: input.reason,
    decisionNote: input.decision_note,
    createdAt: input.created_at,
    updatedAt: input.updated_at,
  };
}

function mapExecutionRecord(input: ApiOcpActionExecutionRecord): OcpActionExecutionRecord {
  return {
    executionId: input.execution_id,
    requestId: input.request_id,
    status: input.status,
    executionMode: input.execution_mode,
    simulated: input.simulated,
    preview: mapPreview(input.preview),
    summary: input.summary,
    preflightChecks: input.preflight_checks,
    outputLines: input.output_lines,
    error: input.error,
    createdAt: input.created_at,
    updatedAt: input.updated_at,
  };
}

function mapAuditRecord(input: ApiOcpActionAuditRecord): OcpActionAuditRecord {
  return {
    eventId: input.event_id,
    eventType: input.event_type,
    actorId: input.actor_id,
    requestId: input.request_id,
    executionId: input.execution_id,
    actionType: input.action_type,
    namespace: input.namespace,
    resourceName: input.resource_name,
    riskLevel: input.risk_level,
    decisionNote: input.decision_note,
    details: input.details,
    createdAt: input.created_at,
  };
}

export async function previewOcpAction(request: OcpActionPreviewRequest): Promise<OcpActionPreviewResponse> {
  const response = await fetch(apiUrl("/api/v1/actions/preview"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(serializePreviewRequest(request)),
  });
  return mapPreview(await readJson<ApiOcpActionPreviewResponse>(response));
}

export async function createOcpActionRequest(request: OcpActionRequestCreateRequest): Promise<OcpActionRequestRecord> {
  const response = await fetch(apiUrl("/api/v1/actions/requests"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      connection_id: request.connectionId,
      actor_id: request.actorId ?? "ui-local",
      actor_roles: request.actorRoles ?? [],
      action_type: request.actionType,
      namespace: request.namespace ?? "",
      resource_name: request.resourceName ?? "",
      replicas: request.replicas ?? 1,
      reason: request.reason ?? "",
      break_glass: request.breakGlass ?? false,
      break_glass_reason: request.breakGlassReason ?? "",
      break_glass_ticket: request.breakGlassTicket ?? "",
      manifest_yaml: request.manifestYaml ?? "",
      resource_version: request.resourceVersion ?? null,
    }),
  });
  return mapRequestRecord(await readJson<ApiOcpActionRequestRecord>(response));
}

export async function listOcpActionRequests(limit = 20): Promise<OcpActionRequestRecord[]> {
  const response = await fetch(apiUrl(`/api/v1/actions/requests?limit=${limit}`));
  const payload = await readJson<{ items: ApiOcpActionRequestRecord[] }>(response);
  return payload.items.map(mapRequestRecord);
}

export async function approveOcpActionRequest(
  requestId: string,
  actorId: string,
  actorRoles: string[],
  decisionNote = "",
): Promise<OcpActionRequestRecord> {
  const response = await fetch(apiUrl(`/api/v1/actions/requests/${requestId}/approve`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision_note: decisionNote, actor_id: actorId, actor_roles: actorRoles }),
  });
  return mapRequestRecord(await readJson<ApiOcpActionRequestRecord>(response));
}

export async function rejectOcpActionRequest(
  requestId: string,
  actorId: string,
  actorRoles: string[],
  decisionNote = "",
): Promise<OcpActionRequestRecord> {
  const response = await fetch(apiUrl(`/api/v1/actions/requests/${requestId}/reject`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision_note: decisionNote, actor_id: actorId, actor_roles: actorRoles }),
  });
  return mapRequestRecord(await readJson<ApiOcpActionRequestRecord>(response));
}

export async function executeOcpActionRequest(
  requestId: string,
  actorId: string,
  actorRoles: string[],
  force = false,
): Promise<OcpActionExecutionRecord> {
  const response = await fetch(apiUrl(`/api/v1/actions/requests/${requestId}/execute`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, actor_roles: actorRoles, execution_note: "requested from UI", force }),
  });
  return mapExecutionRecord(await readJson<ApiOcpActionExecutionRecord>(response));
}

export async function listOcpActionExecutions(limit = 20): Promise<OcpActionExecutionRecord[]> {
  const response = await fetch(apiUrl(`/api/v1/actions/executions?limit=${limit}`));
  const payload = await readJson<{ items: ApiOcpActionExecutionRecord[] }>(response);
  return payload.items.map(mapExecutionRecord);
}

export async function listOcpActionAudit(limit = 20): Promise<OcpActionAuditRecord[]> {
  const response = await fetch(apiUrl(`/api/v1/actions/audit?limit=${limit}`));
  const payload = await readJson<{ items: ApiOcpActionAuditRecord[] }>(response);
  return payload.items.map(mapAuditRecord);
}



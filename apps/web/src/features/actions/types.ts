export type OcpActionType = "scale_deployment" | "rollout_restart" | "log_bundle" | "yaml_apply";

export interface OcpActionPreviewRequest {
  connectionId: string;
  actorId?: string;
  actorRoles?: string[];
  actionType: OcpActionType;
  namespace?: string;
  resourceName?: string;
  replicas?: number;
  reason?: string;
  breakGlass?: boolean;
  breakGlassReason?: string;
  breakGlassTicket?: string;
  manifestYaml?: string;
  resourceVersion?: string;
}

export interface OcpActionPreviewResponse {
  connectionId: string;
  actionType: OcpActionType;
  namespace: string;
  resourceName: string;
  allowed: boolean;
  riskLevel: string;
  summary: string;
  previewCommand: string;
  breakGlass: boolean;
  breakGlassReason: string;
  breakGlassTicket: string;
  requiredApprovals: number;
  approvalStrategy: string;
  requesterRoles: string[];
  approverRoles: string[];
  executorRoles: string[];
  approvalRules: string[];
  policyChecks: string[];
  blockedReasons: string[];
  validationMessages: string[];
  diffUnified: string;
  dryRunStatus: "ok" | "rejected" | "skipped";
  dryRunMessages: string[];
  nextStep: string;
}

export type OcpActionRequestStatus = "pending" | "approved" | "rejected";

export interface OcpActionRequestRecord {
  requestId: string;
  status: OcpActionRequestStatus;
  preview: OcpActionPreviewResponse;
  requestedBy: string;
  requestedRoles: string[];
  requiredApprovals: number;
  approvalCount: number;
  approverIds: string[];
  approverRoleMap: Record<string, string[]>;
  reason: string;
  decisionNote: string;
  createdAt: string;
  updatedAt: string;
}

export interface OcpActionRequestCreateRequest {
  connectionId: string;
  actorId?: string;
  actorRoles?: string[];
  actionType: OcpActionType;
  namespace?: string;
  resourceName?: string;
  replicas?: number;
  reason?: string;
  breakGlass?: boolean;
  breakGlassReason?: string;
  breakGlassTicket?: string;
  manifestYaml?: string;
  resourceVersion?: string;
}

export type OcpActionExecutionStatus = "succeeded" | "failed";

export interface OcpActionExecutionRecord {
  executionId: string;
  requestId: string;
  status: OcpActionExecutionStatus;
  executionMode: string;
  simulated: boolean;
  preview: OcpActionPreviewResponse;
  summary: string;
  preflightChecks: string[];
  outputLines: string[];
  error: string;
  createdAt: string;
  updatedAt: string;
}

export type OcpActionAuditEventType =
  | "request_created"
  | "request_approved"
  | "request_rejected"
  | "execution_succeeded"
  | "execution_failed";

export interface OcpActionAuditRecord {
  eventId: string;
  eventType: OcpActionAuditEventType;
  actorId: string;
  requestId: string;
  executionId: string;
  actionType: OcpActionType;
  namespace: string;
  resourceName: string;
  riskLevel: string;
  decisionNote: string;
  details: Record<string, unknown>;
  createdAt: string;
}


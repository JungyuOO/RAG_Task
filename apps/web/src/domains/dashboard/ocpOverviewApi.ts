import type { WorkspaceRecommendationRecord } from "@/domains/dashboard/types";
import { getOcpDashboardMetrics, getOcpOverview } from "@/domains/resources/ocpResourcesApi";

export { getOcpDashboardMetrics, getOcpOverview };

type ApiRecommendationRecord = {
  recommendation_id: string;
  workspace_id: string;
  connection_id: string;
  namespace: string;
  resource_kind: string;
  resource_name: string;
  recommendation_type: string;
  risk_level: string;
  summary: string;
  rationale: string;
  created_at: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

function mapRecommendation(input: ApiRecommendationRecord): WorkspaceRecommendationRecord {
  return {
    recommendationId: input.recommendation_id,
    workspaceId: input.workspace_id,
    connectionId: input.connection_id,
    namespace: input.namespace,
    resourceKind: input.resource_kind,
    resourceName: input.resource_name,
    recommendationType: input.recommendation_type,
    riskLevel: input.risk_level,
    summary: input.summary,
    rationale: input.rationale,
    createdAt: input.created_at,
  };
}

export async function getWorkspaceRecommendations(workspaceId: string, limit = 10): Promise<WorkspaceRecommendationRecord[]> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/recommendations?limit=${limit}`));
  const payload = await readJson<{ items: ApiRecommendationRecord[] }>(response);
  return (payload.items ?? []).map(mapRecommendation);
}

export async function refreshWorkspaceRecommendations(
  workspaceId: string,
  connectionId: string,
): Promise<WorkspaceRecommendationRecord[]> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/recommendations/refresh`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connection_id: connectionId }),
  });
  const payload = await readJson<{ items: ApiRecommendationRecord[] }>(response);
  return (payload.items ?? []).map(mapRecommendation);
}

import type { WorkspaceCreateRequest, WorkspaceRecord } from "@/domains/workspaces/types";

type ApiWorkspaceRecord = {
  workspace_id: string;
  name: string;
  slug: string;
  industry: string;
  environment: string;
  created_at: string;
  updated_at: string;
};

type ApiWorkspaceListResponse = {
  items: ApiWorkspaceRecord[];
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

function mapWorkspace(input: ApiWorkspaceRecord): WorkspaceRecord {
  return {
    workspaceId: input.workspace_id,
    name: input.name,
    slug: input.slug,
    industry: input.industry ?? "",
    environment: input.environment ?? "",
    createdAt: input.created_at,
    updatedAt: input.updated_at,
  };
}

export async function listWorkspaces(): Promise<WorkspaceRecord[]> {
  const response = await fetch(apiUrl("/api/v1/workspaces"));
  const payload = await readJson<ApiWorkspaceListResponse>(response);
  return (payload.items ?? []).map(mapWorkspace);
}

export async function createWorkspace(request: WorkspaceCreateRequest): Promise<WorkspaceRecord> {
  const response = await fetch(apiUrl("/api/v1/workspaces"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: request.name,
      slug: request.slug ?? "",
      industry: request.industry ?? "",
      environment: request.environment ?? "",
    }),
  });
  return mapWorkspace(await readJson<ApiWorkspaceRecord>(response));
}

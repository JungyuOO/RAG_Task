import type { WorkspaceModelProfile, WorkspaceModelProfileUpdateRequest } from "@/domains/models/types";

type ApiWorkspaceModelProfile = {
  workspace_id: string;
  chat_provider: string;
  chat_base_url: string;
  chat_model: string;
  chat_api_key_mode: string;
  embedding_provider: string;
  embedding_base_url: string;
  embedding_model: string;
  embedding_api_key_mode: string;
  updated_at: string;
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

function mapProfile(input: ApiWorkspaceModelProfile): WorkspaceModelProfile {
  return {
    workspaceId: input.workspace_id,
    chatProvider: input.chat_provider,
    chatBaseUrl: input.chat_base_url,
    chatModel: input.chat_model,
    chatApiKeyMode: input.chat_api_key_mode,
    embeddingProvider: input.embedding_provider,
    embeddingBaseUrl: input.embedding_base_url,
    embeddingModel: input.embedding_model,
    embeddingApiKeyMode: input.embedding_api_key_mode,
    updatedAt: input.updated_at,
  };
}

export async function getDefaultWorkspaceModelProfile(workspaceId: string): Promise<WorkspaceModelProfile> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/models/default`));
  return mapProfile(await readJson<ApiWorkspaceModelProfile>(response));
}

export async function updateDefaultWorkspaceModelProfile(
  workspaceId: string,
  request: WorkspaceModelProfileUpdateRequest,
): Promise<WorkspaceModelProfile> {
  const response = await fetch(apiUrl(`/api/v1/workspaces/${workspaceId}/models/default`), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chat_provider: request.chatProvider,
      chat_base_url: request.chatBaseUrl,
      chat_model: request.chatModel,
      chat_api_key_mode: request.chatApiKeyMode,
      embedding_provider: request.embeddingProvider,
      embedding_base_url: request.embeddingBaseUrl,
      embedding_model: request.embeddingModel,
      embedding_api_key_mode: request.embeddingApiKeyMode,
    }),
  });
  return mapProfile(await readJson<ApiWorkspaceModelProfile>(response));
}

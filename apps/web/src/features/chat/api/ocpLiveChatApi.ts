import type { OcpLiveChatResponse } from "@/features/chat/types";
import type { CopilotChatSourceItem } from "@/features/chat/types";
import type { OcpLiveResourceSummary } from "@/features/connection/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

type ApiOcpLiveResourceSummary = {
  name: string;
  namespace: string;
  kind: string;
  created_at: string;
  phase: string;
  node_name: string;
  ready_replicas: number;
  replicas: number;
  type: string;
  cluster_ip: string;
  host: string;
  to: string;
};

type ApiOcpLiveChatResponse = {
  connection_id: string;
  cluster_url: string;
  mode: string;
  resource: string;
  namespace: string;
  answer: string;
  items: ApiOcpLiveResourceSummary[];
  sources?: Array<{
    source_type: string;
    label: string;
    source_path: string;
    relative_source_path: string;
    page_number: number | null;
    chunk_id: string;
    namespace: string;
    kind: string;
    score: number;
    provenance: string[];
    metadata: Record<string, unknown>;
  }>;
};

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

function mapResourceSummary(input: ApiOcpLiveResourceSummary): OcpLiveResourceSummary {
  return {
    name: input.name,
    namespace: input.namespace,
    kind: input.kind,
    createdAt: input.created_at,
    phase: input.phase,
    nodeName: input.node_name,
    readyReplicas: input.ready_replicas,
    replicas: input.replicas,
    type: input.type,
    clusterIp: input.cluster_ip,
    host: input.host,
    to: input.to,
  };
}

function mapChatResponse(input: ApiOcpLiveChatResponse): OcpLiveChatResponse {
  return {
    connectionId: input.connection_id,
    clusterUrl: input.cluster_url,
    mode: input.mode,
    resource: input.resource,
    namespace: input.namespace,
    answer: input.answer,
    items: input.items.map(mapResourceSummary),
    sources: (input.sources ?? []).map(
      (source): CopilotChatSourceItem => ({
        sourceType: source.source_type,
        label: source.label,
        sourcePath: source.source_path,
        relativeSourcePath: source.relative_source_path,
        pageNumber: source.page_number,
        chunkId: source.chunk_id,
        namespace: source.namespace,
        kind: source.kind,
        score: source.score,
        provenance: source.provenance,
        metadata: source.metadata,
      }),
    ),
  };
}

export async function sendOcpLiveChat(
  connectionId: string,
  message: string,
  namespace = "",
): Promise<OcpLiveChatResponse> {
  const response = await fetch(apiUrl("/api/v1/chat/live"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      connection_id: connectionId,
      message,
      namespace,
    }),
  });
  return mapChatResponse(await readJson<ApiOcpLiveChatResponse>(response));
}



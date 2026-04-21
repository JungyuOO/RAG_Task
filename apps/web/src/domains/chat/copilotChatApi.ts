import type {
  CopilotChatArtifact,
  CopilotChatArtifactItem,
  CopilotCitationMapItem,
  CopilotChatResponse,
  CopilotChatSourceItem,
  CopilotChatStage,
} from "@/domains/chat/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

type ApiCopilotChatSourceItem = {
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
};

type ApiCopilotChatResponse = {
  lane: string;
  mode: string;
  fallback_used: boolean;
  preview_ready: boolean;
  answer: string;
  sources: ApiCopilotChatSourceItem[];
  artifacts?: ApiCopilotChatArtifact[];
  citation_map?: ApiCopilotCitationMapItem[];
};

type ApiCopilotCitationMapItem = {
  citation_number: number;
  source_index: number;
  chunk_id: string;
  section_title: string;
  supporting_text: string;
  command_text: string;
};

type ApiCopilotChatArtifactItem = {
  name: string;
  kind: string;
  namespace: string;
  resource: string;
  phase: string;
  type: string;
  host: string;
  to: string;
  node_name: string;
  cluster_ip: string;
  action_target: Record<string, unknown>;
  metadata: Record<string, unknown>;
};

type ApiCopilotChatArtifact = {
  artifact_type: string;
  title: string;
  description: string;
  resource: string;
  namespace: string;
  resource_name: string;
  payload: Record<string, unknown>;
  items: ApiCopilotChatArtifactItem[];
};

type ApiCopilotChatStage = {
  key: string;
  label: string;
  detail: string;
  status: "running" | "completed" | "error";
};

type ApiCopilotChatStageEvent = {
  type: "stage";
  stage: ApiCopilotChatStage;
};

type ApiCopilotChatAnswerDeltaEvent = {
  type: "answer_delta";
  delta: string;
};

type ApiCopilotChatResultEvent = {
  type: "result";
  response: ApiCopilotChatResponse;
};

type ApiCopilotChatErrorEvent = {
  type: "error";
  message: string;
  status_code?: number;
};

type ApiCopilotChatStreamEvent =
  | ApiCopilotChatStageEvent
  | ApiCopilotChatAnswerDeltaEvent
  | ApiCopilotChatResultEvent
  | ApiCopilotChatErrorEvent;
type CopilotChatHistoryTurn = {
  role: "user" | "assistant";
  text: string;
  lane?: string;
  sourcePaths?: string[];
  resourceNames?: string[];
  namespace?: string;
};

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

function mapSource(input: ApiCopilotChatSourceItem): CopilotChatSourceItem {
  return {
    sourceType: input.source_type,
    label: input.label,
    sourcePath: input.source_path,
    relativeSourcePath: input.relative_source_path,
    pageNumber: input.page_number,
    chunkId: input.chunk_id,
    namespace: input.namespace,
    kind: input.kind,
    score: input.score,
    provenance: input.provenance,
    metadata: input.metadata,
  };
}

function mapStage(input: ApiCopilotChatStage): CopilotChatStage {
  return {
    key: input.key,
    label: input.label,
    detail: input.detail,
    status: input.status,
  };
}

function mapCitationMapItem(input: ApiCopilotCitationMapItem): CopilotCitationMapItem {
  return {
    citationNumber: input.citation_number,
    sourceIndex: input.source_index,
    chunkId: input.chunk_id,
    sectionTitle: input.section_title,
    supportingText: input.supporting_text,
    commandText: input.command_text,
  };
}

function mapArtifactItem(input: ApiCopilotChatArtifactItem): CopilotChatArtifactItem {
  return {
    name: input.name,
    kind: input.kind,
    namespace: input.namespace,
    resource: input.resource,
    phase: input.phase,
    type: input.type,
    host: input.host,
    to: input.to,
    nodeName: input.node_name,
    clusterIp: input.cluster_ip,
    actionTarget: input.action_target ?? {},
    metadata: input.metadata ?? {},
  };
}

function mapArtifact(input: ApiCopilotChatArtifact): CopilotChatArtifact {
  return {
    artifactType: input.artifact_type,
    title: input.title,
    description: input.description,
    resource: input.resource,
    namespace: input.namespace,
    resourceName: input.resource_name,
    payload: input.payload ?? {},
    items: (input.items ?? []).map(mapArtifactItem),
  };
}

function mapResponse(input: ApiCopilotChatResponse): CopilotChatResponse {
  return {
    lane: input.lane,
    mode: input.mode,
    fallbackUsed: input.fallback_used,
    previewReady: input.preview_ready,
    answer: input.answer,
    sources: input.sources.map(mapSource),
    artifacts: (input.artifacts ?? []).map(mapArtifact),
    citationMap: (input.citation_map ?? []).map(mapCitationMapItem),
  };
}

export async function sendCopilotChat(params: {
  message: string;
  connectionId?: string;
  namespace?: string;
  history?: CopilotChatHistoryTurn[];
}): Promise<CopilotChatResponse> {
  const response = await fetch(apiUrl("/api/v1/chat/query"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: params.message,
      connection_id: params.connectionId ?? "",
      namespace: params.namespace ?? "",
      history: params.history ?? [],
    }),
  });
  return mapResponse(await readJson<ApiCopilotChatResponse>(response));
}

export async function streamCopilotChat(
  params: {
    message: string;
    connectionId?: string;
    namespace?: string;
    history?: CopilotChatHistoryTurn[];
  },
  handlers: {
    onStage?: (stage: CopilotChatStage) => void;
    onAnswerDelta?: (delta: string) => void;
    onResult?: (response: CopilotChatResponse) => void;
  },
): Promise<void> {
  const response = await fetch(apiUrl("/api/v1/chat/query/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: params.message,
      connection_id: params.connectionId ?? "",
      namespace: params.namespace ?? "",
      history: params.history ?? [],
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }

  if (!response.body) {
    handlers.onResult?.(await sendCopilotChat(params));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        const event = JSON.parse(trimmed) as ApiCopilotChatStreamEvent;
        if (event.type === "stage") {
          handlers.onStage?.(mapStage(event.stage));
          continue;
        }
        if (event.type === "answer_delta") {
          handlers.onAnswerDelta?.(event.delta);
          continue;
        }
        if (event.type === "result") {
          handlers.onResult?.(mapResponse(event.response));
          continue;
        }
        throw new Error(event.message || "Failed to stream chat response.");
      }

      if (done) {
        break;
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer.trim()) as ApiCopilotChatStreamEvent;
      if (event.type === "stage") {
        handlers.onStage?.(mapStage(event.stage));
        return;
      }
      if (event.type === "answer_delta") {
        handlers.onAnswerDelta?.(event.delta);
        return;
      }
      if (event.type === "result") {
        handlers.onResult?.(mapResponse(event.response));
        return;
      }
      throw new Error(event.message || "Failed to stream chat response.");
    }
  } finally {
    reader.releaseLock();
  }
}




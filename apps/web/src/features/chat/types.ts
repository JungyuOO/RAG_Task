export interface CopilotChatSourceItem {
  sourceType: string;
  label: string;
  sourcePath: string;
  relativeSourcePath: string;
  pageNumber: number | null;
  chunkId: string;
  namespace: string;
  kind: string;
  score: number;
  provenance: string[];
  metadata: Record<string, unknown>;
}

export interface CopilotCitationMapItem {
  citationNumber: number;
  sourceIndex: number;
  chunkId: string;
  sectionTitle: string;
  supportingText: string;
  commandText: string;
}

export interface CopilotChatArtifactItem {
  name: string;
  kind: string;
  namespace: string;
  resource: string;
  phase: string;
  type: string;
  host: string;
  to: string;
  nodeName: string;
  clusterIp: string;
  actionTarget: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface CopilotChatArtifact {
  artifactType: "resource_list" | "resource_detail" | "resource_editor" | "command_template" | string;
  title: string;
  description: string;
  resource: string;
  namespace: string;
  resourceName: string;
  payload: Record<string, unknown>;
  items: CopilotChatArtifactItem[];
}

export interface CopilotChatStage {
  key: string;
  label: string;
  detail: string;
  status: "running" | "completed" | "error";
}

export interface CopilotChatResponse {
  lane: string;
  mode: string;
  fallbackUsed: boolean;
  previewReady: boolean;
  answer: string;
  sources: CopilotChatSourceItem[];
  artifacts: CopilotChatArtifact[];
  citationMap: CopilotCitationMapItem[];
}

export interface DocumentPreviewResponse {
  sourcePath: string;
  relativeSourcePath: string;
  repoRelativePath: string;
  repoLocator: string;
  workspaceRootName: string;
  fileName: string;
  chunkId: string;
  sourceType: string;
  title: string;
  sectionTitle: string;
  sectionPath: string[];
  anchor: string;
  pageNumber: number | null;
  lineStart: number | null;
  lineEnd: number | null;
  sourceLocator: string;
  fileUri: string;
  shellOpenCommand: string;
  shellOpenLabel: string;
  vscodeUri: string;
  vscodeUriWithLine: string;
  codeCommand: string;
  snippet: string;
  lines: string[];
}

export interface OcpLiveChatResponse {
  connectionId: string;
  clusterUrl: string;
  mode: string;
  resource: string;
  namespace: string;
  answer: string;
  items: {
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
  }[];
  sources?: CopilotChatSourceItem[];
}

export type ChatTurn = {
  role: "user" | "assistant";
  text: string;
  meta?: string;
  sources?: CopilotChatSourceItem[];
  artifacts?: CopilotChatArtifact[];
  citationMap?: CopilotCitationMapItem[];
  stages?: CopilotChatStage[];
  lane?: string;
  fallbackUsed?: boolean;
  previewReady?: boolean;
};

export type ChatSessionViewportState = {
  scrollTop: number;
  lastViewedTurnIndex: number;
  selectedSourceKey?: string;
};

export type ChatSessionRecord = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  turns: ChatTurn[];
  viewport: ChatSessionViewportState;
};

export function createChatSessionRecord(): ChatSessionRecord {
  const timestamp = new Date().toISOString();
  return {
    id: `chat-${timestamp}-${Math.random().toString(36).slice(2, 8)}`,
    title: "New Session",
    createdAt: timestamp,
    updatedAt: timestamp,
    turns: [],
    viewport: {
      scrollTop: 0,
      lastViewedTurnIndex: 0,
      selectedSourceKey: "",
    },
  };
}

export function deriveChatSessionTitle(message: string): string {
  const normalized = message.replace(/\s+/g, " ").trim();
  if (!normalized) return "New Session";
  return normalized.length > 42 ? `${normalized.slice(0, 42)}...` : normalized;
}


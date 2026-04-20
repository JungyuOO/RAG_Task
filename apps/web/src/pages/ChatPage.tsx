import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  ChevronDown,
  CircleCheckBig,
  CircleDashed,
  CircleX,
  LoaderCircle,
} from "lucide-react";

import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MarkdownArticle } from "@/components/ui/MarkdownArticle";
import { ResourceList } from "@/features/resources/components/ResourceList";
import { cn } from "@/lib/cn";
import {
  ResourceYamlEditorModal,
  type LiveResourceKind,
} from "@/features/resources/components/ResourceYamlEditorModal";
import { ChatComposer } from "@/features/chat/components/ChatComposer";
import { ChatTranscript } from "@/features/chat/components/ChatTranscript";
import { streamCopilotChat } from "@/features/chat/api/copilotChatApi";
import { fetchDocumentSnippet } from "@/features/chat/api/docsPreviewApi";
import { getOcpResourceDetail } from "@/features/resources/api/ocpResourcesApi";
import type { DocumentPreviewResponse } from "@/features/chat/types";
import type {
  OcpLiveResourceDetailResponse,
  OcpLiveResourceSummary,
} from "@/features/connection/types";
import type { OcpConnectionController } from "@/features/connection/hooks/useOcpConnection";
import {
  deriveChatSessionTitle,
  type ChatSessionRecord,
  type ChatTurn,
  type CopilotChatArtifact,
  type CopilotChatArtifactItem,
  type CopilotChatSourceItem,
  type CopilotChatStage,
  type CopilotCitationMapItem,
} from "@/features/chat/types";

type ChatPageProps = {
  controller: OcpConnectionController;
  session: ChatSessionRecord;
  updateSession: (sessionId: string, updater: (current: ChatSessionRecord) => ChatSessionRecord) => void;
};

type SourceSelection = {
  source: CopilotChatSourceItem;
  preview?: DocumentPreviewResponse;
  citation?: CopilotCitationMapItem;
};

type PendingAssistantState = {
  text: string;
  meta?: string;
  sources: CopilotChatSourceItem[];
  artifacts: CopilotChatArtifact[];
  citationMap: CopilotCitationMapItem[];
  stages: CopilotChatStage[];
};

type DisplayTurn = ChatTurn & { isPending?: boolean };

type EditorTarget = {
  resource: LiveResourceKind;
  namespace: string;
  name: string;
  summary?: OcpLiveResourceSummary | null;
  detail?: OcpLiveResourceDetailResponse | null;
};

type EditorLoadingState = {
  active: boolean;
  title: string;
  detail?: string;
};

const REQUEST_SENT_STAGE: CopilotChatStage = {
  key: "request_sent",
  label: "요청 전송중",
  detail: "백엔드 응답 스트림을 시작하고 있습니다.",
  status: "running",
};

const LIVE_POD_EXAMPLE = "el-kugnus-bot-listener-c9ffbb846-fpcts";

function normalizeToolAnswer(answer: string, mode?: string) {
  const text = String(answer || "").trim();
  if (!text) return text;

  if (mode === "tool:resource_list" || mode === "tool:resource_candidate_list") {
    const withoutExample = text.replace(/\s*(?:예시|example):\s*.+$/iu, "").trim();
    if (withoutExample.includes("전체 목록")) return withoutExample;
    return `${withoutExample} 전체 목록은 아래 artifact에서 확인할 수 있습니다.`;
  }

  return text;
}

function promoteStage(stages: CopilotChatStage[], nextStage: CopilotChatStage[]): CopilotChatStage[] {
  const incoming = Array.isArray(nextStage) ? nextStage[0] : nextStage;
  const completed = stages.map((stage) =>
    stage.status === "running"
      ? { ...stage, status: stage.key === incoming.key ? incoming.status : "completed" }
      : stage,
  );
  const existingIndex = completed.findIndex((stage) => stage.key === incoming.key);
  if (existingIndex >= 0) {
    const next = [...completed];
    next[existingIndex] = incoming;
    return next;
  }
  return [...completed, incoming];
}

function finalizeStages(stages: CopilotChatStage[], status: "completed" | "error"): CopilotChatStage[] {
  return stages.map((stage) => (stage.status === "running" ? { ...stage, status } : stage));
}

function getCurrentStage(stages: CopilotChatStage[]): CopilotChatStage | null {
  return [...stages].reverse().find((stage) => stage.status === "running") ?? stages[stages.length - 1] ?? null;
}

function buildPendingTurn(pendingAssistant: PendingAssistantState | null): DisplayTurn[] {
  if (!pendingAssistant) return [];
  return [
    {
      role: "assistant",
      text: pendingAssistant.text,
      meta: pendingAssistant.meta,
      sources: pendingAssistant.sources,
      artifacts: pendingAssistant.artifacts,
      citationMap: pendingAssistant.citationMap,
      stages: pendingAssistant.stages,
      isPending: true,
    },
  ];
}

function stripTrailingReferenceList(text: string): string {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let end = lines.length;
  while (end > 0 && !lines[end - 1]?.trim()) end -= 1;
  let cursor = end;
  while (cursor > 0 && /^\[\d+\]\s+.+$/.test(lines[cursor - 1]?.trim() ?? "")) cursor -= 1;
  return cursor < end ? lines.slice(0, cursor).join("\n").trim() : text.trim();
}

async function copyToClipboard(text: string) {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  throw new Error("Clipboard API is unavailable in this browser.");
}

function artifactItemsToSummaries(items: CopilotChatArtifactItem[]): OcpLiveResourceSummary[] {
  return items.map((item) => ({
    name: item.name,
    namespace: item.namespace,
    kind: item.kind,
    createdAt: "",
    phase: item.phase,
    nodeName: item.nodeName,
    readyReplicas: 0,
    replicas: 0,
    type: item.type,
    clusterIp: item.clusterIp,
    host: item.host,
    to: item.to,
  }));
}

function buildEditorTargetFromArtifact(
  artifact: CopilotChatArtifact,
  item?: CopilotChatArtifactItem,
): EditorTarget | null {
  const target = item?.actionTarget ?? artifact.payload ?? {};
  const resource = String(target.resource ?? artifact.resource ?? "").trim() as LiveResourceKind;
  const namespace = String(target.namespace ?? artifact.namespace ?? item?.namespace ?? "").trim();
  const name = String(target.name ?? artifact.resourceName ?? item?.name ?? "").trim();
  if (!resource || !namespace || !name) return null;

  const payload = artifact.payload ?? {};
  const detail =
    artifact.artifactType === "resource_editor"
      ? ({
          connectionId: "",
          clusterUrl: "",
          resource,
          namespace,
          name,
          kind: String(payload.kind ?? item?.kind ?? ""),
          manifestYaml: String(payload.manifest_yaml ?? ""),
          manifestJson: (payload.manifest_json as Record<string, unknown> | undefined) ?? {},
        } satisfies OcpLiveResourceDetailResponse)
      : null;

  return {
    resource,
    namespace,
    name,
    summary: item ? artifactItemsToSummaries([item])[0] : null,
    detail,
  };
}

function createLiveSourceFromDetail(detail: OcpLiveResourceDetailResponse): CopilotChatSourceItem {
  return {
    sourceType: "live",
    label: detail.name,
    sourcePath: "",
    relativeSourcePath: "",
    pageNumber: null,
    chunkId: "",
    namespace: detail.namespace,
    kind: detail.kind,
    score: 0,
    provenance: ["live", "chat_apply"],
    metadata: {
      manifest_yaml: detail.manifestYaml,
      manifest_json: detail.manifestJson,
      resource: detail.resource,
      namespace: detail.namespace,
      kind: detail.kind,
      name: detail.name,
    },
  };
}

function createPostApplyArtifacts(detail: OcpLiveResourceDetailResponse): CopilotChatArtifact[] {
  return [
    {
      artifactType: "resource_editor",
      title: `${detail.name} YAML`,
      description: "Updated live YAML",
      resource: detail.resource,
      namespace: detail.namespace,
      resourceName: detail.name,
      payload: {
        kind: detail.kind,
        manifest_yaml: detail.manifestYaml,
        manifest_json: detail.manifestJson,
      },
      items: [
        {
          name: detail.name,
          kind: detail.kind,
          namespace: detail.namespace,
          resource: detail.resource,
          phase: "",
          type: "",
          host: "",
          to: "",
          nodeName: "",
          clusterIp: "",
          actionTarget: {
            resource: detail.resource,
            namespace: detail.namespace,
            name: detail.name,
            kind: detail.kind,
          },
          metadata: {},
        },
      ],
    },
    {
      artifactType: "followup_suggestions",
      title: "Next checks",
      description: "운영 후속 질문을 바로 이어갈 수 있습니다.",
      resource: detail.resource,
      namespace: detail.namespace,
      resourceName: detail.name,
      payload: {
        prompts: [
          `${detail.name} 변경 영향이 뭐가 있는지 다시 설명해줘`,
          `${detail.name} 와 연결된 service가 무엇인지 보여줘`,
          `${detail.name} 관련 공식 문서 기준 추가 개선점을 찾아줘`,
        ],
      },
      items: [],
    },
  ];
}

function ChatStageDetails({ stages, open }: { stages: CopilotChatStage[]; open: boolean }) {
  const [expanded, setExpanded] = useState(open);
  const currentStage = getCurrentStage(stages);

  useEffect(() => {
    if (open) {
      setExpanded(true);
    }
  }, [open]);

  if (!currentStage) return null;

  return (
    <div className="rounded-xl border border-border/70 bg-background/50">
      <button
        type="button"
        className="flex w-full items-start justify-between gap-3 px-4 py-3 text-left"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <div className="flex items-start gap-3">
          <span className="mt-0.5">
            {currentStage.status === "completed" ? (
              <CircleCheckBig className="h-4 w-4 text-emerald-400" />
            ) : currentStage.status === "error" ? (
              <CircleX className="h-4 w-4 text-destructive" />
            ) : (
              <CircleDashed className="h-4 w-4 animate-spin text-primary" />
            )}
          </span>
          <div>
            <div className="font-medium text-foreground">{currentStage.label}</div>
            <div className="text-xs text-muted-foreground">{currentStage.detail}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em]",
              currentStage.status === "completed" && "bg-emerald-500/15 text-emerald-300",
              currentStage.status === "error" && "bg-destructive/15 text-destructive",
              currentStage.status === "running" && "bg-primary/15 text-primary",
            )}
          >
            {currentStage.status}
          </span>
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform duration-200",
              expanded && "rotate-180",
            )}
          />
        </div>
      </button>

      <div
        className={cn(
          "grid transition-[grid-template-rows] duration-200 ease-out",
          expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className="overflow-hidden">
          <div className="space-y-2 px-4 pb-4">
            {stages.map((stage) => (
              <div key={stage.key} className="rounded-lg border border-border/60 bg-card/80 px-3 py-2 text-xs">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-medium text-foreground">{stage.label}</div>
                  <span
                    className={cn(
                      "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em]",
                      stage.status === "completed" && "bg-emerald-500/15 text-emerald-300",
                      stage.status === "error" && "bg-destructive/15 text-destructive",
                      stage.status === "running" && "bg-primary/15 text-primary",
                    )}
                  >
                    {stage.status}
                  </span>
                </div>
                <div className="mt-1 text-muted-foreground">{stage.detail}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ResourceArtifactBlock({
  artifact,
  onOpenEditor,
  onCopy,
  onSuggest,
}: {
  artifact: CopilotChatArtifact;
  onOpenEditor: (target: EditorTarget) => void;
  onCopy: (text: string, errorMessage?: string) => Promise<void>;
  onSuggest: (prompt: string) => void;
}) {
  if (artifact.artifactType === "resource_list") {
    return (
      <div className="space-y-3 rounded-xl border border-border/70 bg-background/50 p-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <strong className="text-foreground">{artifact.title || "Resource List"}</strong>
          <span className="text-muted-foreground">{artifact.namespace}</span>
        </div>
        <div className="max-h-[28rem] overflow-auto rounded-xl">
          <ResourceList
            items={artifactItemsToSummaries(artifact.items)}
            resource={artifact.resource}
            namespace={artifact.namespace}
            onSelect={(item) => {
              const matched = artifact.items.find(
                (candidate) => candidate.name === item.name && candidate.namespace === item.namespace,
              );
              const target = buildEditorTargetFromArtifact(artifact, matched);
              if (target) onOpenEditor(target);
            }}
            emptyMessage="표시할 live resource가 없습니다."
          />
        </div>
      </div>
    );
  }

  if (artifact.artifactType === "resource_relations") {
    return (
      <div className="space-y-2 rounded-xl border border-border/70 bg-background/50 p-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <strong className="text-foreground">{artifact.title || "Resource Relations"}</strong>
          <span className="text-muted-foreground">{artifact.namespace}</span>
        </div>
        {artifact.items.map((item) => {
          const services = Array.isArray(item.metadata.services) ? item.metadata.services : [];
          return (
            <button
              key={`${item.namespace}:${item.name}`}
              type="button"
              className="block w-full rounded-xl border border-border/70 bg-card/80 px-3 py-3 text-left text-sm hover:bg-secondary/40"
              onClick={() => {
                const target = buildEditorTargetFromArtifact(artifact, item);
                if (target) onOpenEditor(target);
              }}
            >
              <div className="font-medium text-foreground">{item.name}</div>
              <div className="text-muted-foreground">
                {item.kind} · {item.namespace}
              </div>
              <div className="text-muted-foreground">
                {services.length ? services.join(", ") : "연결된 service 없음"}
              </div>
            </button>
          );
        })}
      </div>
    );
  }

  if (artifact.artifactType === "resource_editor") {
    const target = buildEditorTargetFromArtifact(artifact, artifact.items[0]);
    const manifestYaml = String(artifact.payload.manifest_yaml ?? "");

    return (
      <div className="space-y-3 rounded-xl border border-border/70 bg-background/50 p-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <strong className="text-foreground">{artifact.title || "Resource YAML"}</strong>
          <span className="text-muted-foreground">{artifact.namespace}</span>
        </div>
        <div className="flex gap-2">
          {target ? (
            <Button type="button" size="sm" variant="outline" onClick={() => onOpenEditor(target)}>
              YAML 열기
            </Button>
          ) : null}
          {manifestYaml ? (
            <Button
              type="button"
              size="sm"
              variant="secondary"
              onClick={() => void onCopy(manifestYaml, "Failed to copy live YAML.")}
            >
              YAML 복사
            </Button>
          ) : null}
        </div>
      </div>
    );
  }

  if (artifact.artifactType === "command_template") {
    const command = String(artifact.payload.resolved_command ?? artifact.payload.template ?? "");
    return (
      <div className="space-y-3 rounded-xl border border-border/70 bg-background/50 p-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <strong className="text-foreground">{artifact.title || "Command Template"}</strong>
          <span className="text-muted-foreground">exact evidence</span>
        </div>
        <pre className="overflow-auto rounded-xl border border-border/70 bg-background/80 p-4 text-xs leading-6 text-muted-foreground">
          <code>{command}</code>
        </pre>
        {command ? (
          <Button type="button" size="sm" variant="outline" onClick={() => void onCopy(command, "Failed to copy command.")}>
            명령어 복사
          </Button>
        ) : null}
      </div>
    );
  }

  if (artifact.artifactType === "followup_suggestions") {
    const prompts = Array.isArray(artifact.payload.prompts) ? artifact.payload.prompts : [];
    return (
      <div className="space-y-3 rounded-xl border border-border/70 bg-background/50 p-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <strong className="text-foreground">{artifact.title || "Next checks"}</strong>
          <span className="text-muted-foreground">{artifact.namespace}</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {prompts.map((prompt) => (
            <button
              key={String(prompt)}
              type="button"
              className="rounded-full border border-border/70 bg-card/80 px-3 py-1.5 text-xs text-muted-foreground hover:bg-secondary hover:text-foreground"
              onClick={() => onSuggest(String(prompt))}
            >
              {String(prompt)}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return null;
}

export function ChatPage({ controller, session, updateSession }: ChatPageProps) {
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const [selection, setSelection] = useState<SourceSelection | null>(null);
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [pendingAssistant, setPendingAssistant] = useState<PendingAssistantState | null>(null);
  const [editorTarget, setEditorTarget] = useState<EditorTarget | null>(null);
  const [editorLoading, setEditorLoading] = useState<EditorLoadingState>({
    active: false,
    title: "",
  });
  const threadRef = useRef<HTMLDivElement | null>(null);
  const scrollSaveTimerRef = useRef<number | null>(null);
  const shouldStickToBottomRef = useRef(false);

  const canSend = useMemo(() => Boolean(draft.trim() && !isSending), [draft, isSending]);
  const sourceDrawerOpen = Boolean(selection);
  const displayTurns = useMemo<DisplayTurn[]>(
    () => [...session.turns, ...buildPendingTurn(pendingAssistant)],
    [pendingAssistant, session.turns],
  );

  useEffect(() => {
    setDraft("");
    setError("");
    setSelection(null);
    setSelectionLoading(false);
    setPendingAssistant(null);
    setEditorTarget(null);
    setEditorLoading({ active: false, title: "" });
    requestAnimationFrame(() => {
      if (threadRef.current) {
        threadRef.current.scrollTop = session.viewport?.scrollTop ?? 0;
      }
    });
  }, [session.id, session.viewport?.scrollTop]);

  useEffect(() => () => {
    if (scrollSaveTimerRef.current !== null) window.clearTimeout(scrollSaveTimerRef.current);
  }, []);

  useEffect(() => {
    if (!shouldStickToBottomRef.current || !threadRef.current) return;
    threadRef.current.scrollTop = threadRef.current.scrollHeight;
    shouldStickToBottomRef.current = false;
  }, [displayTurns.length]);

  function persistViewportState(scrollTop: number) {
    updateSession(session.id, (current) => ({
      ...current,
      viewport: {
        ...current.viewport,
        scrollTop,
        lastViewedTurnIndex: Math.max(0, current.turns.length - 1),
        selectedSourceKey: selection?.source.chunkId || selection?.source.label || "",
      },
    }));
  }

  function handleThreadScroll() {
    if (!threadRef.current) return;
    const scrollTop = threadRef.current.scrollTop;
    if (scrollSaveTimerRef.current !== null) window.clearTimeout(scrollSaveTimerRef.current);
    scrollSaveTimerRef.current = window.setTimeout(() => persistViewportState(scrollTop), 120);
  }

  async function openEditorTarget(target: EditorTarget) {
    if (target.detail || !controller.profile) {
      setEditorTarget(target);
      return;
    }

    setEditorLoading({
      active: true,
      title: "YAML을 불러오는 중",
      detail: `${target.namespace} / ${target.name} 리소스의 live manifest를 가져오는 중입니다.`,
    });

    try {
      const detail = await getOcpResourceDetail(
        controller.profile.connectionId,
        target.resource,
        target.namespace,
        target.name,
      );
      setEditorTarget({
        ...target,
        detail,
      });
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to load live YAML.");
    } finally {
      setEditorLoading({ active: false, title: "" });
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!draft.trim()) return;

    const sessionId = session.id;
    const nextMessage = draft.trim();
    let stageHistory: CopilotChatStage[] = [REQUEST_SENT_STAGE];
    let finalTurn: ChatTurn | null = null;

    setDraft("");
    setError("");
    shouldStickToBottomRef.current = true;
    setPendingAssistant({
      text: "",
      sources: [],
      artifacts: [],
      citationMap: [],
      stages: stageHistory,
    });

    updateSession(sessionId, (current) => ({
      ...current,
      title: current.turns.length === 0 ? deriveChatSessionTitle(nextMessage) : current.title,
      updatedAt: new Date().toISOString(),
      turns: [...current.turns, { role: "user", text: nextMessage }],
    }));

    setIsSending(true);

    try {
      await streamCopilotChat(
        {
          message: nextMessage,
          connectionId: controller.profile?.connectionId,
          namespace: controller.profile?.defaultNamespace,
          history: session.turns.slice(-6).map((turn) => ({
            role: turn.role,
            text: turn.text,
            lane: turn.lane,
            sourcePaths: (turn.sources ?? []).map((source) => source.sourcePath).filter(Boolean),
            resourceNames: (turn.sources ?? [])
              .filter((source) => source.sourceType === "live")
              .map((source) => source.label)
              .filter(Boolean),
            namespace: (turn.sources ?? []).find((source) => source.namespace)?.namespace ?? "",
          })),
        },
        {
          onStage: (stage) => {
            stageHistory = promoteStage(stageHistory, [stage]);
            setPendingAssistant((current) => ({
              text: current?.text ?? "",
              meta: current?.meta,
              sources: current?.sources ?? [],
              artifacts: current?.artifacts ?? [],
              citationMap: current?.citationMap ?? [],
              stages: stageHistory,
            }));
          },
          onAnswerDelta: (delta) => {
            setPendingAssistant((current) => ({
              text: `${current?.text ?? ""}${delta}`,
              meta: current?.meta,
              sources: current?.sources ?? [],
              artifacts: current?.artifacts ?? [],
              citationMap: current?.citationMap ?? [],
              stages: current?.stages ?? stageHistory,
            }));
          },
          onResult: (response) => {
            stageHistory = finalizeStages(stageHistory, "completed");
            const normalizedAnswer = normalizeToolAnswer(response.answer, response.mode);
            finalTurn = {
              role: "assistant",
              text: normalizedAnswer,
              meta: `${response.lane} · ${response.mode}`,
              sources: response.sources,
              artifacts: response.artifacts,
              citationMap: response.citationMap,
              stages: stageHistory,
              lane: response.lane,
              fallbackUsed: response.fallbackUsed,
              previewReady: response.previewReady,
            };
            setPendingAssistant({
              text: normalizedAnswer,
              meta: `${response.lane} · ${response.mode}`,
              sources: response.sources,
              artifacts: response.artifacts,
              citationMap: response.citationMap,
              stages: stageHistory,
            });
          },
        },
      );

      if (!finalTurn) {
        throw new Error("Chat response ended before a final result was received.");
      }

      updateSession(sessionId, (current) => ({
        ...current,
        updatedAt: new Date().toISOString(),
        turns: [...current.turns, finalTurn!],
        viewport: { ...current.viewport, lastViewedTurnIndex: current.turns.length },
      }));
      setPendingAssistant(null);
    } catch (nextError) {
      stageHistory = finalizeStages(stageHistory, "error");
      setPendingAssistant({
        text: "",
        meta: "request failed",
        sources: [],
        artifacts: [],
        citationMap: [],
        stages: stageHistory,
      });
      setError(nextError instanceof Error ? nextError.message : "Failed to load chat response.");
    } finally {
      setIsSending(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    if (!canSend) return;
    event.currentTarget.form?.requestSubmit();
  }

  async function handleSelectSource(source: CopilotChatSourceItem, citation?: CopilotCitationMapItem) {
    setError("");
    setSelection({ source, citation });

    if (source.sourceType === "live") {
      setSelectionLoading(false);
      return;
    }

    setSelectionLoading(true);
    try {
      const preview = await fetchDocumentSnippet(source.sourcePath, source.chunkId);
      setSelection({ source, preview, citation });
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to load source preview.");
    } finally {
      setSelectionLoading(false);
    }
  }

  async function handleCopyText(text: string, errorMessage = "Failed to copy text.") {
    try {
      await copyToClipboard(text);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : errorMessage);
    }
  }

  function handleSuggestion(prompt: string) {
    setDraft(prompt);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Chat"
        title="Copilot workspace"
        description="문서 답변과 live OCP 결과를 한 화면에서 다루고, 리소스 목록과 YAML 편집, 근거 확인까지 이어지는 대화 작업 공간입니다."
      />

      {!controller.profile ? (
        <Card className="border-border/70 bg-background/50">
          <CardContent className="p-6 text-sm text-muted-foreground">
            현재는 <strong>doc-only mode</strong> 입니다. live OCP 질의에는 클러스터 연결이 필요합니다.
          </CardContent>
        </Card>
      ) : null}

      <div className="flex flex-col gap-4 xl:flex-row xl:items-start">
        <Card
          className={cn(
            "min-w-0 overflow-hidden border-border/70 bg-card/95 transition-[width,flex-basis] duration-300 xl:flex-1",
            sourceDrawerOpen ? "xl:basis-[calc(100%-24rem)]" : "xl:basis-full",
          )}
        >
          <CardHeader>
            <CardTitle>Conversation</CardTitle>
            <CardDescription>Streaming answer, citation preview, and live resource artifacts.</CardDescription>
          </CardHeader>
          <CardContent className="flex min-h-[70vh] flex-col gap-4 xl:h-[calc(100vh-11rem)] xl:min-h-0">
            <ChatTranscript
              turns={displayTurns}
              threadRef={threadRef}
              onScroll={handleThreadScroll}
              renderStages={(stages, open) => <ChatStageDetails stages={stages} open={open} />}
              renderArtifact={(artifact, key) => (
                <ResourceArtifactBlock
                  key={key}
                  artifact={artifact}
                  onOpenEditor={(target) => {
                    void openEditorTarget(target);
                  }}
                  onCopy={handleCopyText}
                  onSuggest={handleSuggestion}
                />
              )}
              renderCitationClick={(source, citation) => {
                void handleSelectSource(source, citation);
              }}
              onCopyCode={(code) => handleCopyText(code, "Failed to copy code block.")}
              stripReferenceList={stripTrailingReferenceList}
            />

            {error ? (
              <div className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            ) : null}

            <ChatComposer
              draft={draft}
              canSend={canSend}
              isSending={isSending}
              shortcuts={[
                "OpenShift route 설명",
                "deployment 목록 보여줘",
                `pod ${LIVE_POD_EXAMPLE} yaml 보여줘`,
              ]}
              onDraftChange={setDraft}
              onShortcut={setDraft}
              onSubmit={handleSubmit}
              onKeyDown={handleComposerKeyDown}
            />
          </CardContent>
        </Card>

        <div
          className={cn(
            "transition-[max-width,opacity,transform] duration-300 xl:sticky xl:top-6",
            sourceDrawerOpen
              ? "max-w-full opacity-100 xl:w-96 xl:translate-x-0"
              : "pointer-events-none max-h-0 max-w-0 opacity-0 xl:w-0 xl:translate-x-4",
          )}
        >
          {sourceDrawerOpen ? (
            <Card className="border-border/70 bg-card/95 xl:max-h-[calc(100vh-3rem)]">
              <CardHeader>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <CardTitle>Source</CardTitle>
                    <CardDescription>{selection?.source.label}</CardDescription>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setSelection(null);
                      setSelectionLoading(false);
                    }}
                  >
                    닫기
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 overflow-y-auto xl:max-h-[calc(100vh-9rem)]">
                {selectionLoading ? (
                  <div className="rounded-xl border border-border/70 bg-background/50 px-4 py-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-3">
                      <LoaderCircle className="h-4 w-4 animate-spin text-primary" />
                      <span>Source preview를 불러오는 중입니다.</span>
                    </div>
                  </div>
                ) : null}

                {selection?.citation?.supportingText ? (
                  <div className="rounded-xl border border-primary/30 bg-primary/10 px-4 py-3 text-sm text-foreground">
                    <div><strong>Exact Evidence</strong></div>
                    <div>{selection.citation.supportingText}</div>
                    {selection.citation.commandText ? (
                      <div>
                        <strong>Command</strong>: <code>{selection.citation.commandText}</code>
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {selection?.source.sourceType === "live" && !selectionLoading ? (
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <div><span className="font-medium text-foreground">Name:</span> {selection.source.label}</div>
                    <div><span className="font-medium text-foreground">Kind:</span> {selection.source.kind || "-"}</div>
                    <div><span className="font-medium text-foreground">Namespace:</span> {selection.source.namespace || "-"}</div>
                    {typeof selection.source.metadata.manifest_yaml === "string" && selection.source.metadata.manifest_yaml.trim() ? (
                      <>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            void handleCopyText(
                              String(selection.source.metadata.manifest_yaml),
                              "Failed to copy live YAML.",
                            )
                          }
                        >
                          YAML 복사
                        </Button>
                        <pre className="overflow-auto rounded-xl border border-border/70 bg-background/80 p-4 text-xs leading-6 text-muted-foreground">
                          <code>{String(selection.source.metadata.manifest_yaml)}</code>
                        </pre>
                      </>
                    ) : null}
                  </div>
                ) : null}

                {selection?.preview && !selectionLoading ? (
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <div><span className="font-medium text-foreground">File:</span> {selection.preview.fileName || "-"}</div>
                    <div><span className="font-medium text-foreground">Section:</span> {selection.preview.sectionTitle || "-"}</div>
                    <div><span className="font-medium text-foreground">Path:</span> {selection.preview.relativeSourcePath || selection.preview.sourcePath}</div>
                    <div><span className="font-medium text-foreground">Lines:</span> {selection.preview.lineStart ?? "-"} ~ {selection.preview.lineEnd ?? "-"}</div>
                    <div className="rounded-xl border border-border/70 bg-background/50 p-4">
                      <MarkdownArticle
                        content={selection.preview.snippet || selection.preview.lines.join("\n")}
                        renderCodeActions={(code) => (
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            onClick={() => void handleCopyText(code, "Failed to copy code block.")}
                          >
                            코드 복사
                          </Button>
                        )}
                      />
                    </div>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ) : null}
        </div>
      </div>

      {editorLoading.active ? (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/65 p-4 backdrop-blur-sm">
          <Card className="w-full max-w-md border-border/70 bg-card/95 shadow-2xl">
            <CardContent className="flex items-start gap-4 p-6">
              <div className="rounded-full bg-primary/10 p-3 text-primary">
                <LoaderCircle className="h-5 w-5 animate-spin" />
              </div>
              <div className="space-y-1">
                <div className="font-semibold text-foreground">{editorLoading.title}</div>
                {editorLoading.detail ? (
                  <div className="text-sm leading-6 text-muted-foreground">{editorLoading.detail}</div>
                ) : null}
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}

      {controller.profile && editorTarget ? (
        <ResourceYamlEditorModal
          controller={controller}
          resource={editorTarget.resource}
          namespace={editorTarget.namespace}
          name={editorTarget.name}
          open={Boolean(editorTarget)}
          onClose={() => setEditorTarget(null)}
          initialDetail={editorTarget.detail}
          initialSummary={editorTarget.summary}
          onSaved={(detail) => {
            const liveSource = createLiveSourceFromDetail(detail);
            updateSession(session.id, (current) => ({
              ...current,
              updatedAt: new Date().toISOString(),
              turns: [
                ...current.turns,
                {
                  role: "assistant",
                  text: `${detail.namespace} namespace의 ${detail.kind} ${detail.name} 변경이 반영되었습니다. 운영 관점에서 추가 확인이 필요한지 바로 이어서 점검할 수 있습니다.`,
                  meta: "live · yaml_apply_completed",
                  lane: "live",
                  sources: [liveSource],
                  artifacts: createPostApplyArtifacts(detail),
                },
              ],
            }));
            shouldStickToBottomRef.current = true;
            setEditorTarget(null);
          }}
        />
      ) : null}
    </div>
  );
}

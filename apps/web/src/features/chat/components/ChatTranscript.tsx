import type { ReactNode, RefObject } from "react";

import { Button } from "@/components/ui/button";
import { MarkdownArticle } from "@/components/ui/MarkdownArticle";
import type {
  ChatTurn,
  CopilotCitationMapItem,
  CopilotChatArtifact,
  CopilotChatSourceItem,
  CopilotChatStage,
} from "../types";

type DisplayTurn = ChatTurn & { isPending?: boolean };

type ChatTranscriptProps = {
  turns: DisplayTurn[];
  threadRef: RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  renderStages: (stages: CopilotChatStage[], open: boolean) => ReactNode;
  renderArtifact: (artifact: CopilotChatArtifact, key: string) => ReactNode;
  renderCitationClick: (source: CopilotChatSourceItem, citation?: CopilotCitationMapItem) => void;
  onCopyCode: (code: string) => Promise<void>;
  stripReferenceList: (text: string) => string;
};

export function ChatTranscript({
  turns,
  threadRef,
  onScroll,
  renderStages,
  renderArtifact,
  renderCitationClick,
  onCopyCode,
  stripReferenceList,
}: ChatTranscriptProps) {
  return (
    <div ref={threadRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-auto pr-1">
      <div className="flex min-h-full flex-col gap-4">
        {turns.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-border/70 bg-background/50 px-4 py-8 text-sm leading-6 text-muted-foreground">
            첫 질문을 보내면 여기에서 assistant 응답과 live artifact가 함께 표시됩니다.
          </div>
        ) : null}

        {turns.map((turn, index) => {
          const chatContent = turn.role === "assistant" ? stripReferenceList(turn.text) : turn.text;
          const isUser = turn.role === "user";

          return (
            <div key={`${turn.role}-${index}`} className={`flex gap-3 ${isUser ? "justify-end" : ""}`}>
              {!isUser ? (
                <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-primary/15 text-xs font-semibold text-primary">
                  AI
                </div>
              ) : null}

              <div
                className={`max-w-[85%] space-y-3 rounded-2xl border px-4 py-3 text-sm leading-7 ${
                  isUser
                    ? "border-primary/30 bg-primary text-primary-foreground"
                    : "border-border/70 bg-card text-foreground"
                }`}
              >
                {!isUser ? (
                  <>
                    {turn.stages?.length ? renderStages(turn.stages, Boolean(turn.isPending)) : null}

                    {chatContent ? (
                      <div className="space-y-3">
                        <MarkdownArticle
                          content={chatContent}
                          renderCodeActions={(code) => (
                            <Button type="button" size="sm" variant="outline" onClick={() => void onCopyCode(code)}>
                              코드 복사
                            </Button>
                          )}
                          renderCitation={(sourceIndex) => {
                            const citation = (turn.citationMap ?? []).find(
                              (item) => item.citationNumber === sourceIndex + 1,
                            );
                            const mappedSource = citation
                              ? turn.sources?.[citation.sourceIndex]
                              : turn.sources?.[sourceIndex];
                            return (
                              <button
                                key={`citation-${sourceIndex}`}
                                type="button"
                                className="ml-1 inline-flex items-center justify-center rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary"
                                onClick={() => {
                                  if (mappedSource) renderCitationClick(mappedSource, citation);
                                }}
                                aria-label={`source ${sourceIndex + 1} 열기`}
                              >
                                [{sourceIndex + 1}]
                              </button>
                            );
                          }}
                        />
                      </div>
                    ) : turn.isPending ? (
                      <div className="text-muted-foreground">응답과 근거를 정리하고 있습니다.</div>
                    ) : null}

                    {(turn.artifacts ?? []).map((artifact, artifactIndex) =>
                      renderArtifact(artifact, `${artifact.artifactType}-${artifactIndex}`),
                    )}
                  </>
                ) : (
                  <div className="whitespace-pre-wrap">{turn.text}</div>
                )}

                {turn.meta ? (
                  <div className={`text-xs ${isUser ? "text-primary-foreground/80" : "text-muted-foreground"}`}>
                    {turn.meta}
                  </div>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

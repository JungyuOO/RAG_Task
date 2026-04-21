import { Button } from "@/shared/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/shared/ui/card";
import type { ChatSessionRecord } from "./types";

type ChatSessionRailProps = {
  sessions: ChatSessionRecord[];
  activeSessionId: string;
  onSelect: (sessionId: string) => void;
  onCreate: () => void;
  onRemove: (sessionId: string) => void;
};

function formatSessionTime(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export function ChatSessionRail({
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  onRemove,
}: ChatSessionRailProps) {
  return (
    <Card className="border-border/70 bg-card/90">
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Sessions</CardTitle>
            <CardDescription>최근 대화 세션과 상태를 관리합니다.</CardDescription>
          </div>
          <Button type="button" variant="outline" size="sm" onClick={onCreate}>
            New session
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {sessions.map((session) => {
          const active = session.id === activeSessionId;
          const assistantTurns = session.turns.filter((turn) => turn.role === "assistant").length;

          return (
            <button
              key={session.id}
              type="button"
              onClick={() => onSelect(session.id)}
              className={`block w-full rounded-2xl border px-4 py-3 text-left transition-colors ${
                active
                  ? "border-primary/50 bg-primary/10"
                  : "border-border/70 bg-background/50 hover:bg-secondary/40"
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate font-medium text-foreground">{session.title}</div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    turns={session.turns.length} · assistant={assistantTurns}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    updated {formatSessionTime(session.updatedAt)}
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={(event) => {
                    event.stopPropagation();
                    onRemove(session.id);
                  }}
                >
                  Remove
                </Button>
              </div>
            </button>
          );
        })}
      </CardContent>
    </Card>
  );
}



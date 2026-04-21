import { FormEvent, KeyboardEvent } from "react";

import { Button } from "@/shared/ui/button";

type ChatComposerProps = {
  draft: string;
  canSend: boolean;
  isSending: boolean;
  shortcuts: string[];
  onDraftChange: (value: string) => void;
  onShortcut: (prompt: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
};

export function ChatComposer({
  draft,
  canSend,
  isSending,
  shortcuts,
  onDraftChange,
  onShortcut,
  onSubmit,
  onKeyDown,
}: ChatComposerProps) {
  return (
    <form onSubmit={onSubmit} className="surface-soft space-y-3 rounded-2xl p-4">
      <div className="flex flex-wrap gap-2">
        {shortcuts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="surface-muted rounded-full px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            onClick={() => onShortcut(prompt)}
          >
            {prompt}
          </button>
        ))}
      </div>
      <div className="flex items-end gap-2">
        <textarea
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          onKeyDown={onKeyDown}
          rows={3}
          placeholder="문서 또는 클러스터 상태를 질문해보세요."
          className="min-h-[72px] flex-1 resize-none rounded-xl border border-border bg-input px-3 py-2 text-sm outline-none placeholder:text-muted-foreground"
        />
        <Button type="submit" disabled={!canSend}>
          {isSending ? "Sending..." : "Send"}
        </Button>
      </div>
    </form>
  );
}


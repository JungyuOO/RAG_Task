import { FormEvent, useEffect, useState } from "react";

import { PageHeader } from "@/shared/layout/PageHeader";
import { Button } from "@/shared/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import {
  getDefaultWorkspaceModelProfile,
  updateDefaultWorkspaceModelProfile,
} from "@/domains/models/modelsApi";
import type { WorkspaceModelProfileUpdateRequest } from "@/domains/models/types";
import type { WorkspaceRecord } from "@/domains/workspaces/types";

type ModelsPageProps = {
  selectedWorkspace: WorkspaceRecord | null;
  onLoadingChange?: (state: { active: boolean; title: string; detail?: string }) => void;
};

const EMPTY_FORM: WorkspaceModelProfileUpdateRequest = {
  chatProvider: "openai-compatible",
  chatBaseUrl: "",
  chatModel: "",
  chatApiKeyMode: "managed",
  embeddingProvider: "tei",
  embeddingBaseUrl: "",
  embeddingModel: "",
  embeddingApiKeyMode: "managed",
};

export function ModelsPage({ selectedWorkspace, onLoadingChange }: ModelsPageProps) {
  const [form, setForm] = useState<WorkspaceModelProfileUpdateRequest>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    async function run() {
      if (!selectedWorkspace) {
        setForm(EMPTY_FORM);
        return;
      }
      setLoading(true);
      setError("");
      setMessage("");
      try {
        const profile = await getDefaultWorkspaceModelProfile(selectedWorkspace.workspaceId);
        setForm({
          chatProvider: profile.chatProvider,
          chatBaseUrl: profile.chatBaseUrl,
          chatModel: profile.chatModel,
          chatApiKeyMode: profile.chatApiKeyMode,
          embeddingProvider: profile.embeddingProvider,
          embeddingBaseUrl: profile.embeddingBaseUrl,
          embeddingModel: profile.embeddingModel,
          embeddingApiKeyMode: profile.embeddingApiKeyMode,
        });
      } catch (nextError) {
        setError(nextError instanceof Error ? nextError.message : "Failed to load model settings.");
      } finally {
        setLoading(false);
      }
    }
    void run();
  }, [selectedWorkspace?.workspaceId]);

  useEffect(() => {
    onLoadingChange?.({
      active: loading,
      title: "모델 설정 불러오는 중",
      detail: "선택된 workspace의 LLM/임베딩 설정을 가져오는 중입니다.",
    });
    return () => onLoadingChange?.({ active: false, title: "" });
  }, [loading, onLoadingChange]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedWorkspace) return;
    setLoading(true);
    setError("");
    setMessage("");
    try {
      await updateDefaultWorkspaceModelProfile(selectedWorkspace.workspaceId, form);
      setMessage("기본 모델 설정이 저장되었습니다.");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to save model settings.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Models"
        title="Model settings"
        description="workspace별 chat/embedding 엔드포인트와 기본 모델을 설정합니다."
      />

      {!selectedWorkspace ? (
        <Card className="surface-muted">
          <CardContent className="p-6 text-sm text-muted-foreground">
            먼저 Workspaces 화면에서 활성 workspace를 선택하세요.
          </CardContent>
        </Card>
      ) : null}

      {error ? (
        <Card className="surface-danger">
          <CardContent className="p-6 text-sm text-destructive">{error}</CardContent>
        </Card>
      ) : null}

      {message ? (
        <Card className="surface-brand">
          <CardContent className="p-6 text-sm text-foreground">{message}</CardContent>
        </Card>
      ) : null}

      {selectedWorkspace ? (
        <Card className="surface-soft">
          <CardHeader>
            <CardTitle>{selectedWorkspace.name}</CardTitle>
            <CardDescription>workspace slug={selectedWorkspace.slug}</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="grid gap-4 md:grid-cols-2" onSubmit={handleSubmit}>
              <div className="space-y-2">
                <Label htmlFor="chat-provider">Chat Provider</Label>
                <Input id="chat-provider" value={form.chatProvider} onChange={(event) => setForm((current) => ({ ...current, chatProvider: event.target.value }))} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="chat-model">Chat Model</Label>
                <Input id="chat-model" value={form.chatModel} onChange={(event) => setForm((current) => ({ ...current, chatModel: event.target.value }))} />
              </div>
              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="chat-base-url">Chat Base URL</Label>
                <Input id="chat-base-url" value={form.chatBaseUrl} onChange={(event) => setForm((current) => ({ ...current, chatBaseUrl: event.target.value }))} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="chat-key-mode">Chat API Key Mode</Label>
                <Input id="chat-key-mode" value={form.chatApiKeyMode} onChange={(event) => setForm((current) => ({ ...current, chatApiKeyMode: event.target.value }))} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="embedding-provider">Embedding Provider</Label>
                <Input id="embedding-provider" value={form.embeddingProvider} onChange={(event) => setForm((current) => ({ ...current, embeddingProvider: event.target.value }))} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="embedding-model">Embedding Model</Label>
                <Input id="embedding-model" value={form.embeddingModel} onChange={(event) => setForm((current) => ({ ...current, embeddingModel: event.target.value }))} />
              </div>
              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="embedding-base-url">Embedding Base URL</Label>
                <Input id="embedding-base-url" value={form.embeddingBaseUrl} onChange={(event) => setForm((current) => ({ ...current, embeddingBaseUrl: event.target.value }))} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="embedding-key-mode">Embedding API Key Mode</Label>
                <Input id="embedding-key-mode" value={form.embeddingApiKeyMode} onChange={(event) => setForm((current) => ({ ...current, embeddingApiKeyMode: event.target.value }))} />
              </div>
              <div className="md:col-span-2">
                <Button disabled={loading}>{loading ? "저장 중..." : "Save model settings"}</Button>
              </div>
            </form>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

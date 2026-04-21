import { FormEvent, useMemo, useState } from "react";

import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/shared/ui/card";
import { Input } from "@/shared/ui/input";
import { Label } from "@/shared/ui/label";
import type { OcpConnectionRequest } from "@/domains/connection/types";
import type { OcpConnectionController } from "./useOcpConnection";

type OcpConnectionFormProps = {
  controller: OcpConnectionController;
};

function formatDateTime(value: string) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

export function OcpConnectionForm({ controller }: OcpConnectionFormProps) {
  const [form, setForm] = useState<OcpConnectionRequest>({
    workspaceId: "",
    clusterUrl: "",
    authMode: "token",
    verifySsl: true,
    defaultNamespace: "",
    displayName: "",
    saveProfile: false,
    token: "",
    username: "",
    password: "",
  });

  const {
    submitState,
    isBusy,
    profile,
    testResult,
    savedProfiles,
    activeSavedProfileId,
    createConnection,
    connectSavedProfile,
    removeSavedProfile,
    runConnectionTest,
    refreshLeaseMetadata,
    clearConnection,
  } = controller;

  useMemo(() => {
    setForm((current) => ({ ...current, workspaceId: controller.profile?.workspaceId ?? current.workspaceId ?? "" }));
  }, [controller.profile?.workspaceId]);

  const helperText = useMemo(() => {
    if (form.authMode === "token") {
      return "권장: `oc whoami -t` 또는 콘솔에서 발급한 bearer token을 입력하세요.";
    }
    if (form.authMode === "password") {
      return "현재는 backend token exchange 경유 검증 흐름이라 token 방식보다 제약이 있습니다.";
    }
    return "OAuth redirect / device flow는 다음 단계에서 연결 예정입니다.";
  }, [form.authMode]);

  const connectedUser =
    testResult?.resolvedUser || profile?.usernameHint || profile?.displayName || "현재 사용자";
  const roleSummary = testResult?.resolvedRoles?.length
    ? testResult.resolvedRoles.join(", ")
    : "연결 테스트 전";
  const verificationSummary = testResult?.success ? "검증 완료" : "연결됨";

  function updateField<K extends keyof OcpConnectionRequest>(key: K, value: OcpConnectionRequest[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleConnect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await createConnection(form);
  }

  function applySavedProfileToForm(savedRequest: OcpConnectionRequest) {
    setForm(savedRequest);
  }

  return (
    <div className="space-y-6">
      {profile ? (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Card className="border-border/70 bg-background/60">
              <CardHeader className="pb-3">
                <CardDescription>User</CardDescription>
                <CardTitle className="break-all text-base leading-6">{connectedUser}</CardTitle>
              </CardHeader>
              <CardContent className="pt-0 text-sm text-muted-foreground">
                검증 상태: {verificationSummary}
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-background/60">
              <CardHeader className="pb-3">
                <CardDescription>Cluster</CardDescription>
                <CardTitle className="break-all text-base">{profile.clusterUrl}</CardTitle>
              </CardHeader>
              <CardContent className="pt-0 text-sm text-muted-foreground">
                Namespace: {testResult?.resolvedNamespace || profile.defaultNamespace || "-"}
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-background/60">
              <CardHeader className="pb-3">
                <CardDescription>Roles</CardDescription>
                <CardTitle className="break-words text-base leading-6">{roleSummary}</CardTitle>
              </CardHeader>
              <CardContent className="pt-0 text-sm text-muted-foreground">
                Auth Mode: {profile.authMode}
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-background/60 md:col-span-2 xl:col-span-3">
              <CardHeader className="pb-3">
                <CardDescription>Quick Actions</CardDescription>
                <CardTitle className="text-base">Validate and control</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="grid gap-2 md:grid-cols-3">
                  <Button
                    className="h-auto whitespace-normal px-3 py-2 text-left leading-5"
                    variant="secondary"
                    type="button"
                    disabled={isBusy}
                    onClick={runConnectionTest}
                  >
                    {submitState === "testing" ? "연결 테스트 중..." : "연결 테스트"}
                  </Button>
                  <Button
                    className="h-auto whitespace-normal px-3 py-2 text-left leading-5"
                    variant="outline"
                    type="button"
                    disabled={isBusy}
                    onClick={refreshLeaseMetadata}
                  >
                    {submitState === "testing" ? "lease 메타데이터 갱신 중..." : "lease 메타데이터 갱신"}
                  </Button>
                  <Button
                    className="h-auto whitespace-normal px-3 py-2 text-left leading-5"
                    variant="ghost"
                    type="button"
                    disabled={isBusy}
                    onClick={clearConnection}
                  >
                    {submitState === "disconnecting" ? "로그아웃 중..." : "로그아웃"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="border-border/70 bg-background/40">
            <CardHeader>
              <CardTitle>Connected profile</CardTitle>
              <CardDescription>현재 연결된 세션의 식별 정보와 검증 결과입니다.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm text-muted-foreground md:grid-cols-2">
              <div><span className="font-medium text-foreground">표시 이름:</span> {profile.displayName || "-"}</div>
              <div><span className="font-medium text-foreground">연결 상태:</span> {profile.status}</div>
              <div><span className="font-medium text-foreground">최근 검증:</span> {formatDateTime(profile.lastVerifiedAt)}</div>
              <div><span className="font-medium text-foreground">만료 시각:</span> {formatDateTime(testResult?.expiresAt || profile.expiresAt)}</div>
              <div className="md:col-span-2"><span className="font-medium text-foreground">Secret Ref:</span> {profile.secretRef}</div>
            </CardContent>
          </Card>
        </>
      ) : (
        <form onSubmit={handleConnect} className="space-y-6">
          <div className="rounded-2xl border border-border/70 bg-background/50 p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">Start Here</div>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  먼저 연결 프로필을 생성하면, 이후 같은 화면에서 검증과 lease 메타데이터 갱신을 계속 진행할 수 있습니다.
                </p>
              </div>
              <div className="flex gap-2">
                <Button disabled={isBusy}>
                  {submitState === "connecting" ? "연결 프로필 생성 중..." : "연결 프로필 생성"}
                </Button>
                <Button variant="outline" type="button" disabled>
                  연결 테스트
                </Button>
              </div>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="cluster-url">Cluster URL</Label>
              <Input
                id="cluster-url"
                placeholder="https://api.cluster.example.com:6443"
                value={form.clusterUrl}
                onChange={(event) => updateField("clusterUrl", event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="auth-mode">Auth Mode</Label>
              <select
                id="auth-mode"
                className="flex h-9 w-full rounded-md border border-border bg-input px-3 py-1 text-sm"
                value={form.authMode}
                onChange={(event) =>
                  updateField("authMode", event.target.value as OcpConnectionRequest["authMode"])
                }
              >
                <option value="token">Token</option>
                <option value="password">ID / Password</option>
                <option value="oauth_future">OAuth (Future)</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="display-name">Display Name</Label>
              <Input
                id="display-name"
                placeholder="예: dev-cluster"
                value={form.displayName}
                onChange={(event) => updateField("displayName", event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="default-namespace">Default Namespace</Label>
              <Input
                id="default-namespace"
                placeholder="예: openshift-monitoring"
                value={form.defaultNamespace}
                onChange={(event) => updateField("defaultNamespace", event.target.value)}
              />
            </div>
          </div>

          {form.authMode === "token" ? (
            <div className="space-y-2">
              <Label htmlFor="bearer-token">Bearer Token</Label>
              <Input
                id="bearer-token"
                type="password"
                placeholder="sha256~..."
                value={form.token ?? ""}
                onChange={(event) => updateField("token", event.target.value)}
              />
            </div>
          ) : null}

          {form.authMode === "password" ? (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  placeholder="예: developer"
                  value={form.username ?? ""}
                  onChange={(event) => updateField("username", event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="비밀번호"
                  value={form.password ?? ""}
                  onChange={(event) => updateField("password", event.target.value)}
                />
              </div>
            </div>
          ) : null}

          <div className="flex flex-col gap-3 rounded-2xl border border-border/70 bg-background/40 p-4 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <div className="font-medium text-foreground">Connection guidance</div>
              <div>{helperText}</div>
            </div>
            <div className="flex flex-col gap-3 sm:items-end">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-[#1060ff]"
                  checked={form.verifySsl}
                  onChange={(event) => updateField("verifySsl", event.target.checked)}
                />
                SSL 검증 사용
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-[#1060ff]"
                  checked={form.saveProfile}
                  onChange={(event) => updateField("saveProfile", event.target.checked)}
                />
                백엔드에 프로필 저장
              </label>
            </div>
          </div>
        </form>
      )}

      <Card className="border-border/70 bg-background/40">
        <CardHeader>
          <CardTitle>Saved profiles</CardTitle>
          <CardDescription>한 번 저장하면 불러오기와 재연결을 빠르게 할 수 있습니다.</CardDescription>
        </CardHeader>
        <CardContent>
          {savedProfiles.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2">
              {savedProfiles.map((item) => {
                const active = item.id === activeSavedProfileId && !!profile;
                return (
                  <Card
                    key={item.id}
                    className={active ? "border-primary/60 bg-primary/5" : "border-border/70 bg-background/60"}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-2">
                          <CardTitle className="text-base">{item.displayName || item.clusterUrl}</CardTitle>
                          <div className="flex flex-wrap gap-2">
                            <Badge variant={active ? "success" : "secondary"}>
                              {active ? "Connected" : item.authMode}
                            </Badge>
                            <Badge variant="outline">{item.defaultNamespace || "namespace -"}</Badge>
                          </div>
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          disabled={isBusy}
                          aria-label={`${item.displayName || item.clusterUrl} 저장 프로필 제거`}
                          onClick={() => removeSavedProfile(item.id)}
                        >
                          Remove
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3 pt-0 text-sm text-muted-foreground">
                      <div className="break-all">{item.clusterUrl}</div>
                      <div>
                        Namespace: {item.defaultNamespace || "-"} · User: {item.lastResolvedUser || item.usernameHint || "-"}
                      </div>
                      <div>
                        최근 연결: {formatDateTime(item.lastConnectedAt)} · 최근 검증: {formatDateTime(item.lastVerifiedAt)}
                      </div>
                      <div className="space-y-2">
                        <Button
                          type="button"
                          variant="outline"
                          className="w-full"
                          disabled={isBusy}
                          onClick={() => applySavedProfileToForm(item.request)}
                        >
                          Load values
                        </Button>
                        <div className="text-xs text-muted-foreground">
                          저장된 값을 불러온 뒤 상단에서 현재 연결을 다시 생성하거나 테스트하세요.
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-border/70 bg-background/50 px-4 py-8 text-center text-sm text-muted-foreground">
              아직 저장된 연결 프로필이 없습니다.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}



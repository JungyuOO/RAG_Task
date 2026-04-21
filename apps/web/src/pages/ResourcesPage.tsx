import { useEffect, useMemo, useState } from "react";

import { PageHeader } from "@/shared/layout/PageHeader";
import { Button } from "@/shared/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/shared/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/shared/ui/dropdown-menu";
import type {
  OcpLiveNamespaceListResponse,
  OcpLiveResourceDetailResponse,
  OcpLiveResourceListResponse,
  OcpLiveResourceSummary,
} from "@/domains/connection/types";
import type { OcpConnectionController } from "@/domains/connection/useOcpConnection";
import { getOcpNamespaces, getOcpResourceDetail, getOcpResources } from "@/domains/resources/ocpResourcesApi";
import { ResourceList } from "@/domains/resources/ResourceList";
import {
  ResourceYamlEditorModal,
  type LiveResourceKind,
} from "@/domains/resources/ResourceYamlEditorModal";

const RESOURCE_OPTIONS: LiveResourceKind[] = ["pods", "deployments", "services", "routes", "events"];

type ResourcesPageProps = {
  controller: OcpConnectionController;
  onLoadingChange?: (state: { active: boolean; title: string; detail?: string }) => void;
};

export function ResourcesPage({ controller, onLoadingChange }: ResourcesPageProps) {
  const [resource, setResource] = useState<LiveResourceKind>("pods");
  const [namespace, setNamespace] = useState("");
  const [namespaces, setNamespaces] = useState<OcpLiveNamespaceListResponse | null>(null);
  const [resourceData, setResourceData] = useState<OcpLiveResourceListResponse | null>(null);
  const [selectedItem, setSelectedItem] = useState<OcpLiveResourceSummary | null>(null);
  const [resourceDetail, setResourceDetail] = useState<OcpLiveResourceDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [pageError, setPageError] = useState("");
  const [detailNotice, setDetailNotice] = useState("");
  const [editorOpen, setEditorOpen] = useState(false);

  useEffect(() => {
    async function loadNamespaces() {
      if (!controller.profile) {
        setNamespaces(null);
        setNamespace("");
        return;
      }
      try {
        const result = await getOcpNamespaces(controller.profile.connectionId);
        setNamespaces(result);
        setNamespace((current) => current || controller.profile?.defaultNamespace || result.items[0] || "");
      } catch (nextError) {
        setPageError(nextError instanceof Error ? nextError.message : "Failed to load namespaces.");
      }
    }
    void loadNamespaces();
  }, [controller.profile]);

  useEffect(() => {
    async function loadResources() {
      if (!controller.profile || !namespace) {
        setResourceData(null);
        setSelectedItem(null);
        setResourceDetail(null);
        setDetailNotice("");
        return;
      }
      setLoading(true);
      setPageError("");
      try {
        const result = await getOcpResources(controller.profile.connectionId, resource, namespace);
        setResourceData(result);
        const nextSelected = result.items[0] ?? null;
        setSelectedItem(nextSelected);
        setDetailNotice("");
      } catch (nextError) {
        setPageError(nextError instanceof Error ? nextError.message : "Failed to load resources.");
      } finally {
        setLoading(false);
      }
    }
    void loadResources();
  }, [controller.profile, namespace, resource]);

  useEffect(() => {
    async function loadDetail() {
      if (!controller.profile || !namespace || !selectedItem?.name) {
        setResourceDetail(null);
        setDetailNotice("");
        return;
      }
      setDetailLoading(true);
      setDetailNotice("");
      try {
        const result = await getOcpResourceDetail(
          controller.profile.connectionId,
          resource,
          namespace,
          selectedItem.name,
        );
        setResourceDetail(result);
      } catch (nextError) {
        setResourceDetail(null);
        setDetailNotice(nextError instanceof Error ? nextError.message : "Failed to load resource manifest.");
      } finally {
        setDetailLoading(false);
      }
    }
    void loadDetail();
  }, [controller.profile, namespace, resource, selectedItem]);

  useEffect(() => {
    const active = loading || detailLoading;
    onLoadingChange?.({
      active,
      title: detailLoading ? "리소스 상세 불러오는 중" : "리소스 목록 불러오는 중",
      detail: detailLoading
        ? "선택한 리소스의 live manifest를 가져오는 중입니다."
        : "namespace와 리소스 목록을 클러스터에서 조회하는 중입니다.",
    });
    return () => onLoadingChange?.({ active: false, title: "" });
  }, [detailLoading, loading, onLoadingChange]);

  const namespaceDropdown = useMemo(
    () => (
      <div className="flex flex-wrap gap-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline">Namespace: {namespace || "Select"}</Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Namespaces</DropdownMenuLabel>
            {(namespaces?.items ?? []).map((item) => (
              <DropdownMenuItem key={item} onSelect={() => setNamespace(item)}>
                {item}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    ),
    [namespace, namespaces?.items],
  );

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Resources"
        title="Resource explorer"
        description="Namespace별 live resources를 탐색하고 YAML manifest를 열어 수정 흐름으로 연결합니다."
        actions={namespaceDropdown}
      />

      {!controller.profile ? (
        <Card className="border-border/70 bg-background/50">
          <CardContent className="p-6 text-sm text-muted-foreground">
            Connection 화면에서 클러스터 연결을 먼저 구성해야 합니다.
          </CardContent>
        </Card>
      ) : null}

      {pageError ? (
        <Card className="border-destructive/40 bg-destructive/10">
          <CardContent className="p-6 text-sm text-destructive">{pageError}</CardContent>
        </Card>
      ) : null}

      {controller.profile ? (
        <>
          <Card className="border-border/70 bg-card/95">
            <CardHeader>
              <CardTitle>Scope</CardTitle>
              <CardDescription>리소스 종류와 namespace를 바꾸면 목록과 YAML이 함께 갱신됩니다.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {RESOURCE_OPTIONS.map((item) => (
                  <Button
                    key={item}
                    type="button"
                    size="sm"
                    variant={resource === item ? "default" : "outline"}
                    onClick={() => setResource(item)}
                  >
                    {item}
                  </Button>
                ))}
              </div>
              {resourceData ? (
                <div className="grid gap-4 md:grid-cols-4">
                  {[
                    { label: "Resource", value: resourceData.resource },
                    { label: "Namespace", value: resourceData.namespace },
                    { label: "Count", value: String(resourceData.count) },
                    { label: "Selected", value: selectedItem?.name || "-" },
                  ].map((item) => (
                    <div key={item.label} className="rounded-2xl border border-border/70 bg-background/50 p-4">
                      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        {item.label}
                      </div>
                      <div className="mt-3 text-lg font-semibold text-foreground">{item.value}</div>
                    </div>
                  ))}
                </div>
              ) : null}
            </CardContent>
          </Card>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
            <Card className="border-border/70 bg-card/95">
              <CardHeader>
                <CardTitle>Resource list</CardTitle>
                <CardDescription>항목을 클릭하면 오른쪽 패널에 YAML manifest가 표시됩니다.</CardDescription>
              </CardHeader>
              <CardContent>
                {resourceData ? (
                  <ResourceList
                    items={resourceData.items}
                    resource={resource}
                    namespace={namespace}
                    selectedName={selectedItem?.name || ""}
                    onSelect={(item) => {
                      setSelectedItem(item);
                      setDetailNotice("");
                    }}
                  />
                ) : (
                  <div className="text-sm text-muted-foreground">리소스 목록을 불러오는 중입니다.</div>
                )}
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-card/95">
              <CardHeader>
                <CardTitle>YAML manifest</CardTitle>
                <CardDescription>선택한 리소스의 원문 manifest를 확인하고 필요 시 수정 흐름으로 이동합니다.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {detailNotice ? (
                  <div className="rounded-xl border border-border/70 bg-background/50 px-4 py-3 text-sm text-muted-foreground">
                    {detailNotice}
                  </div>
                ) : null}
                {resourceDetail ? (
                  <>
                    <div className="grid gap-3 text-sm text-muted-foreground md:grid-cols-2">
                      <div><span className="font-medium text-foreground">Name:</span> {resourceDetail.name}</div>
                      <div><span className="font-medium text-foreground">Kind:</span> {resourceDetail.kind}</div>
                      <div className="md:col-span-2"><span className="font-medium text-foreground">Namespace:</span> {resourceDetail.namespace}</div>
                    </div>
                    <div className="flex gap-2">
                      <Button type="button" variant="outline" onClick={() => setEditorOpen(true)}>
                        Open YAML editor
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        onClick={() => {
                          if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
                            void navigator.clipboard.writeText(resourceDetail.manifestYaml);
                            setDetailNotice("리소스 YAML을 복사했습니다.");
                          }
                        }}
                      >
                        YAML 복사
                      </Button>
                    </div>
                    <pre className="max-h-[60vh] overflow-auto rounded-xl border border-border/70 bg-background/70 p-4 text-xs leading-6 text-muted-foreground">
                      <code>{resourceDetail.manifestYaml}</code>
                    </pre>
                  </>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    {detailLoading ? "상세 manifest를 불러오는 중입니다." : "리소스를 선택하면 YAML manifest가 여기 표시됩니다."}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}

      {controller.profile && selectedItem ? (
        <ResourceYamlEditorModal
          controller={controller}
          resource={resource}
          namespace={namespace}
          name={selectedItem.name}
          open={editorOpen}
          onClose={() => setEditorOpen(false)}
          initialDetail={resourceDetail}
          initialSummary={selectedItem}
          onSaved={(detail) => {
            setResourceDetail(detail);
            setDetailNotice("YAML 변경이 적용되었습니다.");
          }}
        />
      ) : null}
    </div>
  );
}



import { useEffect, useMemo, useState } from "react";

import { PageHeader } from "@/shared/layout/PageHeader";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/shared/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog";
import { MarkdownArticle } from "@/shared/ui/MarkdownArticle";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/tabs";
import type { OcpConnectionController } from "@/domains/connection/useOcpConnection";
import { BatchReindexPanel } from "@/domains/library/BatchReindexPanel";
import {
  getLibraryCatalog,
  getLibraryDocumentChunks,
  getLibraryDocumentContent,
  getLibrarySummary,
  libraryDocumentFileUrl,
} from "@/domains/library/libraryApi";
import type {
  LibraryCatalogResponse,
  LibraryDocumentChunksResponse,
  LibraryDocumentContentResponse,
  LibraryDocumentRecord,
  LibrarySummaryResponse,
} from "@/domains/library/types";
import type { WorkspaceRecord } from "@/domains/workspaces/types";

type DetailMode = "chunks" | "original";
type LibraryPageProps = {
  controller: OcpConnectionController;
  selectedWorkspace: WorkspaceRecord | null;
  onLoadingChange?: (state: { active: boolean; title: string; detail?: string }) => void;
};

function buildChunkPreview(sectionTitle: string, previewText: string) {
  const normalizedTitle = sectionTitle.replace(/\s+/g, " ").trim();
  const normalizedPreview = previewText.replace(/\s+/g, " ").trim();
  if (!normalizedTitle || !normalizedPreview) return previewText.trim();
  if (normalizedPreview.startsWith(normalizedTitle)) {
    const next = normalizedPreview.slice(normalizedTitle.length).trim();
    if (next) return next;
  }
  return previewText.trim();
}

export function LibraryPage({ controller, selectedWorkspace, onLoadingChange }: LibraryPageProps) {
  const [summary, setSummary] = useState<LibrarySummaryResponse | null>(null);
  const [catalog, setCatalog] = useState<LibraryCatalogResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [activeDocumentKey, setActiveDocumentKey] = useState("");
  const [detailMode, setDetailMode] = useState<DetailMode>("chunks");
  const [detailOpen, setDetailOpen] = useState(false);
  const [chunkData, setChunkData] = useState<LibraryDocumentChunksResponse | null>(null);
  const [contentData, setContentData] = useState<LibraryDocumentContentResponse | null>(null);

  useEffect(() => {
    async function run() {
      setLoading(true);
      setError("");
      try {
        const workspaceId = selectedWorkspace?.workspaceId ?? "";
        const [nextSummary, nextCatalog] = await Promise.all([
          getLibrarySummary(workspaceId),
          getLibraryCatalog(workspaceId),
        ]);
        setSummary(nextSummary);
        setCatalog(nextCatalog);
      } catch (nextError) {
        setError(nextError instanceof Error ? nextError.message : "Failed to load library data.");
      } finally {
        setLoading(false);
      }
    }
    void run();
  }, [selectedWorkspace?.workspaceId]);

  const visibleDocuments = useMemo(() => catalog?.documents ?? [], [catalog]);
  const activeDocument = useMemo(
    () => visibleDocuments.find((item) => item.documentKey === activeDocumentKey) ?? null,
    [activeDocumentKey, visibleDocuments],
  );

  useEffect(() => {
    const active = loading || detailLoading;
    onLoadingChange?.({
      active,
      title: detailLoading ? "문서 상세 불러오는 중" : "자료실 불러오는 중",
      detail: detailLoading
        ? "선택한 문서의 청크 또는 원문을 가져오는 중입니다."
        : "자료실 요약과 문서 목록을 가져오는 중입니다.",
    });
    return () => onLoadingChange?.({ active: false, title: "" });
  }, [detailLoading, loading, onLoadingChange]);

  async function activateDocument(document: LibraryDocumentRecord, mode: DetailMode) {
    setActiveDocumentKey(document.documentKey);
    setDetailMode(mode);
    setDetailLoading(true);
    setError("");
    try {
      if (mode === "chunks") {
        setContentData(null);
        setChunkData(await getLibraryDocumentChunks(document.documentKey, selectedWorkspace?.workspaceId ?? ""));
      } else if (document.originalKind === "markdown") {
        setChunkData(null);
        setContentData(await getLibraryDocumentContent(document.originalKey, selectedWorkspace?.workspaceId ?? ""));
      } else {
        setChunkData(null);
        setContentData(null);
      }
      setDetailOpen(true);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to load document detail.");
    } finally {
      setDetailLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Library"
        title="Document library"
        description="공식 문서 카탈로그와 배치 색인 작업을 한 화면에서 관리합니다."
      />

      {selectedWorkspace ? (
        <div className="surface-muted rounded-2xl px-4 py-3 text-sm text-muted-foreground">
          Active workspace: <span className="font-medium text-foreground">{selectedWorkspace.name}</span>
        </div>
      ) : null}

      {error ? (
        <Card className="surface-danger">
          <CardContent className="p-6 text-sm text-destructive">{error}</CardContent>
        </Card>
      ) : null}

      <Tabs defaultValue="summary" className="w-full space-y-4">
        <TabsList>
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="catalog">Catalog</TabsTrigger>
        </TabsList>

        <TabsContent value="summary">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {[
              { label: "Official Docs", value: String(catalog?.documents.length ?? 0), hint: "Markdown render available" },
              { label: "Indexed Documents", value: String(summary?.indexedDocuments ?? 0), hint: `${summary?.indexedChunks ?? 0} total chunks` },
              { label: "Connected Cluster", value: controller.profile?.clusterUrl ?? "Not connected", hint: controller.profile?.defaultNamespace || "offline" },
              { label: "Corpus Files", value: String(summary?.corpusFiles ?? 0), hint: summary?.latestBatchStatus || "idle" },
            ].map((item) => (
              <Card key={item.label} className="surface-soft">
                <CardHeader className="pb-3">
                  <CardDescription>{item.label}</CardDescription>
                  <CardTitle className="text-base break-all">{item.value}</CardTitle>
                </CardHeader>
                <CardContent className="pt-0 text-sm text-muted-foreground">{item.hint}</CardContent>
              </Card>
            ))}
          </div>
          {summary?.sourceBreakdown?.length ? (
            <Card className="surface-soft mt-4">
              <CardHeader>
                <CardTitle>Source breakdown</CardTitle>
                <CardDescription>{summary.message}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {summary.sourceBreakdown.map((item) => (
                  <div key={item.label} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-foreground">{item.label}</span>
                      <span className="text-muted-foreground">{item.count}</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${Math.max(8, (item.count / Math.max(...summary.sourceBreakdown.map((entry) => entry.count), 1)) * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          ) : null}
          <BatchReindexPanel workspaceId={selectedWorkspace?.workspaceId ?? ""} />
        </TabsContent>

        <TabsContent value="catalog">
          <Card className="surface-soft">
            <CardHeader>
              <CardTitle>Catalog</CardTitle>
              <CardDescription>문서를 골라 청크 또는 원문 뷰로 확인합니다.</CardDescription>
            </CardHeader>
            <CardContent>
              {visibleDocuments.length === 0 ? (
                <div className="text-sm text-muted-foreground">표시할 문서가 없습니다.</div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Title</TableHead>
                      <TableHead>Path</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Chunks</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {visibleDocuments.map((document) => (
                      <TableRow key={document.documentKey}>
                        <TableCell className="font-medium">{document.title}</TableCell>
                        <TableCell className="max-w-[320px] truncate">{document.relativePath}</TableCell>
                        <TableCell>
                          <Badge variant={document.indexed ? "success" : "warning"}>
                            {document.indexed ? "indexed" : "not indexed"}
                          </Badge>
                        </TableCell>
                        <TableCell>{document.chunkCount}</TableCell>
                        <TableCell className="space-x-2">
                          <Button type="button" size="sm" variant="outline" onClick={() => activateDocument(document, "chunks")}>
                            청크 보기
                          </Button>
                          <Button type="button" size="sm" variant="secondary" onClick={() => activateDocument(document, "original")}>
                            {document.originalKind === "markdown" ? "원문 보기" : "원본 보기"}
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

      </Tabs>

      <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
        <DialogContent className="max-w-5xl data-[state=open]:!animate-none data-[state=closed]:!animate-none">
          <DialogHeader>
            <DialogTitle>
              {activeDocument
                ? `${activeDocument.title} · ${detailMode === "chunks" ? "Chunk Inspector" : "Original View"}`
                : "Document Detail"}
            </DialogTitle>
            <DialogDescription>
              {activeDocument ? activeDocument.relativePath : "문서를 선택하면 상세 정보가 여기에 표시됩니다."}
            </DialogDescription>
          </DialogHeader>

          {!activeDocument ? (
            <div className="text-sm text-muted-foreground">왼쪽 문서 목록에서 보고 싶은 문서를 선택하세요.</div>
          ) : null}

          {activeDocument && detailMode === "chunks" && chunkData ? (
            <div className="space-y-3 max-h-[60vh] overflow-auto pr-1">
              {chunkData.chunks.map((chunk, index) => (
                <div key={chunk.chunkId} className="surface-muted rounded-xl p-4 text-sm text-muted-foreground">
                  <div className="font-medium text-foreground">Chunk {index + 1}</div>
                  <div>chunk={chunk.chunkId} · page={chunk.pageNumber ?? "-"}</div>
                  <div>block={chunk.blockTypes.join(", ") || "-"}</div>
                  {chunk.sectionTitle ? <div>section={chunk.sectionTitle}</div> : null}
                  <div className="mt-2 leading-6">{buildChunkPreview(chunk.sectionTitle, chunk.previewText)}</div>
                </div>
              ))}
            </div>
          ) : null}

          {activeDocument && detailMode === "original" && activeDocument.originalKind === "markdown" && contentData ? (
            <div className="max-h-[60vh] overflow-auto pr-1">
              <MarkdownArticle content={contentData.content} />
            </div>
          ) : null}

          {activeDocument && detailMode === "original" && activeDocument.originalKind === "pdf" ? (
            <iframe
              title={`${activeDocument.title} pdf viewer`}
              className="surface-muted h-[70vh] w-full rounded-xl"
              src={libraryDocumentFileUrl(activeDocument.originalKey, selectedWorkspace?.workspaceId ?? "")}
            />
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}



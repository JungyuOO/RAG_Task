import { useState } from "react";

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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/ui/table";
import type { BatchIndexRequest } from "@/domains/library/types";
import { useBatchIndexJob } from "./useBatchIndexJob";

function formatTimestamp(value: string) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export function BatchReindexPanel() {
  const [form, setForm] = useState<BatchIndexRequest>({
    rootPath: "data",
    sourceType: "generated-manual",
    documentGroup: "official_ocp",
    versionTag: "",
    locale: "",
    maxFiles: 3,
    includeSubdirectories: true,
  });
  const { job, history, isSubmitting, error, submit, retryFailed, cancel } = useBatchIndexJob();
  const hasFailedItems = Boolean(job?.result?.items.some((item) => Boolean(item.error)));
  const canCancel = Boolean(job && (job.status === "pending" || job.status === "running"));

  return (
    <div className="space-y-4">
      <Card className="border-border/70 bg-card/95">
        <CardHeader>
          <CardTitle>Batch reindex</CardTitle>
          <CardDescription>여러 source를 한 번에 태우고 submit, polling, retry, cancel까지 같은 화면에서 관리합니다.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="batch-root-path">Root Path</Label>
              <Input
                id="batch-root-path"
                value={form.rootPath ?? ""}
                onChange={(event) => setForm((current) => ({ ...current, rootPath: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-source-type">Source Type</Label>
              <select
                id="batch-source-type"
                className="flex h-9 w-full rounded-md border border-border bg-input px-3 py-1 text-sm"
                value={form.sourceType ?? ""}
                onChange={(event) =>
                  setForm((current) => ({ ...current, sourceType: event.target.value as BatchIndexRequest["sourceType"] }))
                }
              >
                <option value="generated-manual">generated-manual</option>
                <option value="html">html</option>
                <option value="html-single">html-single</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-document-group">Document Group</Label>
              <select
                id="batch-document-group"
                className="flex h-9 w-full rounded-md border border-border bg-input px-3 py-1 text-sm"
                value={form.documentGroup ?? ""}
                onChange={(event) =>
                  setForm((current) => ({ ...current, documentGroup: event.target.value as BatchIndexRequest["documentGroup"] }))
                }
              >
                <option value="official_ocp">official_ocp</option>
                <option value="customer_generated">customer_generated</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-version-tag">Version Tag</Label>
              <Input
                id="batch-version-tag"
                value={form.versionTag ?? ""}
                onChange={(event) => setForm((current) => ({ ...current, versionTag: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-locale">Locale</Label>
              <Input
                id="batch-locale"
                value={form.locale ?? ""}
                onChange={(event) => setForm((current) => ({ ...current, locale: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-max-files">Max Files</Label>
              <Input
                id="batch-max-files"
                type="number"
                min={0}
                value={form.maxFiles ?? 0}
                onChange={(event) => setForm((current) => ({ ...current, maxFiles: Number(event.target.value) || 0 }))}
              />
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button type="button" onClick={() => submit(form)} disabled={isSubmitting}>
              {isSubmitting ? "job 생성 중..." : "batch reindex 시작"}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => job && retryFailed(job.jobId)}
              disabled={isSubmitting || !job || !hasFailedItems}
            >
              실패 항목만 재시도
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => job && cancel(job.jobId)}
              disabled={isSubmitting || !canCancel}
            >
              현재 job 취소
            </Button>
          </div>

          {error ? (
            <div className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          ) : null}

          {job ? (
            <Card className="border-border/70 bg-background/50">
              <CardHeader>
                <CardTitle>Current job</CardTitle>
                <CardDescription>{job.jobId}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-4">
                  {[
                    { label: "Status", value: job.status },
                    { label: "Progress", value: `${job.progressPct}%` },
                    { label: "Current", value: job.currentFile || "-" },
                    { label: "Failed items", value: String(job.result?.failedFiles ?? 0) },
                  ].map((item) => (
                    <div key={item.label} className="rounded-xl border border-border/70 bg-card/80 p-3">
                      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">{item.label}</div>
                      <div className="mt-2 text-sm font-medium text-foreground">{item.value}</div>
                    </div>
                  ))}
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-secondary">
                  <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${Math.max(4, job.progressPct)}%` }} />
                </div>

                {job.result ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Source</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Indexed</TableHead>
                        <TableHead>Chunks</TableHead>
                        <TableHead>Error</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {job.result.items.slice(0, 8).map((item) => (
                        <TableRow key={item.sourcePath}>
                          <TableCell className="max-w-[320px] truncate">{item.sourcePath}</TableCell>
                          <TableCell>{item.sourceType}</TableCell>
                          <TableCell>{item.indexed ? "yes" : "no"}</TableCell>
                          <TableCell>{item.chunks}</TableCell>
                          <TableCell className="text-destructive">{item.error || "-"}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-sm text-muted-foreground">job 결과를 대기 중입니다.</div>
                )}
              </CardContent>
            </Card>
          ) : null}
        </CardContent>
      </Card>

      <Card className="border-border/70 bg-card/95">
        <CardHeader>
          <CardTitle>Recent jobs</CardTitle>
          <CardDescription>가장 최근의 batch reindex 이력입니다.</CardDescription>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <div className="text-sm text-muted-foreground">기록된 batch job이 없습니다.</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Job</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Current</TableHead>
                  <TableHead>Updated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.map((item) => (
                  <TableRow key={item.jobId}>
                    <TableCell className="font-medium">{item.jobId}</TableCell>
                    <TableCell>{item.status}</TableCell>
                    <TableCell>{item.progressPct}%</TableCell>
                    <TableCell>{item.currentFile || "-"}</TableCell>
                    <TableCell>{formatTimestamp(item.updatedAt)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}



import { useEffect, useMemo, useState } from "react";

import type { OcpActionPreviewResponse } from "@/domains/actions/types";
import type { OcpLiveResourceDetailResponse, OcpLiveResourceSummary } from "@/domains/connection/types";
import type { OcpConnectionController } from "@/domains/connection/useOcpConnection";
import {
  approveOcpActionRequest,
  createOcpActionRequest,
  executeOcpActionRequest,
  previewOcpAction,
} from "@/domains/actions/actionPreviewApi";
import { getOcpResourceDetail } from "@/domains/resources/ocpResourcesApi";
import { Button } from "@/shared/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/shared/ui/dialog";

const YAML_EDITABLE_RESOURCES = ["deployments", "services", "routes"] as const;
type EditableResourceKind = (typeof YAML_EDITABLE_RESOURCES)[number];
export type LiveResourceKind = "pods" | "deployments" | "services" | "routes" | "events";

type ResourceYamlEditorModalProps = {
  controller: OcpConnectionController;
  resource: LiveResourceKind;
  namespace: string;
  name: string;
  open: boolean;
  onClose: () => void;
  initialDetail?: OcpLiveResourceDetailResponse | null;
  initialSummary?: OcpLiveResourceSummary | null;
  onSaved?: (detail: OcpLiveResourceDetailResponse) => void;
};

async function copyToClipboard(text: string) {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  throw new Error("Clipboard API is unavailable in this browser.");
}

function getManifestResourceVersion(detail: OcpLiveResourceDetailResponse | null) {
  if (!detail) return "";
  const metadata = detail.manifestJson?.metadata as { resourceVersion?: string } | undefined;
  return String(metadata?.resourceVersion || "");
}

function normalizeYamlText(value: string) {
  return value
    .split("\n")
    .map((line) => line.trimEnd())
    .join("\n")
    .trim();
}

function extractRootSpecReplicas(yamlText: string): number | null {
  const lines = yamlText.split("\n");
  let inRootSpec = false;

  for (const line of lines) {
    const trimmed = line.trim();
    const indent = line.length - line.trimStart().length;

    if (!inRootSpec) {
      if (indent === 0 && trimmed === "spec:") {
        inRootSpec = true;
      }
      continue;
    }

    if (indent === 0 && trimmed) {
      break;
    }

    if (indent === 2 && trimmed.startsWith("replicas:")) {
      const value = trimmed.slice("replicas:".length).trim();
      const parsed = Number.parseInt(value, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
  }

  return null;
}

function replaceRootSpecReplicas(yamlText: string, nextReplicas: number): string {
  const lines = yamlText.split("\n");
  let inRootSpec = false;

  return lines
    .map((line) => {
      const trimmed = line.trim();
      const indent = line.length - line.trimStart().length;

      if (!inRootSpec) {
        if (indent === 0 && trimmed === "spec:") {
          inRootSpec = true;
        }
        return line;
      }

      if (indent === 0 && trimmed) {
        inRootSpec = false;
        return line;
      }

      if (indent === 2 && trimmed.startsWith("replicas:")) {
        return `${" ".repeat(indent)}replicas: ${nextReplicas}`;
      }

      return line;
    })
    .join("\n");
}

function resolveReplicaOnlyChange(
  detail: OcpLiveResourceDetailResponse | null,
  draftYaml: string,
  resource: LiveResourceKind,
) {
  if (!detail || resource !== "deployments") {
    return { replicaOnly: false, desiredReplicas: null as number | null };
  }

  const currentReplicas = Number((detail.manifestJson?.spec as { replicas?: number } | undefined)?.replicas ?? 0);
  const desiredReplicas = extractRootSpecReplicas(draftYaml);

  if (desiredReplicas === null || desiredReplicas === currentReplicas) {
    return { replicaOnly: false, desiredReplicas };
  }

  const normalizedOriginal = normalizeYamlText(detail.manifestYaml);
  const normalizedDraftWithoutReplicaDiff = normalizeYamlText(
    replaceRootSpecReplicas(draftYaml, currentReplicas),
  );

  return {
    replicaOnly: normalizedOriginal === normalizedDraftWithoutReplicaDiff,
    desiredReplicas,
  };
}

export function ResourceYamlEditorModal({
  controller,
  resource,
  namespace,
  name,
  open,
  onClose,
  initialDetail = null,
  initialSummary = null,
  onSaved,
}: ResourceYamlEditorModalProps) {
  const [resourceDetail, setResourceDetail] = useState<OcpLiveResourceDetailResponse | null>(initialDetail);
  const [resourceSummary, setResourceSummary] = useState<OcpLiveResourceSummary | null>(initialSummary);
  const [detailLoading, setDetailLoading] = useState(false);
  const [editorError, setEditorError] = useState("");
  const [draftYaml, setDraftYaml] = useState(initialDetail?.manifestYaml ?? "");
  const [preview, setPreview] = useState<OcpActionPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [applyLoading, setApplyLoading] = useState(false);
  const [pendingRequestId, setPendingRequestId] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    setResourceDetail(initialDetail);
    setResourceSummary(initialSummary);
    setDraftYaml(initialDetail?.manifestYaml ?? "");
  }, [initialDetail, initialSummary]);

  useEffect(() => {
    async function loadDetail() {
      if (!open || !controller.profile) return;
      if (initialDetail && initialDetail.name === name && initialDetail.resource === resource && initialDetail.namespace === namespace) {
        return;
      }
      setDetailLoading(true);
      setEditorError("");
      try {
        const detail = await getOcpResourceDetail(controller.profile.connectionId, resource, namespace, name);
        setResourceDetail(detail);
        setDraftYaml(detail.manifestYaml);
        setPreview(null);
        setPendingRequestId("");
      } catch (nextError) {
        setEditorError(nextError instanceof Error ? nextError.message : "리소스 manifest를 불러오지 못했습니다.");
      } finally {
        setDetailLoading(false);
      }
    }
    void loadDetail();
  }, [controller.profile, initialDetail, name, namespace, open, resource]);

  const isYamlEditable = Boolean(
    resourceDetail && YAML_EDITABLE_RESOURCES.includes(resource as EditableResourceKind),
  );
  const hasDraftChanges = Boolean(resourceDetail && draftYaml !== resourceDetail.manifestYaml);
  const replicaOnlyChange = useMemo(
    () => resolveReplicaOnlyChange(resourceDetail, draftYaml, resource),
    [draftYaml, resource, resourceDetail],
  );
  const actorId = controller.testResult?.resolvedUser || controller.profile?.displayName || "ui-local";
  const actorRoles = controller.testResult?.resolvedRoles || [];

  useEffect(() => {
    if (!open) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, open]);

  if (!open) return null;

  async function handleCopyYaml(text: string, successMessage: string) {
    try {
      await copyToClipboard(text);
      setNotice(successMessage);
      setEditorError("");
    } catch (nextError) {
      setEditorError(nextError instanceof Error ? nextError.message : "YAML 복사에 실패했습니다.");
    }
  }

  async function handlePreview() {
    if (!controller.profile || !resourceDetail) return;
    setPreviewLoading(true);
    setEditorError("");
    try {
      const result = await previewOcpAction(
        replicaOnlyChange.replicaOnly && replicaOnlyChange.desiredReplicas !== null
          ? {
              connectionId: controller.profile.connectionId,
              actorId,
              actorRoles,
              actionType: "scale_deployment",
              namespace,
              resourceName: resourceDetail.name,
              replicas: replicaOnlyChange.desiredReplicas,
              reason: "edit resource yaml",
            }
          : {
              connectionId: controller.profile.connectionId,
              actorId,
              actorRoles,
              actionType: "yaml_apply",
              namespace,
              resourceName: resourceDetail.name,
              reason: "edit resource yaml",
              manifestYaml: draftYaml,
              resourceVersion: getManifestResourceVersion(resourceDetail),
            },
      );
      setPreview(result);
      setPendingRequestId("");
    } catch (nextError) {
      setEditorError(nextError instanceof Error ? nextError.message : "yaml preview를 불러오지 못했습니다.");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function handleApply(force = false) {
    if (!controller.profile || !resourceDetail) return;
    setApplyLoading(true);
    setEditorError("");
    try {
      let requestId = pendingRequestId;
      if (!requestId) {
        const created = await createOcpActionRequest(
          replicaOnlyChange.replicaOnly && replicaOnlyChange.desiredReplicas !== null
            ? {
                connectionId: controller.profile.connectionId,
                actorId,
                actorRoles,
                actionType: "scale_deployment",
                namespace,
                resourceName: resourceDetail.name,
                replicas: replicaOnlyChange.desiredReplicas,
                reason: "edit resource yaml",
              }
            : {
                connectionId: controller.profile.connectionId,
                actorId,
                actorRoles,
                actionType: "yaml_apply",
                namespace,
                resourceName: resourceDetail.name,
                reason: "edit resource yaml",
                manifestYaml: draftYaml,
                resourceVersion: getManifestResourceVersion(resourceDetail),
              },
        );
        requestId = created.requestId;
        setPendingRequestId(requestId);
        if (created.status !== "approved") {
          const approved = await approveOcpActionRequest(requestId, actorId, actorRoles, "approve yaml apply");
          if (approved.status !== "approved") {
            throw new Error("yaml apply request is still not approved.");
          }
        }
      }

      const executed = await executeOcpActionRequest(requestId, actorId, actorRoles, force);
      if (executed.status !== "succeeded") {
        if (executed.error.includes("field ownership conflict")) {
          setEditorError("field ownership conflict detected. Force Apply로 다시 시도할 수 있습니다.");
        } else {
          setEditorError(executed.error || "yaml apply 실행에 실패했습니다.");
        }
        return;
      }

      const refreshed = await getOcpResourceDetail(controller.profile.connectionId, resource, namespace, resourceDetail.name);
      setResourceDetail(refreshed);
      setDraftYaml(refreshed.manifestYaml);
      setPreview(null);
      setPendingRequestId("");
      setNotice("YAML 변경이 적용되었습니다.");
      onSaved?.(refreshed);
      onClose();
    } catch (nextError) {
      setEditorError(nextError instanceof Error ? nextError.message : "yaml apply 실행에 실패했습니다.");
    } finally {
      setApplyLoading(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => { if (!nextOpen) onClose(); }}>
      <DialogContent className="grid w-[min(100vw-2rem,72rem)] max-w-none grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden border-border/70 bg-card p-0 data-[state=open]:!animate-none data-[state=closed]:!animate-none">
        <DialogHeader className="border-b border-border/70 px-6 py-5 pr-14">
          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">YAML Editor</div>
            <DialogTitle id="resource-yaml-editor-title" className="font-display text-2xl font-semibold tracking-tight">
              {name}
            </DialogTitle>
            <DialogDescription>
              {(resourceDetail?.kind || resourceSummary?.kind || resource) as string} · {namespace}
            </DialogDescription>
          </div>
        </DialogHeader>

        <div className="space-y-4 overflow-y-auto px-6 py-5">
          {detailLoading ? <div className="rounded-xl border border-border/70 bg-background/50 px-4 py-3 text-sm text-muted-foreground">리소스 manifest를 불러오는 중입니다.</div> : null}
          {notice ? <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">{notice}</div> : null}
          {editorError ? <div className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">{editorError}</div> : null}
          {preview ? (
            <div className="space-y-4 rounded-2xl border border-border/70 bg-background/50 p-4">
              <div className="grid gap-3 md:grid-cols-4">
                {[
                  { label: "Dry Run", value: preview.dryRunStatus },
                  { label: "Allowed", value: preview.allowed ? "yes" : "no" },
                  { label: "Approval", value: `${preview.approvalStrategy} (${preview.requiredApprovals})` },
                  { label: "Resource", value: `${preview.namespace}/${preview.resourceName}` },
                ].map((item) => (
                  <div key={item.label} className="rounded-xl border border-border/70 bg-card/80 p-3">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">{item.label}</div>
                    <div className="mt-2 text-sm font-medium text-foreground">{item.value}</div>
                  </div>
                ))}
              </div>
              <div className={`rounded-xl border px-4 py-3 text-sm ${preview.allowed ? "border-primary/30 bg-primary/10 text-foreground" : "border-destructive/30 bg-destructive/10 text-destructive"}`}>
                <div><strong>Summary</strong>: {preview.summary}</div>
                <div><strong>Preview Command</strong>: {preview.previewCommand}</div>
                <div><strong>Next Step</strong>: {preview.nextStep}</div>
              </div>
              {preview.dryRunMessages.length > 0 ? (
                <div className="rounded-xl border border-border/70 bg-card/80 px-4 py-3 text-sm text-muted-foreground">
                  {preview.dryRunMessages.join(" / ")}
                </div>
              ) : null}
              {preview.validationMessages.length > 0 ? (
                <div className="rounded-xl border border-border/70 bg-card/80 px-4 py-3 text-sm text-muted-foreground">
                  {preview.validationMessages.join(" / ")}
                </div>
              ) : null}
              {preview.diffUnified ? (
                <pre className="max-h-60 overflow-auto rounded-xl border border-border/70 bg-background/80 p-4 text-xs leading-6 text-muted-foreground">
                  <code>{preview.diffUnified}</code>
                </pre>
              ) : null}
            </div>
          ) : null}

          {resourceDetail ? (
            <label className="block space-y-3">
              <span className="text-sm font-medium text-foreground">Draft YAML</span>
              <div>
                <Button
                  variant="outline"
                  type="button"
                  onClick={() => void handleCopyYaml(draftYaml, "Draft YAML을 복사했습니다.")}
                >
                  Draft 복사
                </Button>
              </div>
              <textarea
                className="h-[min(60vh,40rem)] w-full resize-none rounded-xl border border-border bg-input p-4 font-mono text-xs leading-6 text-foreground"
                value={draftYaml}
                onChange={(event) => {
                  setDraftYaml(event.target.value);
                  setPreview(null);
                  setPendingRequestId("");
                  setEditorError("");
                }}
                rows={28}
                disabled={!isYamlEditable}
              />
            </label>
          ) : null}
        </div>

        <DialogFooter className="border-t border-border/70 px-6 py-5">
          <Button
            variant="outline"
            type="button"
            onClick={() => {
              if (!resourceDetail) return;
              setDraftYaml(resourceDetail.manifestYaml);
              setPreview(null);
              setPendingRequestId("");
              setEditorError("");
            }}
            disabled={!resourceDetail}
          >
            Reset Draft
          </Button>
          <Button
            type="button"
            onClick={() => void handlePreview()}
            disabled={previewLoading || !hasDraftChanges || !isYamlEditable}
          >
            {previewLoading ? "Preview 생성 중..." : "Preview"}
          </Button>
          <Button
            variant="secondary"
            type="button"
            onClick={() => void handleApply(false)}
            disabled={applyLoading || !preview || !preview.allowed || !isYamlEditable}
          >
            {applyLoading ? "Apply 실행 중..." : "Apply"}
          </Button>
          <Button
            variant="outline"
            type="button"
            onClick={() => void handleApply(true)}
            disabled={applyLoading || !editorError.includes("field ownership conflict") || !isYamlEditable}
          >
            Force Apply
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}






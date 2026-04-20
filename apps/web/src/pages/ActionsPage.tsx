import { useEffect, useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  approveOcpActionRequest,
  createOcpActionRequest,
  executeOcpActionRequest,
  listOcpActionAudit,
  listOcpActionExecutions,
  listOcpActionRequests,
  previewOcpAction,
  rejectOcpActionRequest,
} from "@/features/actions/api/actionPreviewApi";
import { ActionPreviewDialog } from "@/features/actions/components/ActionPreviewDialog";
import { AuditTable } from "@/features/actions/components/AuditTable";
import { ExecutionsTable } from "@/features/actions/components/ExecutionsTable";
import { RequestsTable } from "@/features/actions/components/RequestsTable";
import type {
  OcpActionAuditRecord,
  OcpActionExecutionRecord,
  OcpActionPreviewResponse,
  OcpActionRequestRecord,
  OcpActionType,
} from "@/features/actions/types";
import type { OcpConnectionController } from "@/features/connection/hooks/useOcpConnection";

function joinOrDash(items: string[]) {
  return items.length > 0 ? items.join(", ") : "-";
}

export function ActionsPage({ controller }: { controller: OcpConnectionController }) {
  const [actorId, setActorId] = useState("ui-local");
  const [actorRolesInput, setActorRolesInput] = useState("operator");
  const [actionType, setActionType] = useState<OcpActionType>("scale_deployment");
  const [resourceName, setResourceName] = useState("");
  const [namespace, setNamespace] = useState(controller.profile?.defaultNamespace ?? "");
  const [replicas, setReplicas] = useState(1);
  const [reason, setReason] = useState("");
  const [breakGlass, setBreakGlass] = useState(false);
  const [breakGlassReason, setBreakGlassReason] = useState("");
  const [breakGlassTicket, setBreakGlassTicket] = useState("");
  const [preview, setPreview] = useState<OcpActionPreviewResponse | null>(null);
  const [requests, setRequests] = useState<OcpActionRequestRecord[]>([]);
  const [executions, setExecutions] = useState<OcpActionExecutionRecord[]>([]);
  const [auditItems, setAuditItems] = useState<OcpActionAuditRecord[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);

  const actorRoles = actorRolesInput
    .split(",")
    .map((role) => role.trim().toLowerCase())
    .filter(Boolean);

  async function refreshRequests() {
    try {
      setRequests(await listOcpActionRequests(20));
    } catch {
      // ignore
    }
  }

  async function refreshExecutions() {
    try {
      setExecutions(await listOcpActionExecutions(20));
    } catch {
      // ignore
    }
  }

  async function refreshAudit() {
    try {
      setAuditItems(await listOcpActionAudit(20));
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    void refreshRequests();
    void refreshExecutions();
    void refreshAudit();
  }, []);

  useEffect(() => {
    if (controller.testResult?.resolvedUser) {
      setActorId(controller.testResult.resolvedUser);
    }
    if (controller.testResult?.resolvedRoles?.length) {
      setActorRolesInput(controller.testResult.resolvedRoles.join(","));
    }
  }, [controller.testResult]);

  async function handlePreview() {
    if (!controller.profile) return;
    setLoading(true);
    setError("");
    try {
      setPreview(
        await previewOcpAction({
          connectionId: controller.profile.connectionId,
          actorId,
          actorRoles,
          actionType,
          namespace,
          resourceName,
          replicas,
          reason,
          breakGlass,
          breakGlassReason,
          breakGlassTicket,
        }),
      );
      setPreviewDialogOpen(true);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to load action preview.");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateApprovalRequest() {
    if (!controller.profile) return;
    setLoading(true);
    setError("");
    try {
      const created = await createOcpActionRequest({
        connectionId: controller.profile.connectionId,
        actorId,
        actorRoles,
        actionType,
        namespace,
        resourceName,
        replicas,
        reason,
        breakGlass,
        breakGlassReason,
        breakGlassTicket,
      });
      setPreview(created.preview);
      await refreshRequests();
      await refreshExecutions();
      await refreshAudit();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to create approval request.");
    } finally {
      setLoading(false);
    }
  }

  async function handleApprove(requestId: string) {
    setLoading(true);
    setError("");
    try {
      await approveOcpActionRequest(requestId, actorId, actorRoles, "approved from UI");
      await refreshRequests();
      await refreshExecutions();
      await refreshAudit();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to approve request.");
    } finally {
      setLoading(false);
    }
  }

  async function handleReject(requestId: string) {
    setLoading(true);
    setError("");
    try {
      await rejectOcpActionRequest(requestId, actorId, actorRoles, "rejected from UI");
      await refreshRequests();
      await refreshExecutions();
      await refreshAudit();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to reject request.");
    } finally {
      setLoading(false);
    }
  }

  async function handleExecute(requestId: string) {
    setLoading(true);
    setError("");
    try {
      await executeOcpActionRequest(requestId, actorId, actorRoles);
      await refreshExecutions();
      await refreshRequests();
      await refreshAudit();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to execute request.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Actions"
        title="Approvals and Guarded Actions"
        description="Manage preview, approval requests, approve/reject, execute, and audit flows for guarded cluster actions."
      />

      {controller.testResult ? (
        <Card className="border-border/70 bg-card/95">
          <CardHeader>
            <CardTitle>Resolved Connection Identity</CardTitle>
            <CardDescription>Current resolved user and RBAC summary for this connected cluster context.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 text-sm text-muted-foreground md:grid-cols-2">
            <div><span className="font-medium text-foreground">User:</span> {controller.testResult.resolvedUser || "-"}</div>
            <div><span className="font-medium text-foreground">Roles:</span> {joinOrDash(controller.testResult.resolvedRoles)}</div>
            <div><span className="font-medium text-foreground">Groups:</span> {joinOrDash(controller.testResult.resolvedGroups)}</div>
            <div><span className="font-medium text-foreground">Namespace:</span> {controller.testResult.resolvedNamespace || "-"}</div>
          </CardContent>
        </Card>
      ) : null}

      <Card className="border-border/70 bg-card/95">
        <CardHeader>
          <CardTitle>Action Request Builder</CardTitle>
          <CardDescription>Create previews and approval requests against the current cluster context.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {!controller.profile ? (
            <div className="text-sm text-muted-foreground">Connect a cluster first from the Connection page.</div>
          ) : (
            <>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="actor-id">Actor ID</Label>
                  <Input id="actor-id" value={actorId} onChange={(event) => setActorId(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="actor-roles">Actor Roles</Label>
                  <Input id="actor-roles" value={actorRolesInput} onChange={(event) => setActorRolesInput(event.target.value)} placeholder="operator,admin" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="action-type">Action Type</Label>
                  <select
                    id="action-type"
                    className="flex h-9 w-full rounded-md border border-border bg-input px-3 py-1 text-sm"
                    value={actionType}
                    onChange={(event) => setActionType(event.target.value as OcpActionType)}
                  >
                    <option value="scale_deployment">scale_deployment</option>
                    <option value="rollout_restart">rollout_restart</option>
                    <option value="log_bundle">log_bundle</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="namespace">Namespace</Label>
                  <Input id="namespace" value={namespace} onChange={(event) => setNamespace(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="resource-name">Resource Name</Label>
                  <Input id="resource-name" value={resourceName} onChange={(event) => setResourceName(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="replicas">Replicas</Label>
                  <Input id="replicas" type="number" min={0} value={replicas} onChange={(event) => setReplicas(Number(event.target.value) || 0)} />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="reason">Reason</Label>
                <textarea
                  id="reason"
                  className="min-h-[96px] w-full rounded-md border border-border bg-input px-3 py-2 text-sm"
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                />
              </div>

              <label className="flex items-center gap-2 rounded-xl border border-border/70 bg-background/50 px-4 py-3 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-[#1060ff]"
                  checked={breakGlass}
                  onChange={(event) => setBreakGlass(event.target.checked)}
                />
                Break-glass emergency override
              </label>

              {breakGlass ? (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="breakglass-ticket">Incident / Ticket</Label>
                    <Input id="breakglass-ticket" value={breakGlassTicket} onChange={(event) => setBreakGlassTicket(event.target.value)} placeholder="INC-2048" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="breakglass-reason">Emergency Justification</Label>
                    <textarea
                      id="breakglass-reason"
                      className="min-h-[96px] w-full rounded-md border border-border bg-input px-3 py-2 text-sm"
                      value={breakGlassReason}
                      onChange={(event) => setBreakGlassReason(event.target.value)}
                      placeholder="Explain why the standard guardrail must be bypassed."
                    />
                  </div>
                </div>
              ) : null}

              <div className="flex flex-wrap gap-2">
                <Button type="button" onClick={handlePreview} disabled={loading}>
                  {loading ? "Creating Preview..." : "Create Preview"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleCreateApprovalRequest}
                  disabled={loading || (preview !== null && !preview.allowed)}
                >
                  Create Approval Request
                </Button>
              </div>
            </>
          )}

          {error ? (
            <div className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Tabs defaultValue="requests" className="space-y-4">
        <TabsList>
          <TabsTrigger value="requests">Requests</TabsTrigger>
          <TabsTrigger value="executions">Executions</TabsTrigger>
          <TabsTrigger value="audit">Audit</TabsTrigger>
        </TabsList>

        <TabsContent value="requests">
          <Card className="border-border/70 bg-card/95">
            <CardHeader>
              <CardTitle>Requests</CardTitle>
              <CardDescription>Review request state, approval counts, and next actions.</CardDescription>
            </CardHeader>
            <CardContent>
              <RequestsTable
                actorId={actorId}
                loading={loading}
                requests={requests}
                onApprove={handleApprove}
                onReject={handleReject}
                onExecute={handleExecute}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="executions">
          <Card className="border-border/70 bg-card/95">
            <CardHeader>
              <CardTitle>Executions</CardTitle>
              <CardDescription>Review execution state, preflight checks, and output summary.</CardDescription>
            </CardHeader>
            <CardContent>
              <ExecutionsTable executions={executions} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audit">
          <Card className="border-border/70 bg-card/95">
            <CardHeader>
              <CardTitle>Audit</CardTitle>
              <CardDescription>Recent audit events and break-glass metadata.</CardDescription>
            </CardHeader>
            <CardContent>
              <AuditTable auditItems={auditItems} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <ActionPreviewDialog
        open={previewDialogOpen}
        preview={preview}
        onClose={() => setPreviewDialogOpen(false)}
      />
    </div>
  );
}

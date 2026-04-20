import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { OcpActionPreviewResponse } from "@/features/actions/types";

type ActionPreviewDialogProps = {
  open: boolean;
  preview: OcpActionPreviewResponse | null;
  onClose: () => void;
};

export function ActionPreviewDialog({ open, preview, onClose }: ActionPreviewDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(nextOpen) => { if (!nextOpen) onClose(); }}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Action preview</DialogTitle>
          <DialogDescription>실행 전 영향과 정책 검사 결과를 확인합니다.</DialogDescription>
        </DialogHeader>

        {preview ? (
          <div className="space-y-4">
            <div className="grid gap-3 md:grid-cols-4">
              {[
                { label: "Allowed", value: preview.allowed ? "yes" : "no" },
                { label: "Risk", value: preview.riskLevel },
                { label: "Approval", value: `${preview.approvalStrategy} (${preview.requiredApprovals})` },
                { label: "Namespace", value: preview.namespace || "-" },
              ].map((item) => (
                <div key={item.label} className="rounded-xl border border-border/70 bg-background/50 p-3">
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
            {preview.policyChecks.length > 0 ? (
              <div className="rounded-xl border border-border/70 bg-background/50 px-4 py-3 text-sm text-muted-foreground">
                <strong className="text-foreground">Policy Checks</strong>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  {preview.policyChecks.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </div>
            ) : null}
            {preview.blockedReasons.length > 0 ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                <strong>Blocked Reasons</strong>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  {preview.blockedReasons.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </div>
            ) : null}
            {preview.diffUnified ? (
              <pre className="max-h-[50vh] overflow-auto rounded-xl border border-border/70 bg-background/80 p-4 text-xs leading-6 text-muted-foreground">
                {preview.diffUnified}
              </pre>
            ) : null}
          </div>
        ) : null}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

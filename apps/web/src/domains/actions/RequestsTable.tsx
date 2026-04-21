import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/ui/table";
import type { OcpActionRequestRecord } from "@/domains/actions/types";

function formatTimestamp(value: string) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function statusVariant(status: string) {
  if (status === "approved" || status === "succeeded") return "success" as const;
  if (status === "pending") return "secondary" as const;
  if (status === "rejected" || status === "failed") return "destructive" as const;
  return "outline" as const;
}

type RequestsTableProps = {
  actorId: string;
  loading: boolean;
  requests: OcpActionRequestRecord[];
  onApprove: (requestId: string) => void;
  onReject: (requestId: string) => void;
  onExecute: (requestId: string) => void;
};

export function RequestsTable({
  actorId,
  loading,
  requests,
  onApprove,
  onReject,
  onExecute,
}: RequestsTableProps) {
  if (requests.length === 0) {
    return <div className="text-sm text-muted-foreground">생성된 action request가 없습니다.</div>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Request</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Action</TableHead>
          <TableHead>Approvals</TableHead>
          <TableHead>Summary</TableHead>
          <TableHead>Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {requests.map((item) => {
          const canApprove = !loading && item.status === "pending" && !item.approverIds.includes(actorId);
          const canReject = !loading && item.status === "pending";
          const canExecute = !loading && item.status === "approved";
          return (
            <TableRow key={item.requestId}>
              <TableCell className="font-medium">
                <div>{item.requestId}</div>
                <div className="text-xs text-muted-foreground">{formatTimestamp(item.updatedAt)}</div>
              </TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-2">
                  {item.preview.breakGlass ? <Badge variant="warning">break-glass</Badge> : null}
                  <Badge variant={statusVariant(item.status)}>{item.status}</Badge>
                </div>
              </TableCell>
              <TableCell>{item.preview.actionType}</TableCell>
              <TableCell>{item.approvalCount}/{item.requiredApprovals}</TableCell>
              <TableCell className="max-w-[340px] whitespace-normal text-sm text-muted-foreground">
                {item.preview.summary}
              </TableCell>
              <TableCell className="space-x-2">
                <Button size="sm" type="button" onClick={() => onApprove(item.requestId)} disabled={!canApprove}>approve</Button>
                <Button size="sm" variant="destructive" type="button" onClick={() => onReject(item.requestId)} disabled={!canReject}>reject</Button>
                <Button size="sm" variant="outline" type="button" onClick={() => onExecute(item.requestId)} disabled={!canExecute}>execute</Button>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}


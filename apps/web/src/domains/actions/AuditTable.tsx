import { Badge } from "@/shared/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/ui/table";
import type { OcpActionAuditRecord } from "@/domains/actions/types";

function formatTimestamp(value: string) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function readAuditString(details: Record<string, unknown>, key: string) {
  const value = details[key];
  return typeof value === "string" ? value : "";
}

function readAuditBool(details: Record<string, unknown>, key: string) {
  return details[key] === true;
}

type AuditTableProps = {
  auditItems: OcpActionAuditRecord[];
};

export function AuditTable({ auditItems }: AuditTableProps) {
  if (auditItems.length === 0) {
    return <div className="text-sm text-muted-foreground">기록된 audit event가 없습니다.</div>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Event</TableHead>
          <TableHead>Actor</TableHead>
          <TableHead>Action</TableHead>
          <TableHead>Risk</TableHead>
          <TableHead>Notes</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {auditItems.map((item) => (
          <TableRow key={item.eventId}>
            <TableCell className="font-medium">
              <div className="flex flex-wrap items-center gap-2">
                <span>{item.eventType}</span>
                {readAuditBool(item.details, "break_glass") ? <Badge variant="warning">break-glass</Badge> : null}
              </div>
              <div className="text-xs text-muted-foreground">{formatTimestamp(item.createdAt)}</div>
            </TableCell>
            <TableCell>{item.actorId}</TableCell>
            <TableCell>{item.actionType}</TableCell>
            <TableCell>{item.riskLevel}</TableCell>
            <TableCell className="max-w-[420px] whitespace-normal text-sm text-muted-foreground">
              {item.decisionNote || "-"}
              {readAuditBool(item.details, "break_glass") ? (
                <div className="mt-2 text-xs">
                  ticket={readAuditString(item.details, "break_glass_ticket") || "-"} · {readAuditString(item.details, "break_glass_reason") || "emergency override"}
                </div>
              ) : null}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}


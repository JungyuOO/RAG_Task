import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { OcpActionExecutionRecord } from "@/features/actions/types";

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

type ExecutionsTableProps = {
  executions: OcpActionExecutionRecord[];
};

export function ExecutionsTable({ executions }: ExecutionsTableProps) {
  if (executions.length === 0) {
    return <div className="text-sm text-muted-foreground">실행된 action이 없습니다.</div>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Execution</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Action</TableHead>
          <TableHead>Mode</TableHead>
          <TableHead>Summary</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {executions.map((item) => (
          <TableRow key={item.executionId}>
            <TableCell className="font-medium">
              <div>{item.executionId}</div>
              <div className="text-xs text-muted-foreground">{formatTimestamp(item.updatedAt)}</div>
            </TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-2">
                {item.preview.breakGlass ? <Badge variant="warning">break-glass</Badge> : null}
                <Badge variant={statusVariant(item.status)}>{item.status}</Badge>
              </div>
            </TableCell>
            <TableCell>{item.preview.actionType}</TableCell>
            <TableCell>{item.executionMode}</TableCell>
            <TableCell className="max-w-[400px] whitespace-normal text-sm text-muted-foreground">
              {item.summary}
              {item.outputLines.length > 0 ? (
                <pre className="mt-2 max-h-32 overflow-auto rounded-xl border border-border/70 bg-background/70 p-3 text-xs leading-6 text-muted-foreground">
                  {item.outputLines.join("\n")}
                </pre>
              ) : null}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

import type { OcpLiveResourceSummary } from "@/domains/connection/types";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/shared/ui/table";

type ResourceListProps = {
  items: OcpLiveResourceSummary[];
  resource: string;
  namespace: string;
  selectedName?: string;
  onSelect: (item: OcpLiveResourceSummary) => void;
  emptyMessage?: string;
};

export function ResourceList({
  items,
  resource,
  namespace,
  selectedName = "",
  onSelect,
  emptyMessage = "현재 조건에 해당하는 리소스가 없습니다.",
}: ResourceListProps) {
  if (!items.length) {
    return (
      <div className="rounded-xl border border-border/70 bg-background/50 px-4 py-3 text-sm text-muted-foreground">
        {emptyMessage}
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Kind</TableHead>
          <TableHead>Namespace</TableHead>
          <TableHead>Status</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((item) => (
          <TableRow
            key={`${item.namespace}:${item.name}`}
            className={selectedName === item.name ? "bg-secondary/60" : ""}
            onClick={() => onSelect(item)}
            style={{ cursor: "pointer" }}
          >
            <TableCell className="font-medium">{item.name}</TableCell>
            <TableCell>{item.kind || resource}</TableCell>
            <TableCell>{item.namespace || namespace}</TableCell>
            <TableCell>{item.phase || item.type || item.host || item.to || item.clusterIp || item.nodeName || "-"}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}


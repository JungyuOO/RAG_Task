import { CheckCircle2, ChevronUp, CircleAlert, LogOut, PlugZap, RefreshCcw, Settings2 } from "lucide-react";

import { Button } from "@/shared/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/shared/ui/dropdown-menu";
import type { OcpConnectionController } from "@/domains/connection/useOcpConnection";

type ConnectionProfileMenuProps = {
  controller: OcpConnectionController;
  onOpenConnections: () => void;
};

function resolveDisplayName(controller: OcpConnectionController) {
  return (
    controller.testResult?.resolvedUser ||
    controller.profile?.usernameHint ||
    controller.profile?.displayName ||
    "No connection"
  );
}

function resolveStatusTone(controller: OcpConnectionController) {
  if (!controller.profile) return "idle";
  if (controller.testResult?.success) return "connected";
  if (controller.testResult && !controller.testResult.success) return "warning";
  return "idle";
}

export function ConnectionProfileMenu({
  controller,
  onOpenConnections,
}: ConnectionProfileMenuProps) {
  const displayName = resolveDisplayName(controller);
  const tone = resolveStatusTone(controller);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex w-full items-center justify-between rounded-xl border border-border/70 bg-[#15181e] px-3 py-2.5 text-left hover:bg-[#1a1d24]"
        >
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-foreground">{displayName}</div>
          </div>
          <div className="ml-3 flex items-center gap-2">
            {tone === "connected" ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            ) : tone === "warning" ? (
              <CircleAlert className="h-4 w-4 text-amber-400" />
            ) : (
              <PlugZap className="h-4 w-4 text-muted-foreground" />
            )}
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          </div>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="start" className="w-56">
        <DropdownMenuLabel>Connection</DropdownMenuLabel>
        <div className="px-2 pb-2 text-xs text-muted-foreground">
          {controller.profile?.displayName || controller.profile?.clusterUrl || "활성 연결이 없습니다."}
        </div>
        <DropdownMenuItem onSelect={onOpenConnections}>
          <Settings2 className="mr-2 h-4 w-4" />
          Manage connections
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          disabled={!controller.profile || controller.isBusy}
          onSelect={() => {
            void controller.runConnectionTest();
          }}
        >
          <PlugZap className="mr-2 h-4 w-4" />
          Test connection
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={!controller.profile || controller.isBusy}
          onSelect={() => {
            void controller.refreshLeaseMetadata();
          }}
        >
          <RefreshCcw className="mr-2 h-4 w-4" />
          Refresh lease
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          disabled={!controller.profile || controller.isBusy}
          onSelect={() => {
            void controller.clearConnection();
          }}
        >
          <LogOut className="mr-2 h-4 w-4" />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

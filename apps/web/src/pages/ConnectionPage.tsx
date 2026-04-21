import { PageHeader } from "@/shared/layout/PageHeader";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/shared/ui/card";
import { OcpConnectionForm } from "@/domains/connection/OcpConnectionForm";
import type { OcpConnectionController } from "@/domains/connection/useOcpConnection";
import type { WorkspaceRecord } from "@/domains/workspaces/types";

type ConnectionPageProps = {
  controller: OcpConnectionController;
  selectedWorkspace: WorkspaceRecord | null;
};

export function ConnectionPage({ controller, selectedWorkspace }: ConnectionPageProps) {
  return (
    <div className="mx-auto max-w-3xl space-y-8 py-4">
      <PageHeader
        eyebrow="OCP Operator Console"
        title="Connect your OpenShift cluster"
        description="URL과 자격 증명만 있으면 연결 상태, RBAC, secret backend, lease posture까지 한 번에 검증합니다. 연결이 끝나면 대시보드와 live operations 화면으로 바로 이어집니다."
        className="text-center md:block"
      />

      {selectedWorkspace ? (
        <div className="surface-muted rounded-2xl px-4 py-3 text-sm text-muted-foreground">
          Active workspace: <span className="font-medium text-foreground">{selectedWorkspace.name}</span>
        </div>
      ) : null}

      <Card className="surface-soft shadow-xl">
        <CardHeader className="space-y-2">
          <CardTitle>Cluster profile</CardTitle>
          <CardDescription>
            Server URL, 인증 방식, 기본 namespace를 입력하세요. 현재 선택된 workspace 기준으로 연결 프로필이 생성되고 재사용됩니다.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <OcpConnectionForm controller={controller} />
        </CardContent>
      </Card>
    </div>
  );
}



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

type ConnectionPageProps = {
  controller: OcpConnectionController;
};

export function ConnectionPage({ controller }: ConnectionPageProps) {
  return (
    <div className="mx-auto max-w-3xl space-y-8 py-4">
      <PageHeader
        eyebrow="OCP Operator Console"
        title="Connect your OpenShift cluster"
        description="URL과 자격 증명만 있으면 연결 상태, RBAC, secret backend, lease posture까지 한 번에 검증합니다. 연결이 끝나면 대시보드와 live operations 화면으로 바로 이어집니다."
        className="text-center md:block"
      />

      <Card className="border-border/70 bg-card/95 shadow-xl">
        <CardHeader className="space-y-2">
          <CardTitle>Cluster profile</CardTitle>
          <CardDescription>
            Server URL, 인증 방식, 기본 namespace를 입력하세요. 연결 후에는 같은 프로필을 재사용해 빠르게 다시 접속할 수 있습니다.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <OcpConnectionForm controller={controller} />
        </CardContent>
      </Card>
    </div>
  );
}



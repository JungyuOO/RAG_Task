from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECTION_INDEX = ROOT / "tests" / "data" / "section_index.json"
OUTPUT_PATH = ROOT / "tests" / "data" / "service_eval_dataset_v2.json"


# Each seed: (source_path, section_title, persona, scenario, question_turns)
# The question turns are crafted to target the specific section, NOT its confusable siblings.
SEEDS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "official/en/advanced_networking.md",
        "4.2. Enabling Stream Control Transmission Protocol (SCTP)",
        "인프라 엔지니어가 신규 기능 켜는 톤",
        "single_turn",
        ["sctp 프로토콜 처음 활성화하려면 어떻게 해?"],
    ),
    (
        "official/en/advanced_networking.md",
        "4.3. Verifying Stream Control Transmission Protocol (SCTP) is enabled",
        "운영 담당자가 설정 확인하는 톤",
        "followup",
        [
            "sctp 가 잘 켜져있는지 확인하는 방법",
            "확인용 명령만 짧게 정리해줘",
        ],
    ),
    (
        "official/en/authentication_and_authorization.md",
        "3.4. Configuring the internal OAuth server\u2019s token duration",
        "보안 담당자가 정책 바꾸는 톤",
        "single_turn",
        ["oauth 토큰 유효 기간 자체를 늘리려면 어디 설정해?"],
    ),
    (
        "official/en/authentication_and_authorization.md",
        "3.5. Configuring token inactivity timeout for the internal OAuth server",
        "보안 담당자가 유휴 세션 처리 확인하는 톤",
        "single_turn",
        ["oauth 토큰이 일정 시간 안 쓰면 자동으로 만료되게 하고 싶어"],
    ),
    (
        "official/en/authentication_and_authorization.md",
        "5.1. Listing user-owned OAuth access tokens",
        "운영 담당자가 감사용으로 조회하는 톤",
        "followup",
        [
            "유저가 가진 oauth 토큰 리스트 뽑는 방법",
            "출력에서 봐야 할 필드도 알려줘",
        ],
    ),
    (
        "official/en/authentication_and_authorization.md",
        "5.3. Deleting user-owned OAuth access tokens",
        "보안 담당자가 회수 처리하는 톤",
        "single_turn",
        ["특정 유저 oauth 토큰 삭제해서 세션 끊으려면?"],
    ),
    (
        "official/en/ingress_and_load_balancing.md",
        "2.6.2.1. Switching the Ingress Controller from using a Classic Load Balancer to a Network Load Balancer",
        "실무자가 마이그레이션 절차 확인하는 톤",
        "followup",
        [
            "classic load balancer 쓰던걸 network load balancer 로 바꾸고 싶은데",
            "절차만 번호로 정리해줘",
        ],
    ),
    (
        "official/en/ingress_and_load_balancing.md",
        "2.6.2.2. Switching the Ingress Controller from using a Network Load Balancer to a Classic Load Balancer",
        "실무자가 롤백 절차 묻는 톤",
        "single_turn",
        ["nlb 에서 다시 classic load balancer 로 되돌리려면 어떻게?"],
    ),
    (
        "official/en/ingress_and_load_balancing.md",
        "2.6.2.4. Configuring an Ingress Controller Network Load Balancer on an existing AWS cluster",
        "운영 담당자가 기존 클러스터에 적용하려는 톤",
        "single_turn",
        ["이미 올라가 있는 aws 클러스터에 ingress nlb 붙이려면?"],
    ),
    (
        "official/en/ingress_and_load_balancing.md",
        "2.6.3.1. Configuring an Ingress Controller Network Load Balancer on a new AWS cluster",
        "설치 담당자가 신규 설치 계획하는 톤",
        "single_turn",
        ["aws 클러스터 새로 깔 때 처음부터 nlb 로 ingress 구성하는 방법"],
    ),
    (
        "official/en/kubernetes_nmstate.md",
        "1.1. Viewing the network state of a node by using the CLI",
        "운영자가 터미널에서 확인하는 톤",
        "followup",
        [
            "노드 네트워크 상태 cli 로 보려면 무슨 명령어?",
            "출력 필드 의미도 짧게 알려줘",
        ],
    ),
    (
        "official/en/kubernetes_nmstate.md",
        "1.2. Viewing a graphical representation of the network state of a node (NNS) topology from the web console",
        "신규 사용자가 웹 UI 로 보는 톤",
        "single_turn",
        ["웹 콘솔에서 노드 네트워크 상태 그래픽으로 보는 방법"],
    ),
    (
        "official/en/networking_operators.md",
        "1.1.1. Installing the Kubernetes NMState Operator by using the web console",
        "초보 사용자가 UI 설치하는 톤",
        "single_turn",
        ["nmstate 오퍼레이터 웹 콘솔에서 설치하려면 순서가?"],
    ),
    (
        "official/en/networking_operators.md",
        "1.1.2. Installing the Kubernetes NMState Operator by using the CLI",
        "자동화 담당자가 스크립트 만드는 톤",
        "single_turn",
        ["nmstate 오퍼레이터 oc 명령으로 설치하는 절차 알려줘"],
    ),
    (
        "official/en/multiple_networks.md",
        "3.1.2. Benefits of a user-defined network",
        "기획자가 도입 효과 정리하는 톤",
        "single_turn",
        ["user-defined network 쓰면 뭐가 좋은지 장점만 알려줘"],
    ),
    (
        "official/en/multiple_networks.md",
        "3.1.3. Limitations of a user-defined network",
        "아키텍트가 제약 조건 파악하는 톤",
        "single_turn",
        ["user-defined network 도입 전에 알아둬야 할 제약사항이 뭐야?"],
    ),
    (
        "official/en/nodes.md",
        "1.2. About pods",
        "초보 사용자가 개념 묻는 톤",
        "single_turn",
        ["openshift 에서 파드가 정확히 뭔지 개념 설명해줘"],
    ),
    (
        "official/en/nodes.md",
        "1.4. About autoscaling pods on a node",
        "운영자가 오토스케일 이해하는 톤",
        "followup",
        [
            "노드 위에서 파드 오토스케일링 어떻게 동작해?",
            "HPA 랑 뭐가 달라?",
        ],
    ),
    (
        "official/en/storage.md",
        "2.3.1. Ephemeral storage limits and requests units",
        "개발자가 yaml 작성하는 톤",
        "single_turn",
        ["ephemeral storage 요청량 단위 어떻게 표기해야 되지?"],
    ),
    (
        "official/en/storage.md",
        "2.3.2. Ephemeral storage requests and limits example",
        "실무자가 예시 찾는 톤",
        "single_turn",
        ["ephemeral storage requests limits 설정한 yaml 예제 보여줘"],
    ),
    (
        "official/en/network_security.md",
        "Chapter 1. Understanding network policy APIs",
        "팀 리드가 개념 브리핑하는 톤",
        "single_turn",
        ["network policy api 전반 개요부터 설명해줘"],
    ),
    (
        "official/en/network_security.md",
        "Chapter 2. Admin network policy",
        "보안 담당자가 관리자용 정책 찾는 톤",
        "single_turn",
        ["admin network policy 가 일반 network policy 랑 어떻게 다른지"],
    ),
    (
        "official/en/security_and_compliance.md",
        "Chapter 1. OpenShift Container Platform security and compliance",
        "규정 담당자가 전체 보안 스택 묻는 톤",
        "single_turn",
        ["openshift 전반 보안 컴플라이언스 체계 한번에 정리해줘"],
    ),
    (
        "official/en/security_and_compliance.md",
        "Chapter 2. Container security",
        "컨테이너 보안 엔지니어 톤",
        "single_turn",
        ["컨테이너 레벨 보안 고려사항만 따로 알려줘"],
    ),
    (
        "official/en/scalability_and_performance.md",
        "2.1. Recommended control plane practices",
        "아키텍트가 컨트롤 플레인 설계하는 톤",
        "single_turn",
        ["control plane 운영할 때 권장되는 practice 가 뭐야?"],
    ),
    (
        "official/en/scalability_and_performance.md",
        "2.2. Selecting a larger AWS instance type for control plane machines",
        "운영자가 aws 인스턴스 타입 바꾸려는 톤",
        "followup",
        [
            "aws 에서 control plane 머신 인스턴스 타입 더 큰걸로 바꾸려면",
            "고려할 제약사항도 알려줘",
        ],
    ),
    (
        "official/en/etcd.md",
        "1.1. How etcd works",
        "운영자가 etcd 동작 이해하는 톤",
        "single_turn",
        ["etcd 가 내부적으로 어떻게 동작하는지 원리"],
    ),
    (
        "official/en/gitops.md",
        "Chapter 1. About Red Hat OpenShift GitOps",
        "신규 도입 검토자 톤",
        "single_turn",
        ["openshift gitops 가 뭐고 뭘 할 수 있는지 처음부터"],
    ),
    (
        "official/en/building_applications.md",
        "1.1. Working on a project",
        "초보 개발자가 onboarding 톤",
        "single_turn",
        ["openshift 에서 프로젝트 단위로 작업한다는게 무슨 의미야"],
    ),
    (
        "official/en/builds_using_buildconfig.md",
        "Chapter 1. Understanding image builds",
        "초보 개발자가 빌드 개념 잡는 톤",
        "single_turn",
        ["openshift image build 가 어떤 방식으로 돌아가는지"],
    ),
    (
        "official/en/builds_using_buildconfig.md",
        "Chapter 2. Understanding build configurations",
        "실무자가 설정 구조 파악하는 톤",
        "single_turn",
        ["buildconfig 리소스가 뭘 정의하는지"],
    ),
    (
        "official/en/machine_configuration.md",
        "1.1. About the Machine Config Operator",
        "클러스터 관리자 톤",
        "single_turn",
        ["machine config operator 역할 간단히"],
    ),
    (
        "official/en/machine_management.md",
        "1.1. Machine API overview",
        "초보 관리자 톤",
        "single_turn",
        ["openshift machine api 전반 개요 설명"],
    ),
    (
        "official/en/jenkins.md",
        "1.3. Providing Jenkins cross project access",
        "CI 담당자 톤",
        "single_turn",
        ["jenkins 가 다른 프로젝트 리소스에 접근하게 하려면"],
    ),
    (
        "official/en/jenkins.md",
        "1.4. Jenkins cross volume mount points",
        "CI 담당자가 볼륨 설정 묻는 톤",
        "single_turn",
        ["jenkins 파드가 여러 볼륨 마운트 잡는 지점 어디서 설정해?"],
    ),
    (
        "official/en/registry.md",
        "1.2. Integrated OpenShift image registry",
        "이미지 저장소 관리자 톤",
        "single_turn",
        ["openshift 내장 레지스트리가 따로 있는지, 있다면 뭐가 특별한지"],
    ),
    (
        "official/en/configuring_network_settings.md",
        "Chapter 2. Configuring the node port service range",
        "네트워크 관리자 톤",
        "single_turn",
        ["nodeport 로 쓸 수 있는 포트 범위 바꾸려면?"],
    ),
    (
        "official/en/configuring_network_settings.md",
        "Chapter 3. Configuring the cluster network range",
        "네트워크 관리자 톤",
        "single_turn",
        ["클러스터 내부 네트워크 cidr 범위 바꾸는 방법"],
    ),
    (
        "official/en/ovn-kubernetes_network_plugin.md",
        "1.1. OVN-Kubernetes purpose",
        "초보 사용자가 용도 묻는 톤",
        "single_turn",
        ["ovn-kubernetes 플러그인 왜 쓰는건지 용도"],
    ),
    (
        "official/en/pipelines.md",
        "Chapter 1. About Red Hat OpenShift Pipelines",
        "ci/cd 엔지니어 톤",
        "single_turn",
        ["openshift pipelines 가 tekton 이랑 무슨 관계인지"],
    ),
    (
        "official/en/updating_clusters.md",
        "1.1. Introduction to OpenShift updates",
        "운영자 톤",
        "single_turn",
        ["openshift 클러스터 업데이트 큰 흐름이 어떻게 되는지"],
    ),
]


def main() -> None:
    index = json.loads(SECTION_INDEX.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], dict] = {
        (item["source_path"], item["section_title"]): item for item in index
    }

    dataset: list[dict] = []
    missing: list[tuple[str, str]] = []
    for order, (source_path, section_title, persona, scenario, turns) in enumerate(SEEDS, start=1):
        entry = lookup.get((source_path, section_title))
        if entry is None:
            missing.append((source_path, section_title))
            continue
        dataset.append(
            {
                "id": f"eval-v2-{order:04d}",
                "group": "official",
                "persona": persona,
                "scenario": scenario,
                "source_path": source_path,
                "section_title": section_title,
                "turns": turns,
                "question": turns[0],
                "snippet": entry["body_preview"],
            }
        )

    if missing:
        print("MISSING:", json.dumps(missing, ensure_ascii=False))
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(dataset), "output": str(OUTPUT_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

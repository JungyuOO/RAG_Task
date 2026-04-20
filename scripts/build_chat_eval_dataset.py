from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUTPUT_PATH = ROOT / "tests" / "data" / "chat_eval_dataset_v1.json"
API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000")


@dataclass(slots=True)
class LiveSelection:
    namespace: str
    deployment_name: str
    deployment_replicas: int
    pod_name: str


def load_env() -> dict[str, str]:
    data: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def api_get(path: str) -> Any:
    response = httpx.get(f"{API_BASE}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any], timeout: float = 120) -> Any:
    response = httpx.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_connection(env: dict[str, str]) -> str:
    payload = {
        "cluster_url": env["OCP_API_BASE_URL"],
        "auth_mode": "token",
        "verify_ssl": False,
        "default_namespace": env.get("OCP_DEFAULT_NAMESPACE", "demo"),
        "display_name": "chat-eval-runner",
        "save_profile": False,
        "token": env["OCP_API_TOKEN"],
        "username": "",
        "password": None,
    }
    data = api_post("/api/v1/auth/ocp/connect", payload, timeout=60)
    connection_id = str(data["connection"]["connection_id"])
    api_post("/api/v1/auth/ocp/test", {"connection_id": connection_id}, timeout=120)
    return connection_id


def api_get_retry(path: str, *, attempts: int = 4, delay_sec: float = 2.0) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return api_get(path)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(delay_sec)
    assert last_error is not None
    raise last_error


def pick_live_targets(connection_id: str, namespace: str) -> LiveSelection:
    try:
        deployments = api_get_retry(
            f"/api/v1/ocp/resources/{connection_id}?resource=deployments&namespace={namespace}",
            attempts=6,
            delay_sec=3.0,
        )
        pods = api_get_retry(
            f"/api/v1/ocp/resources/{connection_id}?resource=pods&namespace={namespace}",
            attempts=6,
            delay_sec=3.0,
        )
    except Exception:
        previous = _load_previous_selection()
        if previous is not None:
            return previous
        raise

    deployment_items = [
        item for item in deployments["items"]
        if not str(item["name"]).startswith("build-and-push")
    ]
    pod_items = [
        item for item in pods["items"]
        if str(item["name"]).startswith("demo-app-")
    ]

    if not deployment_items or not pod_items:
        raise RuntimeError("Unable to select stable live OCP targets for scenario generation.")

    deployment = deployment_items[0]
    deployment_detail = api_get(
        f"/api/v1/ocp/resource-detail/{connection_id}?resource=deployments&namespace={namespace}&name={deployment['name']}"
    )
    spec = deployment_detail.get("manifest_json", {}).get("spec", {}) or {}

    return LiveSelection(
        namespace=namespace,
        deployment_name=str(deployment["name"]),
        deployment_replicas=int(spec.get("replicas") or 0),
        pod_name=str(pod_items[0]["name"]),
    )


def _load_previous_selection() -> LiveSelection | None:
    if not OUTPUT_PATH.exists():
        return None
    try:
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    selection = payload.get("live_selection") or {}
    namespace = str(selection.get("namespace") or "").strip()
    deployment_name = str(selection.get("deployment_name") or "").strip()
    pod_name = str(selection.get("pod_name") or "").strip()
    replicas = int(selection.get("deployment_replicas") or 0)
    if not namespace or not deployment_name or not pod_name:
        return None
    return LiveSelection(
        namespace=namespace,
        deployment_name=deployment_name,
        deployment_replicas=replicas,
        pod_name=pod_name,
    )


def doc_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "id": "doc-route-basics-01",
            "family": "doc_route",
            "category": "doc",
            "description": "Route 기본 개념과 운영 포인트",
            "turns": [
                {"step_type": "chat", "message": "OpenShift Route가 무엇인지 설명해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["Route", "라우트", "Ingress"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "Service와 Route 차이를 공식 문서 기준으로 설명해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["Service", "Route", "서비스", "라우트"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "그러면 Route를 만드는 기본 방법도 알려줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["Route", "Ingress", "oc create", "oc expose"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "reencrypt route는 언제 쓰는지 설명해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["reencrypt", "route", "라우트"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "지금까지 내용을 운영 체크포인트 3개로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["운영", "정리", "Route", "라우트"], "source_path_tokens": ["ingress", "network"]}},
            ],
        },
        {
            "id": "doc-route-ops-02",
            "family": "doc_route",
            "category": "doc",
            "description": "Route 생성과 TLS 관련 운영 질문",
            "turns": [
                {"step_type": "chat", "message": "공식 문서 기준으로 Route를 oc expose로 만드는 방법을 알려줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["Route", "oc expose", "service"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "Ingress object로 route를 만드는 방식도 설명해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["Ingress", "route", "oc create"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "edge route와 reencrypt route 차이를 설명해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["edge", "reencrypt", "route"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "destination ca certificate secret이 언제 필요한지도 설명해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["destination", "ca", "secret", "reencrypt"], "source_path_tokens": ["ingress", "network"]}},
                {"step_type": "chat", "message": "운영자가 Route TLS 설정 볼 때 체크할 항목을 4개로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["TLS", "Route", "운영", "정리"], "source_path_tokens": ["ingress", "network"]}},
            ],
        },
        {
            "id": "doc-auth-rbac-03",
            "family": "doc_auth",
            "category": "doc",
            "description": "authentication, authorization, RBAC 기본 흐름",
            "turns": [
                {"step_type": "chat", "message": "OpenShift에서 authentication과 authorization 차이를 설명해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["authentication", "authorization", "인증", "권한"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "인증되지 않은 그룹을 클러스터 역할에 바인딩하는 절차를 알려줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["ClusterRole", "ClusterRoleBinding", "클러스터 역할", "바인딩"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "권한 문제를 볼 때는 어떤 Role과 Binding을 같이 봐야 하는지 정리해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["RoleBinding", "ClusterRole", "권한", "role"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "방금 설명한 RBAC 기준으로 클러스터 역할, 로컬 역할, 바인딩, 사용자/그룹/서비스계정 관계를 설명해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["클러스터", "로컬", "바인딩", "사용자", "그룹", "서비스 계정"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "방금 설명한 RBAC 핵심 객체를 4개로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["RBAC", "role", "binding", "ClusterRole", "RoleBinding"], "source_path_tokens": ["authentication", "authorization"]}},
            ],
        },
        {
            "id": "doc-auth-rbac-04",
            "family": "doc_auth",
            "category": "doc",
            "description": "RBAC 객체 관계와 점검 포인트",
            "turns": [
                {"step_type": "chat", "message": "RBAC overview를 기준으로 핵심 객체가 무엇인지 설명해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["RBAC", "role", "binding"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "클러스터 역할 바인딩과 로컬 역할 바인딩 차이를 설명해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["클러스터", "로컬", "바인딩"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "사용자, 그룹, 서비스 계정이 RBAC에 어떻게 연결되는지도 설명해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["사용자", "그룹", "서비스 계정", "RBAC"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "권한 문제 분석 시 먼저 볼 객체를 우선순위로 정리해줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["권한", "Role", "Binding", "정리"], "source_path_tokens": ["authentication", "authorization"]}},
                {"step_type": "chat", "message": "운영자용 RBAC 점검 체크리스트 4개로 마무리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["RBAC", "체크리스트", "운영", "정리"], "source_path_tokens": ["authentication", "authorization"]}},
            ],
        },
        {
            "id": "doc-deployment-rollout-05",
            "family": "doc_deployment",
            "category": "doc",
            "description": "deployment rollout 관련 공식 문서 질문",
            "turns": [
                {"step_type": "chat", "message": "OpenShift에서 deployment rollout history를 확인하는 방법을 알려줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["deployment", "rollout", "history", "oc"], "source_path_tokens": ["applications"]}},
                {"step_type": "chat", "message": "deployment 상태를 점검할 때 어떤 명령이나 정보를 봐야 하는지도 알려줘.", "expect": {"lane_any": ["doc"], "min_doc_citations": 1, "answer_any_keywords": ["deployment", "상태", "rollout", "history"]}},
                {"step_type": "chat", "message": "replica 수와 rollout 상태를 함께 점검하는 관점으로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["replica", "rollout", "정리", "deployment"]}},
                {"step_type": "chat", "message": "운영자가 배포 이력 확인할 때 놓치면 안 되는 포인트도 설명해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["운영", "배포", "이력", "포인트"]}},
                {"step_type": "chat", "message": "마지막으로 rollout 점검 체크리스트 4개로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": ["rollout", "체크리스트", "정리"]}},
            ],
        },
    ]


def live_scenarios(namespace: str, deployment: str, pod: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "live-deployment-yaml-01",
            "family": "live_deployment",
            "category": "live",
            "description": "deployment list -> yaml -> pod list -> pod yaml",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest", "Deployment"]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{pod} pod yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest", "Pod"]}},
            ],
        },
        {
            "id": "live-deployment-yaml-02",
            "family": "live_deployment",
            "category": "live",
            "description": "pod list -> pod yaml -> deployment yaml",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{pod} pod yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest", "Pod"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
            ],
        },
        {
            "id": "live-deployment-yaml-03",
            "family": "live_deployment",
            "category": "live",
            "description": "deployment list 반복 확인 후 pod yaml",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{pod} pod yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest"]}},
            ],
        },
        {
            "id": "live-deployment-yaml-04",
            "family": "live_deployment",
            "category": "live",
            "description": "deployment yaml과 pod yaml 교차 확인",
            "turns": [
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{pod} pod yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
            ],
        },
        {
            "id": "live-deployment-yaml-05",
            "family": "live_deployment",
            "category": "live",
            "description": "pod 중심 live 확인",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{pod} pod yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest"]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{pod} pod yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest"]}},
            ],
        },
    ]


def mixed_scenarios(namespace: str, deployment: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "mixed-deployment-doc-01",
            "family": "mixed_deployment",
            "category": "mixed",
            "description": "live deployment 후 rollout history 문서 연결",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"방금 본 {deployment} deployment 기준으로 공식 문서에서 rollout history 확인 방법을 알려줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "rollout", "history", "문서"]}},
                {"step_type": "chat", "message": f"{deployment} deployment 기준으로 replica와 rollout 확인 포인트를 문서 기준으로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "replica", "rollout", "정리"]}},
                {"step_type": "chat", "message": f"방금 문서 기준 내용을 바탕으로 {deployment} deployment 점검 체크리스트 4개로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "deployment", "체크리스트", "정리"]}},
            ],
        },
        {
            "id": "mixed-deployment-doc-02",
            "family": "mixed_deployment",
            "category": "mixed",
            "description": "live deployment 후 운영 점검 포인트 문서 연결",
            "turns": [
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"방금 본 {deployment} deployment 기준으로 운영자가 봐야 하는 문서 체크포인트를 알려줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "운영", "문서", "체크포인트"]}},
                {"step_type": "chat", "message": f"{deployment} deployment 기준으로 rollout 관련 명령이나 확인 방법을 문서 기준으로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "rollout", "정리", "문서"]}},
                {"step_type": "chat", "message": f"{deployment} deployment 운영 요약을 4개 항목으로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "운영", "정리"]}},
            ],
        },
        {
            "id": "mixed-deployment-doc-03",
            "family": "mixed_deployment",
            "category": "mixed",
            "description": "deployment yaml 후 replica 관점 문서 연결",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"방금 본 {deployment} deployment 기준으로 rollout history 확인 포인트를 공식 문서 기준으로 설명해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "rollout", "history", "문서"]}},
                {"step_type": "chat", "message": f"{deployment} deployment를 운영할 때 rollout history를 확인하는 이유를 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "rollout", "history", "정리"]}},
                {"step_type": "chat", "message": f"{deployment} deployment 문서 요약을 체크리스트 4개로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "문서", "체크리스트", "정리"]}},
            ],
        },
        {
            "id": "mixed-deployment-doc-04",
            "family": "mixed_deployment",
            "category": "mixed",
            "description": "deployment list 후 rollout 상태 문서 연결",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"공식 문서 기준으로 {deployment} 같은 deployment의 rollout 상태를 어떻게 확인하는지 알려줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "rollout", "상태", "문서"]}},
                {"step_type": "chat", "message": f"{deployment} deployment 운영 시 이력 확인 포인트를 문서 기준으로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "이력", "문서", "정리"]}},
                {"step_type": "chat", "message": f"{deployment} deployment 관련 공식 문서 체크리스트를 4개로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "공식", "체크리스트", "정리"]}},
            ],
        },
        {
            "id": "mixed-deployment-doc-05",
            "family": "mixed_deployment",
            "category": "mixed",
            "description": "deployment yaml 후 운영 체크리스트 문서 연결",
            "turns": [
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"방금 본 {deployment} deployment를 기준으로 운영 체크리스트를 공식 문서 기준으로 설명해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "운영", "체크리스트", "문서"]}},
                {"step_type": "chat", "message": f"{deployment} deployment에서 replica와 rollout을 확인해야 하는 이유를 문서 근거로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "replica", "rollout", "정리"]}},
                {"step_type": "chat", "message": f"{deployment} deployment 운영 요약을 문서 기반 4개 항목으로 정리해줘.", "expect": {"lane_any": ["doc", "mixed"], "min_doc_citations": 1, "answer_any_keywords": [deployment, "운영", "문서", "정리"]}},
            ],
        },
    ]


def edit_scenarios(namespace: str, deployment: str, pod: str, base_replicas: int) -> list[dict[str, Any]]:
    target_replicas = base_replicas + 1
    variants = [
        {
            "id": "edit-scale-followup-01",
            "description": f"deployment replica를 {target_replicas}로 수정하고 YAML 확인",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "yaml_apply", "target_resource": "deployments", "target_name": deployment, "target_namespace": namespace, "replicas_before": base_replicas, "replicas_after": target_replicas},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
            ],
        },
        {
            "id": "edit-scale-followup-02",
            "description": f"deployment replica를 {target_replicas}로 수정하고 목록 재확인",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "yaml_apply", "target_resource": "deployments", "target_name": deployment, "target_namespace": namespace, "replicas_before": base_replicas, "replicas_after": target_replicas},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
            ],
        },
        {
            "id": "edit-scale-followup-03",
            "description": f"deployment replica를 {target_replicas}로 수정하고 pod 관점 확인",
            "turns": [
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "yaml_apply", "target_resource": "deployments", "target_name": deployment, "target_namespace": namespace, "replicas_before": base_replicas, "replicas_after": target_replicas},
                {"step_type": "chat", "message": f"{pod} pod yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest"]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
            ],
        },
        {
            "id": "edit-scale-followup-04",
            "description": f"deployment replica를 {target_replicas}로 수정하고 deployment/pod 교차 확인",
            "turns": [
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "yaml_apply", "target_resource": "deployments", "target_name": deployment, "target_namespace": namespace, "replicas_before": base_replicas, "replicas_after": target_replicas},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{pod} pod yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [pod], "answer_any_keywords": [pod, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
            ],
        },
        {
            "id": "edit-scale-followup-05",
            "description": f"deployment replica를 {target_replicas}로 수정하고 최종 YAML 재점검",
            "turns": [
                {"step_type": "chat", "message": f"{deployment} deployment yaml 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 deployment 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [deployment]}},
                {"step_type": "yaml_apply", "target_resource": "deployments", "target_name": deployment, "target_namespace": namespace, "replicas_before": base_replicas, "replicas_after": target_replicas},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 다시 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment], "answer_any_keywords": [deployment, "manifest"]}},
                {"step_type": "chat", "message": f"{namespace} namespace의 pod 목록을 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_list"], "live_resource_names": [pod]}},
                {"step_type": "chat", "message": f"{deployment} deployment yaml 마지막으로 한 번 더 보여줘.", "expect": {"lane_any": ["live"], "artifact_types": ["resource_editor"], "live_resource_names": [deployment]}},
            ],
        },
    ]
    return [
        {
            "family": "edit_followup",
            "category": "edit_followup",
            **variant,
        }
        for variant in variants
    ]


def build_dataset(connection_id: str, selection: LiveSelection) -> dict[str, Any]:
    namespace = selection.namespace
    deployment = selection.deployment_name
    pod = selection.pod_name

    scenarios = [
        *doc_scenarios(),
        *live_scenarios(namespace, deployment, pod),
        *mixed_scenarios(namespace, deployment),
        *edit_scenarios(namespace, deployment, pod, selection.deployment_replicas),
    ]

    return {
        "version": "2026-04-20",
        "generated_by": "scripts/build_chat_eval_dataset.py",
        "api_base": API_BASE,
        "connection_id": connection_id,
        "connection_profile": {
            "default_namespace": namespace,
            "verify_ssl": False,
            "uses_env_token": True,
        },
        "live_selection": {
            "namespace": namespace,
            "deployment_name": deployment,
            "deployment_replicas": selection.deployment_replicas,
            "pod_name": pod,
        },
        "scenarios": scenarios,
    }


def main() -> int:
    env = load_env()
    namespace = env.get("OCP_DEFAULT_NAMESPACE", "demo")
    connection_id = create_connection(env)
    selection = pick_live_targets(connection_id, namespace)
    dataset = build_dataset(connection_id, selection)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUTPUT_PATH}")
    print(
        json.dumps(
            {
                "namespace": selection.namespace,
                "deployment": selection.deployment_name,
                "replicas": selection.deployment_replicas,
                "pod": selection.pod_name,
                "scenario_count": len(dataset["scenarios"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import re
from pathlib import Path


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_-]+")


def tokenize_query(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(str(text or "").casefold())
        if len(token) >= 2
    ]


def answer_source_budget(message: str) -> int:
    text = str(message or "").strip()
    tokens = tokenize_query(text)
    if not tokens:
        return 2

    clause_parts = [
        part.strip()
        for part in re.split(r"[\n\r]+|[.!?;]+|,+", text)
        if part.strip()
    ]
    clause_count = max(len(clause_parts), 1)
    complexity = len(tokens) + min(clause_count, 3) + len(set(tokens[:10])) * 0.25
    return 3 if complexity >= 8.5 else 2


def preferred_source_paths_for_query(message: str) -> list[str]:
    lowered = str(message or "").casefold()
    rel_paths: list[str] = []

    if any(marker in lowered for marker in ("gitops", "git ops", "argocd", "아르고", "깃옵스")):
        rel_paths.append("official/en/gitops.md")

    if any(
        marker in lowered
        for marker in (
            "oauth",
            "rbac",
            "authentication",
            "authorization",
            "인증",
            "인가",
            "권한",
            "clusterrole",
            "rolebinding",
            "clusterrolebinding",
            "serviceaccount",
            "service account",
        )
    ):
        rel_paths.append("official/en/authentication_and_authorization.md")

    if any(marker in lowered for marker in ("jenkins", "cross project", "cross-project", "cross volume")):
        rel_paths.append("official/en/jenkins.md")

    if any(marker in lowered for marker in ("machine api", "머신 api", "머신api", "machine management")):
        rel_paths.append("official/en/machine_management.md")

    if any(marker in lowered for marker in ("route", "routes", "ingress", "라우트", "load balancer", "nlb", "clb")):
        rel_paths.append("official/en/ingress_and_load_balancing.md")

    if any(
        marker in lowered
        for marker in (
            "deployment",
            "deployments",
            "rollout",
            "replica",
            "deploymentconfig",
            "deployment config",
            "배포",
            "롤아웃",
            "리플리카",
        )
    ):
        rel_paths.append("official/en/building_applications.md")

    if any(marker in lowered for marker in ("pod", "pods", "파드", "node", "nodes", "노드")):
        rel_paths.append("official/en/nodes.md")

    if any(marker in lowered for marker in ("pvc", "persistent volume claim", "storageclass", "storage class", "스토리지")):
        rel_paths.append("official/en/storage.md")

    if any(marker in lowered for marker in ("etcd", "backup", "restore", "snapshot", "백업", "복구", "스냅샷")):
        rel_paths.extend(
            [
                "official/en/etcd.md",
                "official/en/backup_and_restore.md",
            ]
        )

    root = Path.cwd() / "data" / "corpus" / "pdfs"
    resolved: list[str] = []
    seen: set[str] = set()
    for rel_path in rel_paths:
        if rel_path in seen:
            continue
        seen.add(rel_path)
        resolved.append(str((root / rel_path).resolve()))
    return resolved

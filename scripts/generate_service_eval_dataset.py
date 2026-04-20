from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "data" / "corpus" / "pdfs" / "official" / "en"
OUTPUT_PATH = ROOT / "tests" / "data" / "service_eval_dataset_v1.json"


@dataclass(slots=True)
class CuratedCaseSpec:
    source_path: str
    section_hint: str
    persona: str
    scenario: str
    turns: tuple[str, ...]


CURATED_CASES: tuple[CuratedCaseSpec, ...] = (
    CuratedCaseSpec(
        source_path="official/en/nodes.md",
        section_hint="2.2. Viewing pods",
        persona="운영 담당자가 급하게 묻는 톤",
        scenario="followup",
        turns=(
            "파드 확인하는 명령어 뭐 써?",
            "확인할 때 쓰는 명령어도 같이 알려줘",
            "관련 리소스 이름도 같이 알려줘",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/nodes.md",
        section_hint="4.3. Configuring a pod anti-affinity rule",
        persona="실무자가 바로 적용하려는 톤",
        scenario="deep_followup",
        turns=(
            "파드 생성 예시 yaml 보여줘",
            "필수 필드만 짧게 정리해줘",
            "적용 전에 체크할 포인트도 알려줘",
            "관련 리소스 이름도 같이 알려줘",
            "실무 기준으로 제일 중요한 것만 마지막으로 말해줘",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/ingress_and_load_balancing.md",
        section_hint="1.1. Creating basic routes",
        persona="초보 사용자가 쉽게 묻는 톤",
        scenario="followup",
        turns=(
            "route 예시 yaml 있어?",
            "필수 필드만 짧게 정리해줘",
            "확인할 때 어떤 리소스를 같이 봐야 해?",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/authentication_and_authorization.md",
        section_hint="3.8. Troubleshooting OAuth API events",
        persona="문제 해결 중 확인하려는 톤",
        scenario="followup",
        turns=(
            "oauth 문제나면 어디부터 봐야 해?",
            "관련 명령어만 더 짧게 알려줘",
            "확인할 때 보는 리소스 이름도 같이 알려줘",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/authentication_and_authorization.md",
        section_hint="5.4. Adding unauthenticated groups to cluster roles",
        persona="문서 읽다가 이어서 묻는 톤",
        scenario="followup",
        turns=(
            "rbac 확인할 때 뭘 봐야 해?",
            "관련 명령어만 더 짧게 알려줘",
            "실제 적용 순서만 다시 정리해줘",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/storage.md",
        section_hint="3.4. Persistent volume claims",
        persona="실무자가 바로 적용하려는 톤",
        scenario="followup",
        turns=(
            "pvc 예시 yaml 있어?",
            "필수 필드만 짧게 정리해줘",
            "적용 순서만 다시 짧게 정리해줘",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/etcd.md",
        section_hint="Chapter 1. Overview of etcd",
        persona="초보 사용자가 쉽게 묻는 톤",
        scenario="single_turn",
        turns=(
            "etcd가 뭐야?",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/backup_and_restore.md",
        section_hint="6.1. Backing up etcd",
        persona="운영 담당자가 급하게 묻는 톤",
        scenario="deep_followup",
        turns=(
            "etcd 백업이나 복구 전에 뭘 봐야 해?",
            "복구 전에 먼저 뭘 봐야 해?",
            "관련 명령어만 더 짧게 알려줘",
            "체크리스트만 짧게 정리해줘",
            "정리하면 etcd 쪽은 뭐부터 보면 돼?",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/gitops.md",
        section_hint="Chapter 1. About Red Hat OpenShift GitOps",
        persona="문서 읽다가 이어서 묻는 톤",
        scenario="followup",
        turns=(
            "gitops 적용 순서만 간단히 알려줘",
            "관련 리소스 이름도 같이 알려줘",
            "운영할 때 주의할 점도 알려줘",
        ),
    ),
    CuratedCaseSpec(
        source_path="official/en/advanced_networking.md",
        section_hint="Chapter 2. Changing the MTU for the cluster network",
        persona="문제 해결 중 확인하려는 톤",
        scenario="followup",
        turns=(
            "mtu 바꿀 때 어떤 순서로 봐야 해?",
            "적용 순서만 다시 짧게 정리해줘",
            "확인할 때 어떤 리소스를 같이 봐야 해?",
        ),
    ),
)


def clean_text(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_section(path: Path, heading_hint: str) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = -1

    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("#") and heading_hint.casefold() in stripped.casefold():
            start = index
            break

    if start == -1:
        raise ValueError(f"Heading not found: {heading_hint} in {path}")

    title = lines[start].strip().lstrip("#").strip()
    body_lines: list[str] = []
    start_level = len(lines[start]) - len(lines[start].lstrip("#"))

    for raw in lines[start + 1 :]:
        stripped = raw.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if level <= start_level:
                break
        if stripped.startswith("```"):
            continue
        cleaned = clean_text(stripped.lstrip("-*0123456789. ").strip())
        if cleaned:
            body_lines.append(cleaned)

    snippet = clean_text(" ".join(body_lines[:12]))
    if len(snippet) < 60:
        raise ValueError(f"Snippet too short for {heading_hint} in {path}")
    return title, snippet[:420]


def generate() -> list[dict]:
    items: list[dict] = []
    for index, spec in enumerate(CURATED_CASES, start=1):
        path = ROOT / "data" / "corpus" / "pdfs" / spec.source_path
        title, snippet = extract_section(path, spec.section_hint)
        items.append(
            {
                "id": f"eval-{index:04d}",
                "group": "official",
                "persona": spec.persona,
                "scenario": spec.scenario,
                "source_path": spec.source_path,
                "section_title": title,
                "turns": list(spec.turns),
                "question": spec.turns[0],
                "snippet": snippet,
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    items = generate()[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated {len(items)} items -> {output}")


if __name__ == "__main__":
    main()

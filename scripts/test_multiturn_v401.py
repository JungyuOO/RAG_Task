from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def load_scenarios(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_group(source_path: str) -> str:
    normalized = source_path.replace("\\", "/")
    if "/generated_pdf/" in normalized or "/generated/" in normalized:
        return "customer_generated"
    return "official_ocp"


def source_version(source_path: str) -> str | None:
    match = re.search(r"ocp-(\d+\.\d+)", source_path, re.IGNORECASE)
    return match.group(1) if match else None


def stream_chat(base_url: str, session_id: str, message: str, version_tag: str | None) -> list[dict]:
    payload = {"session_id": session_id, "message": message}
    if version_tag:
        payload["version_tag"] = version_tag
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-Id": "test-v401-multiturn-client",
        },
        method="POST",
    )
    events: list[dict] = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_bytes in resp:
            raw = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not raw.startswith("data: "):
                continue
            try:
                events.append(json.loads(raw[6:]))
            except json.JSONDecodeError:
                continue
    return events


def run_turn(base_url: str, session_id: str, turn_idx: int, question: str, version_tag: str | None) -> dict:
    print(f"\n{'=' * 90}")
    print(f"[Turn {turn_idx}] {question}")
    answer_parts: list[str] = []
    replaced_answer: str | None = None
    context_event: dict = {}
    status_events: list[dict] = []
    done = False
    started = time.time()

    for event in stream_chat(base_url, session_id, question, version_tag):
        event_type = event.get("type")
        if event_type == "token":
            answer_parts.append(event.get("content", ""))
        elif event_type == "replace_answer":
            replaced_answer = event.get("content", "")
        elif event_type == "context":
            context_event = event
        elif event_type == "status":
            status_events.append({"stage": event.get("stage"), "message": event.get("message")})
        elif event_type == "done":
            done = True

    answer = replaced_answer if replaced_answer is not None else "".join(answer_parts).strip()
    elapsed = round(time.time() - started, 2)
    retrieved_sources = []
    for item in context_event.get("items", [])[:10]:
        path = str(item.get("source_path") or "")
        retrieved_sources.append(
            {
                "file_name": Path(path).name,
                "source_path": path,
                "group": source_group(path),
                "version_tag": source_version(path),
                "page_number": item.get("page_number"),
                "rerank_score": item.get("rerank_score"),
            }
        )
    answer_citations = [
        {
            "file_name": citation.get("file_name"),
            "page_number": citation.get("page_number"),
            "score": citation.get("score"),
        }
        for citation in context_event.get("answer_citations", [])
    ]
    top_source = retrieved_sources[0] if retrieved_sources else {}
    version_mismatch = False
    if version_tag and retrieved_sources:
        top_versions = [item["version_tag"] for item in retrieved_sources[:5] if item.get("version_tag")]
        version_mismatch = bool(top_versions) and version_tag not in top_versions

    print(f"[elapsed] {elapsed}s")
    if top_source:
        print(
            "[top-source]",
            top_source.get("file_name"),
            top_source.get("group"),
            top_source.get("version_tag"),
            top_source.get("page_number"),
        )
    print(f"[citations] {len(answer_citations)}")
    print(f"A: {answer[:500]}{'...' if len(answer) > 500 else ''}")

    return {
        "turn": turn_idx,
        "question": question,
        "answer": answer,
        "elapsed_sec": elapsed,
        "done": done,
        "status_events": status_events,
        "top_score": context_event.get("top_score"),
        "mode": context_event.get("mode"),
        "retrieved_sources": retrieved_sources,
        "answer_citations": answer_citations,
        "version_mismatch": version_mismatch,
        "llm_generation_failed": "LLM 응답 생성에 실패했습니다" in answer,
    }


def group_check(expected_group: str, turns: list[dict]) -> tuple[bool, dict]:
    top_groups = [
        turn["retrieved_sources"][0]["group"]
        for turn in turns
        if turn.get("retrieved_sources")
    ]
    seen_groups = sorted(
        {
            item["group"]
            for turn in turns
            for item in turn.get("retrieved_sources", [])
        }
    )
    if expected_group == "mixed":
        passed = {"official_ocp", "customer_generated"}.issubset(set(seen_groups))
        return passed, {"top_groups": top_groups, "seen_groups": seen_groups}

    if not top_groups:
        return False, {"top_groups": [], "seen_groups": seen_groups}
    matched = sum(1 for group in top_groups if group == expected_group)
    ratio = matched / max(len(top_groups), 1)
    return ratio >= 0.7, {"top_groups": top_groups, "seen_groups": seen_groups, "match_ratio": round(ratio, 2)}


def source_token_check(expected_source_tokens: list[str], turns: list[dict]) -> tuple[bool, dict]:
    files = [
        item["file_name"]
        for turn in turns
        for item in turn.get("retrieved_sources", [])
    ]
    lowered_files = [file_name.casefold() for file_name in files]
    lowered_tokens = [token.casefold() for token in expected_source_tokens]
    matched_tokens = [
        token for token in lowered_tokens
        if any(token in file_name for file_name in lowered_files)
    ]
    return bool(matched_tokens), {"matched_tokens": matched_tokens, "observed_files": files[:10]}


def top_source_token_check(expected_source_tokens: list[str], turns: list[dict]) -> tuple[bool, dict]:
    top_files = [
        turn["retrieved_sources"][0]["file_name"]
        for turn in turns
        if turn.get("retrieved_sources")
    ]
    lowered_top_files = [file_name.casefold() for file_name in top_files]
    lowered_tokens = [token.casefold() for token in expected_source_tokens]
    matched_turns = 0
    for file_name in lowered_top_files:
        if any(token in file_name for token in lowered_tokens):
            matched_turns += 1
    ratio = matched_turns / max(len(lowered_top_files), 1)
    return ratio >= 0.7, {
        "top_files": top_files,
        "matched_turns": matched_turns,
        "match_ratio": round(ratio, 2),
    }


def low_signal_answer_check(turns: list[dict]) -> tuple[bool, dict]:
    low_signal_markers = (
        "legal notice",
        "copyright",
        "creative commons",
        "all rights reserved",
        "table of contents",
        "default oauth clients",
        "registering an additional oauth client",
    )
    violations: list[dict] = []
    for turn in turns:
        lowered_answer = str(turn.get("answer", "")).casefold()
        matched = [marker for marker in low_signal_markers if marker in lowered_answer]
        if matched:
            violations.append(
                {
                    "turn": turn.get("turn"),
                    "markers": matched,
                    "answer_preview": str(turn.get("answer", ""))[:180],
                }
            )
    return len(violations) == 0, {"violations": violations}


def summarize_scenario(scenario: dict, turns: list[dict]) -> dict:
    total_turns = len(scenario["turns"])
    completed = sum(1 for turn in turns if turn.get("done"))
    llm_failures = sum(1 for turn in turns if turn.get("llm_generation_failed"))
    version_mismatches = sum(1 for turn in turns if turn.get("version_mismatch"))
    turns_with_citations = sum(1 for turn in turns if turn.get("answer_citations"))
    avg_elapsed = round(sum(turn.get("elapsed_sec", 0.0) for turn in turns) / max(total_turns, 1), 2)
    group_ok, group_detail = group_check(scenario["expected_group"], turns)
    source_ok, source_detail = source_token_check(scenario.get("expected_source_tokens", []), turns)
    top_source_ok, top_source_detail = top_source_token_check(scenario.get("expected_source_tokens", []), turns)
    low_signal_ok, low_signal_detail = low_signal_answer_check(turns)

    checks = {
        "all_turns_completed": completed == total_turns,
        "llm_generation_stable": llm_failures == 0,
        "version_tag_filter_works": version_mismatches == 0,
        "citations_present_in_most_turns": turns_with_citations >= math.ceil(total_turns * 0.7),
        "expected_group_routed": group_ok,
        "expected_source_family_hit": source_ok,
        "top_source_stays_on_expected_family": top_source_ok,
        "answers_avoid_low_signal_front_matter": low_signal_ok,
    }
    return {
        "scenario_id": scenario["id"],
        "title": scenario["title"],
        "version_tag": scenario["version_tag"],
        "expected_group": scenario["expected_group"],
        "grounding_markdown": scenario["grounding_markdown"],
        "turn_results": {
            "total": total_turns,
            "completed": completed,
            "avg_elapsed_sec": avg_elapsed,
            "llm_generation_failed": llm_failures,
            "version_mismatch": version_mismatches,
            "turns_with_citations": turns_with_citations,
        },
        "checks": checks,
        "group_detail": group_detail,
        "source_detail": source_detail,
        "top_source_detail": top_source_detail,
        "answer_quality_detail": low_signal_detail,
    }


def run_scenario(base_url: str, scenario: dict, output_dir: Path) -> dict:
    session_id = f"v401-{scenario['id']}-{uuid.uuid4().hex[:8]}"
    print(f"\n{'#' * 90}")
    print(f"[Scenario] {scenario['title']}")
    print(f"[Session ] {session_id}")
    print(f"[Version ] {scenario['version_tag']}")
    print(f"[Group   ] {scenario['expected_group']}")
    turns = []
    for turn_idx, question in enumerate(scenario["turns"], start=1):
        turns.append(run_turn(base_url, session_id, turn_idx, question, scenario.get("version_tag")))

    summary = summarize_scenario(scenario, turns)
    payload = {
        "summary": summary,
        "turns": turns,
        "executed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    output_path = output_dir / f"{session_id}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output_path"] = str(output_path)
    return payload


def print_summary(payloads: list[dict]) -> None:
    print(f"\n{'#' * 90}")
    print("# v4.0.1 Multiturn Scenario Summary")
    print("#" * 90)
    for payload in payloads:
        summary = payload["summary"]
        print(f"\n[{summary['scenario_id']}] {summary['title']}")
        for key, value in summary["checks"].items():
            marker = "PASS" if value else "FAIL"
            print(f"  [{marker}] {key}")
        print(
            "  turns=",
            summary["turn_results"]["completed"],
            "/",
            summary["turn_results"]["total"],
            "citations=",
            summary["turn_results"]["turns_with_citations"],
            "avg_sec=",
            summary["turn_results"]["avg_elapsed_sec"],
        )
        print("  groups=", summary["group_detail"])
        print("  sources=", summary["source_detail"])
        print("  output=", summary["output_path"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v4.0.1 multiturn scenario pack.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--scenario-file",
        default="tests/data/v4_0_1_multiturn_scenarios.json",
    )
    parser.add_argument(
        "--scenario-id",
        default="",
        help="Run only one scenario id.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/results/v4_0_2",
    )
    args = parser.parse_args()

    scenario_file = Path(args.scenario_file)
    bundle = load_scenarios(scenario_file)
    scenarios = bundle["scenarios"]
    if args.scenario_id:
        scenarios = [scenario for scenario in scenarios if scenario["id"] == args.scenario_id]
        if not scenarios:
            raise SystemExit(f"Scenario not found: {args.scenario_id}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = [run_scenario(args.base_url.rstrip("/"), scenario, output_dir) for scenario in scenarios]
    print_summary(payloads)

    all_checks = [
        value
        for payload in payloads
        for value in payload["summary"]["checks"].values()
    ]
    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    sys.exit(main())

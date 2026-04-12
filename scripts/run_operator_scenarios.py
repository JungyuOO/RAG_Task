from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path


def _stream_chat(base_url: str, session_id: str, message: str) -> tuple[list[dict], float]:
    started = time.perf_counter()
    payload = json.dumps({"session_id": session_id, "message": message}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-Id": "operator-scenario-runner",
        },
        method="POST",
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_bytes in response:
            raw = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not raw.startswith("data: "):
                continue
            try:
                events.append(json.loads(raw[6:]))
            except json.JSONDecodeError:
                continue
    return events, round(time.perf_counter() - started, 2)


def _extract_turn_result(events: list[dict], elapsed_sec: float) -> dict:
    context_events = [event for event in events if event.get("type") == "context"]
    context = context_events[-1] if context_events else {}
    answer = "\n".join(str(event.get("content", "")) for event in events if event.get("type") == "token").strip()
    items = list(context.get("items", []) or [])
    answer_citations = list(context.get("answer_citations", []) or [])
    preview_pages = list(context.get("preview_pages", []) or [])
    source_paths = [str(item.get("source_path") or "") for item in items]
    source_paths.extend(str(citation.get("source_path") or "") for citation in answer_citations if citation.get("source_path"))
    source_paths.extend(str(page.get("source_path") or "") for page in preview_pages if page.get("source_path"))
    deduped_source_paths: list[str] = []
    for path in source_paths:
        if path and path not in deduped_source_paths:
            deduped_source_paths.append(path)
    return {
        "context": context,
        "answer": answer,
        "elapsed_sec": elapsed_sec,
        "source_paths": deduped_source_paths,
        "source_tokens": [Path(path).name.casefold() for path in deduped_source_paths if path],
        "context_item_count": len(items),
        "answer_citation_count": len(answer_citations),
        "preview_page_count": len(preview_pages),
    }


def _route_name(context: dict) -> str:
    return str(context.get("answer_route") or "")


def _matches_expected_route(expected_route: str, context: dict) -> bool:
    actual = _route_name(context)
    if expected_route == "document":
        return actual in {"extractive_code", "extractive_text", "extractive_table", "extractive_compare", "grounded_generation", ""}
    return actual == expected_route


def _evaluate_turn(turn_spec: dict, turn_result: dict) -> dict:
    answer = str(turn_result["answer"])
    lowered_answer = answer.casefold()
    context = turn_result["context"]
    failures: list[str] = []
    expected_route = str(turn_spec.get("expected_route") or "")
    if expected_route and not _matches_expected_route(expected_route, context):
        failures.append(f"route mismatch: expected={expected_route} actual={_route_name(context)}")

    for keyword in turn_spec.get("required_keywords", []) or []:
        if str(keyword).casefold() not in lowered_answer:
            failures.append(f"missing keyword: {keyword}")

    for pattern in turn_spec.get("required_regexes", []) or []:
        if not re.search(str(pattern), answer, flags=re.IGNORECASE | re.MULTILINE):
            failures.append(f"missing regex: {pattern}")

    for pattern in turn_spec.get("forbidden_regexes", []) or []:
        if re.search(str(pattern), answer, flags=re.IGNORECASE | re.MULTILINE):
            failures.append(f"forbidden regex matched: {pattern}")

    source_tokens = turn_result["source_tokens"]
    expected_source_tokens = [str(token).casefold() for token in (turn_spec.get("required_source_tokens", []) or [])]
    if expected_source_tokens and not any(token in " ".join(source_tokens) for token in expected_source_tokens):
        failures.append("missing expected source token")

    min_answer_citations = int(turn_spec.get("min_answer_citations") or 0)
    if min_answer_citations and int(turn_result["answer_citation_count"]) < min_answer_citations:
        failures.append(
            f"answer citations too low: {turn_result['answer_citation_count']} < {min_answer_citations}"
        )

    min_preview_pages = int(turn_spec.get("min_preview_pages") or 0)
    if min_preview_pages and int(turn_result["preview_page_count"]) < min_preview_pages:
        failures.append(
            f"preview pages too low: {turn_result['preview_page_count']} < {min_preview_pages}"
        )

    if turn_spec.get("require_inline_source"):
        inline_source_patterns = (
            r"\[source:[^\]]+\]",
            r"\[[^\[\]\n]+\.[A-Za-z0-9]{2,8}\]\s*p\.\d+",
        )
        if not any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in inline_source_patterns):
            failures.append("missing inline source marker")

    max_latency_sec = float(turn_spec.get("max_latency_sec") or 0.0)
    if max_latency_sec and float(turn_result["elapsed_sec"]) > max_latency_sec:
        failures.append(f"latency too high: {turn_result['elapsed_sec']}s > {max_latency_sec}s")

    if int(turn_result["context_item_count"]) > 3:
        failures.append(f"context item count too high: {turn_result['context_item_count']}")

    return {
        "message": turn_spec.get("message"),
        "route": _route_name(context),
        "elapsed_sec": turn_result["elapsed_sec"],
        "context_item_count": turn_result["context_item_count"],
        "answer_citation_count": turn_result["answer_citation_count"],
        "preview_page_count": turn_result["preview_page_count"],
        "source_tokens": source_tokens[:6],
        "answer_preview": answer[:500],
        "passed": not failures,
        "failures": failures,
    }


def run_scenarios(base_url: str, scenario_path: Path) -> dict:
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    results: list[dict] = []
    for scenario in scenarios:
        session_id = f"scenario-{scenario['id']}-{uuid.uuid4().hex[:8]}"
        turn_results: list[dict] = []
        for turn in scenario.get("turns", []):
            events, elapsed_sec = _stream_chat(base_url, session_id, str(turn.get("message") or ""))
            turn_result = _extract_turn_result(events, elapsed_sec)
            turn_results.append(_evaluate_turn(turn, turn_result))
        results.append(
            {
                "id": scenario.get("id"),
                "kind": scenario.get("kind"),
                "passed": all(result["passed"] for result in turn_results),
                "turns": turn_results,
            }
        )
    return {
        "version": payload.get("version"),
        "title": payload.get("title"),
        "scenario_count": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run operator scenario checks against /api/chat.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--scenario-file", default="tests/data/operator_scenarios_v1.json")
    args = parser.parse_args()

    payload = run_scenarios(args.base_url, Path(args.scenario_file))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.golden_dataset import GoldenCase, GoldenDataset, load_golden_dataset


def source_group(source_path: str) -> str:
    normalized = str(source_path or "").replace("\\", "/").casefold()
    if "/generated_pdf/" in normalized or "/generated/" in normalized or "customer-guide" in normalized:
        return "customer_generated"
    return "official_ocp"


def stream_chat(base_url: str, session_id: str, message: str, version_tag: str | None) -> list[dict]:
    payload = {"session_id": session_id, "message": message}
    if version_tag:
        payload["version_tag"] = version_tag
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-Id": "golden-dataset-runner",
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
    return events


def extract_turn_result(base_url: str, case: GoldenCase) -> dict[str, object]:
    session_id = f"golden-{case.id}-{uuid.uuid4().hex[:8]}"
    answer_parts: list[str] = []
    replaced_answer: str | None = None
    context_event: dict[str, object] = {}
    done = False
    started = time.time()

    for event in stream_chat(base_url, session_id, case.question, case.version_tag):
        event_type = event.get("type")
        if event_type == "token":
            answer_parts.append(str(event.get("content", "")))
        elif event_type == "replace_answer":
            replaced_answer = str(event.get("content", ""))
        elif event_type == "context":
            context_event = event
        elif event_type == "done":
            done = True

    answer = replaced_answer if replaced_answer is not None else "".join(answer_parts).strip()
    retrieved_sources = []
    for item in context_event.get("items", [])[:10]:
        path = str(item.get("source_path") or "")
        retrieved_sources.append(
            {
                "file_name": Path(path).name,
                "source_path": path,
                "group": source_group(path),
                "page_number": item.get("page_number"),
            }
        )
    answer_citations = list(context_event.get("answer_citations", []) or [])
    return {
        "session_id": session_id,
        "elapsed_sec": round(time.time() - started, 2),
        "done": done,
        "answer": answer,
        "retrieved_sources": retrieved_sources,
        "answer_citations": answer_citations,
        "context_event": context_event,
    }


def detect_answer_shape(answer: str) -> str:
    text = str(answer or "").strip()
    lowered = text.casefold()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(re.findall(r"^\s*(?:[-*]\s+|\d+\.\s+)", text, flags=re.MULTILINE)) >= 3:
        if "vs" in lowered or "차이" in lowered or "비교" in lowered:
            return "comparison"
        if any(re.match(r"^\d+\.\s+", line) for line in lines):
            return "procedure"
        return "checklist"

    if "비교" in lowered or "차이" in lowered or "각각" in lowered or "반면" in lowered:
        return "comparison"
    if "단계" in lowered or "순서" in lowered or "먼저" in lowered or "다음" in lowered:
        return "procedure"
    if "상태" in lowered or "warning" in lowered or "이벤트" in lowered or "점검" in lowered:
        return "status_analysis"
    return "summary"


def evaluate_case(case: GoldenCase, turn_result: dict[str, object]) -> dict[str, object]:
    answer = str(turn_result.get("answer", ""))
    lowered_answer = answer.casefold()
    retrieved_sources = list(turn_result.get("retrieved_sources", []))
    observed_groups = sorted({str(item.get("group") or "") for item in retrieved_sources if item.get("group")})
    top_group = str(retrieved_sources[0].get("group") or "") if retrieved_sources else ""
    observed_files = [str(item.get("file_name") or "") for item in retrieved_sources]
    lowered_files = [file_name.casefold() for file_name in observed_files]
    matched_source_tokens = [
        token for token in case.expected_source_tokens
        if any(token.casefold() in file_name for file_name in lowered_files)
    ]
    matched_keywords = [
        keyword for keyword in case.required_keywords
        if keyword.casefold() in lowered_answer
    ]
    forbidden_hits = [
        marker for marker in case.forbidden_markers
        if marker.casefold() in lowered_answer
    ]
    detected_shape = detect_answer_shape(answer)
    group_ok = (
        {"official_ocp", "customer_generated"}.issubset(set(observed_groups))
        if case.expected_group == "mixed"
        else top_group == case.expected_group
    )
    checks = {
        "completed": bool(turn_result.get("done")),
        "group_match": group_ok,
        "source_match": bool(matched_source_tokens),
        "keywords_present": len(matched_keywords) >= max(1, min(2, len(case.required_keywords))),
        "forbidden_markers_absent": not forbidden_hits,
        "answer_shape_match": detected_shape == case.expected_answer_shape,
        "citations_present": bool(turn_result.get("answer_citations")),
    }
    return {
        "case_id": case.id,
        "question": case.question,
        "expected_group": case.expected_group,
        "expected_answer_shape": case.expected_answer_shape,
        "detected_answer_shape": detected_shape,
        "checks": checks,
        "matched_source_tokens": matched_source_tokens,
        "matched_keywords": matched_keywords,
        "forbidden_hits": forbidden_hits,
        "observed_groups": observed_groups,
        "observed_files": observed_files[:10],
        "elapsed_sec": turn_result.get("elapsed_sec"),
        "answer_preview": answer[:400],
    }


def run_dataset(base_url: str, dataset: GoldenDataset, *, case_id: str = "", max_cases: int = 0) -> dict[str, object]:
    selected_cases = list(dataset.cases)
    if case_id:
        selected_cases = [case for case in selected_cases if case.id == case_id]
        if not selected_cases:
            raise ValueError(f"case_id not found: {case_id}")
    if max_cases > 0:
        selected_cases = selected_cases[:max_cases]

    results = []
    for case in selected_cases:
        turn_result = extract_turn_result(base_url, case)
        results.append(evaluate_case(case, turn_result))
    passed = sum(1 for result in results if all(result["checks"].values()))
    return {
        "dataset_version": dataset.version,
        "title": dataset.title,
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def summarize_failures(payload: dict[str, object]) -> dict[str, object]:
    results = list(payload.get("results", []))
    failed_results = [result for result in results if not all((result.get("checks") or {}).values())]
    failure_buckets: dict[str, int] = {}
    for result in failed_results:
        for check_name, passed in (result.get("checks") or {}).items():
            if not passed:
                failure_buckets[check_name] = failure_buckets.get(check_name, 0) + 1
    return {
        "failed_case_count": len(failed_results),
        "failure_buckets": dict(sorted(failure_buckets.items(), key=lambda item: item[0])),
    }


def render_markdown_report(payload: dict[str, object]) -> str:
    lines = [
        "# Golden Dataset Report",
        "",
        f"- Dataset: `{payload.get('title', '')}`",
        f"- Version: `{payload.get('dataset_version', '')}`",
        f"- Cases: `{payload.get('case_count', 0)}`",
        f"- Passed: `{payload.get('passed', 0)}`",
        f"- Failed: `{payload.get('failed', 0)}`",
        "",
        "## Failures",
    ]
    failure_summary = summarize_failures(payload)
    failure_buckets = failure_summary.get("failure_buckets", {})
    if not failure_buckets:
        lines.append("- None")
    else:
        for key, count in failure_buckets.items():
            lines.append(f"- `{key}`: {count}")

    lines.extend(["", "## Case Results"])
    for result in payload.get("results", []):
        checks = result.get("checks", {})
        passed = all(checks.values())
        lines.append(f"### {'PASS' if passed else 'FAIL'} `{result.get('case_id', '')}`")
        lines.append(f"- Question: {result.get('question', '')}")
        lines.append(f"- Expected group: `{result.get('expected_group', '')}`")
        lines.append(f"- Expected shape: `{result.get('expected_answer_shape', '')}`")
        lines.append(f"- Detected shape: `{result.get('detected_answer_shape', '')}`")
        lines.append(f"- Elapsed: `{result.get('elapsed_sec', 0)}`s")
        lines.append(f"- Observed groups: `{', '.join(result.get('observed_groups', []))}`")
        lines.append(f"- Matched source tokens: `{', '.join(result.get('matched_source_tokens', []))}`")
        lines.append(f"- Matched keywords: `{', '.join(result.get('matched_keywords', []))}`")
        forbidden_hits = result.get("forbidden_hits", [])
        lines.append(f"- Forbidden hits: `{', '.join(forbidden_hits) if forbidden_hits else 'none'}`")
        failed_checks = [name for name, ok in checks.items() if not ok]
        lines.append(f"- Failed checks: `{', '.join(failed_checks) if failed_checks else 'none'}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run golden dataset checks against /api/chat.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="tests/data/golden_dataset_v1.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_golden_dataset(args.dataset)
    payload = run_dataset(args.base_url, dataset, case_id=args.case_id, max_cases=args.max_cases)
    payload["failure_summary"] = summarize_failures(payload)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        json_path = output_dir / f"golden-run-{stamp}.json"
        md_path = output_dir / f"golden-run-{stamp}.md"
        json_path.write_text(rendered, encoding="utf-8")
        md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

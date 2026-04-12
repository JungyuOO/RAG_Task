from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_run(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_map(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(case.get("case_id") or ""): case
        for case in payload.get("results", [])
        if case.get("case_id")
    }


def compare_runs(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    baseline_cases = _case_map(baseline)
    candidate_cases = _case_map(candidate)
    all_case_ids = sorted(set(baseline_cases) | set(candidate_cases))

    changed_cases: list[dict[str, object]] = []
    improved = 0
    regressed = 0

    for case_id in all_case_ids:
        before = baseline_cases.get(case_id, {})
        after = candidate_cases.get(case_id, {})
        before_checks = dict(before.get("checks", {}))
        after_checks = dict(after.get("checks", {}))
        if not before_checks and not after_checks:
            continue

        before_pass = all(before_checks.values()) if before_checks else False
        after_pass = all(after_checks.values()) if after_checks else False
        changed_checks = {
            key: {"before": before_checks.get(key), "after": after_checks.get(key)}
            for key in sorted(set(before_checks) | set(after_checks))
            if before_checks.get(key) != after_checks.get(key)
        }
        before_elapsed = float(before.get("elapsed_sec", 0.0) or 0.0)
        after_elapsed = float(after.get("elapsed_sec", 0.0) or 0.0)

        if before_pass != after_pass or changed_checks or before_elapsed != after_elapsed:
            if not before_pass and after_pass:
                improved += 1
            elif before_pass and not after_pass:
                regressed += 1
            changed_cases.append(
                {
                    "case_id": case_id,
                    "before_pass": before_pass,
                    "after_pass": after_pass,
                    "changed_checks": changed_checks,
                    "before_elapsed_sec": before_elapsed,
                    "after_elapsed_sec": after_elapsed,
                    "elapsed_delta_sec": round(after_elapsed - before_elapsed, 2),
                }
            )

    return {
        "baseline_cases": len(baseline_cases),
        "candidate_cases": len(candidate_cases),
        "improved": improved,
        "regressed": regressed,
        "changed_cases": changed_cases,
    }


def render_markdown_report(diff: dict[str, object], baseline_path: str, candidate_path: str) -> str:
    lines = [
        "# Golden Run Comparison",
        "",
        f"- Baseline: `{baseline_path}`",
        f"- Candidate: `{candidate_path}`",
        f"- Improved: `{diff.get('improved', 0)}`",
        f"- Regressed: `{diff.get('regressed', 0)}`",
        f"- Changed cases: `{len(diff.get('changed_cases', []))}`",
        "",
        "## Case Changes",
    ]
    changed_cases = list(diff.get("changed_cases", []))
    if not changed_cases:
        lines.append("- None")
    else:
        for case in changed_cases:
            lines.append(f"### `{case.get('case_id', '')}`")
            lines.append(f"- Pass: `{case.get('before_pass')}` -> `{case.get('after_pass')}`")
            lines.append(f"- Elapsed: `{case.get('before_elapsed_sec')}`s -> `{case.get('after_elapsed_sec')}`s")
            changed_checks = case.get("changed_checks", {})
            if changed_checks:
                for name, values in changed_checks.items():
                    lines.append(f"- Check `{name}`: `{values.get('before')}` -> `{values.get('after')}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two golden run JSON files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    diff = compare_runs(baseline, candidate)
    payload = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "diff": diff,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(rendered, encoding="utf-8")
        output_path.with_suffix(".md").write_text(
            render_markdown_report(diff, args.baseline, args.candidate),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

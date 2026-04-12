from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_golden_runs import compare_runs, render_markdown_report
from scripts.golden_dataset import load_golden_dataset, summarize_golden_dataset
from scripts.run_golden_dataset import run_dataset


DEFAULT_DATASET = Path("tests/data/golden_dataset_v1.json")
DEFAULT_OUTPUT_DIR = Path("tests/results/golden")


def _latest_golden_jsons(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("golden-run-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def _build_run_paths(output_dir: Path) -> tuple[Path, Path]:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        output_dir / f"golden-run-{stamp}.json",
        output_dir / f"golden-run-{stamp}.md",
    )


def execute_pipeline(
    *,
    dataset_path: Path,
    base_url: str,
    output_dir: Path,
    case_id: str = "",
    max_cases: int = 0,
    compare_latest: bool = True,
) -> dict[str, object]:
    dataset = load_golden_dataset(dataset_path)
    dataset_summary = summarize_golden_dataset(dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_json_path, run_md_path = _build_run_paths(output_dir)

    payload = run_dataset(base_url, dataset, case_id=case_id, max_cases=max_cases)
    payload["dataset_summary"] = dataset_summary
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    run_json_path.write_text(rendered, encoding="utf-8")

    from scripts.run_golden_dataset import render_markdown_report, summarize_failures

    payload["failure_summary"] = summarize_failures(payload)
    run_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_md_path.write_text(render_markdown_report(payload), encoding="utf-8")

    compare_payload: dict[str, object] | None = None
    if compare_latest:
        latest_runs = [path for path in _latest_golden_jsons(output_dir) if path != run_json_path]
        if latest_runs:
            baseline_path = latest_runs[0]
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            compare_payload = {
                "baseline": str(baseline_path),
                "candidate": str(run_json_path),
                "diff": compare_runs(baseline, payload),
            }
            compare_json_path = output_dir / f"golden-compare-{run_json_path.stem.replace('golden-run-', '')}.json"
            compare_json_path.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            compare_json_path.with_suffix(".md").write_text(
                render_markdown_report(compare_payload["diff"], str(baseline_path), str(run_json_path)),
                encoding="utf-8",
            )

    return {
        "dataset": str(dataset_path),
        "output_json": str(run_json_path),
        "output_md": str(run_md_path),
        "case_count": payload["case_count"],
        "passed": payload["passed"],
        "failed": payload["failed"],
        "compared": bool(compare_payload),
        "dataset_summary": dataset_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the golden dataset pipeline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case-id", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--skip-compare", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = execute_pipeline(
        dataset_path=Path(args.dataset),
        base_url=args.base_url,
        output_dir=Path(args.output_dir),
        case_id=args.case_id,
        max_cases=args.max_cases,
        compare_latest=not args.skip_compare,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

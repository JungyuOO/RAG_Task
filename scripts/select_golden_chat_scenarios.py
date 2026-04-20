from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests" / "data" / "chat_eval_dataset_v1.json"
RESULTS_DIR = ROOT / "tests" / "results" / "chat-eval"
OUTPUT_PATH = ROOT / "tests" / "data" / "chat_eval_golden_scenarios.json"


RESULT_FILES = [
    "chat-eval-20260420-095339.json",
    "chat-eval-20260420-095543.json",
    "chat-eval-20260420-100310.json",
    "chat-eval-20260420-100531.json",
]


def score_result(result: dict) -> tuple[float, float, float]:
    overlaps: list[float] = []
    elapsed: list[float] = []
    for step in result.get("steps", []):
        elapsed.append(float(step.get("elapsed_sec") or 0.0))
        evidence = step.get("evidence") or {}
        checks: list[dict] = []
        if isinstance(evidence, dict):
            checks.extend(evidence.get("citation_checks") or [])
            doc = evidence.get("doc") or {}
            if isinstance(doc, dict):
                checks.extend(doc.get("citation_checks") or [])
        overlaps.extend(
            float(check.get("snippet_overlap") or 0.0)
            for check in checks
            if check.get("snippet_overlap") is not None
        )
    avg_overlap = mean(overlaps) if overlaps else 1.0
    total_elapsed = sum(elapsed)
    score = round((avg_overlap * 100.0) - (total_elapsed / 20.0), 2)
    return score, round(avg_overlap, 3), round(total_elapsed, 1)


def main() -> int:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset_by_id = {scenario["id"]: scenario for scenario in dataset["scenarios"]}

    passed_results: list[dict] = []
    for file_name in RESULT_FILES:
        path = RESULTS_DIR / file_name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            if result.get("passed"):
                passed_results.append(result)

    scored_results: list[dict] = []
    for result in passed_results:
        scenario = dataset_by_id.get(result["id"])
        if scenario is None:
            continue
        score, avg_overlap, total_elapsed = score_result(result)
        scored_results.append(
            {
                "id": result["id"],
                "family": scenario["family"],
                "category": scenario["category"],
                "score": score,
                "avg_citation_overlap": avg_overlap,
                "total_elapsed_sec": total_elapsed,
                "description": scenario["description"],
            }
        )

    by_family: dict[str, list[dict]] = defaultdict(list)
    for item in scored_results:
        by_family[item["family"]].append(item)
    for family_items in by_family.values():
        family_items.sort(key=lambda item: (-item["score"], item["total_elapsed_sec"], item["id"]))

    golden_ids: list[str] = []
    golden_summary: list[dict] = []
    for family, items in sorted(by_family.items()):
        best = items[0]
        golden_ids.append(best["id"])
        golden_summary.append(best)

    # Also keep extra strong scenarios with very high score for broader smoke coverage.
    extra_candidates = [
        item for item in scored_results
        if item["id"] not in golden_ids and item["score"] >= 97.0
    ]
    extra_candidates.sort(key=lambda item: (-item["score"], item["total_elapsed_sec"], item["id"]))
    for item in extra_candidates:
        golden_ids.append(item["id"])
        golden_summary.append(item)

    golden_scenarios = [dataset_by_id[scenario_id] for scenario_id in golden_ids]
    output = {
        "version": dataset["version"],
        "source_dataset": str(DATASET_PATH),
        "selected_from_results": RESULT_FILES,
        "golden_count": len(golden_scenarios),
        "golden_ids": golden_ids,
        "summary": golden_summary,
        "scenarios": golden_scenarios,
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUTPUT_PATH}")
    print(json.dumps({"golden_count": len(golden_scenarios), "golden_ids": golden_ids}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

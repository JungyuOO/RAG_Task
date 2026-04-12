from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.golden_dataset import load_golden_dataset, summarize_golden_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the golden dataset schema.")
    parser.add_argument(
        "--dataset",
        default="tests/data/golden_dataset_v1.json",
        help="Path to the golden dataset JSON file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_golden_dataset(Path(args.dataset))
    summary = summarize_golden_dataset(dataset)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

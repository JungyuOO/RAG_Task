"""Prune imported openshift-docs markdown tree to an allowlisted subset."""
from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="data/corpus/pdfs/ocp-4.20-openshift-docs",
        help="Imported openshift-docs markdown root.",
    )
    parser.add_argument(
        "--allowlist",
        default="scripts/allowlists/openshift_docs_core_4.20.txt",
        help="Relative paths to keep, one per line.",
    )
    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help="Remove empty directories after pruning.",
    )
    return parser.parse_args()


def read_allowlist(path: Path) -> set[str]:
    rows: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().replace("\\", "/")
        if not line or line.startswith("#"):
            continue
        rows.add(line)
    return rows


def _delete_file(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    path.unlink()


def prune_tree(input_root: Path, allowlist: set[str], *, remove_empty_dirs: bool = False) -> tuple[int, int]:
    kept = 0
    deleted = 0
    input_root = input_root.resolve()

    for path in sorted(input_root.rglob("*.md")):
        relative = str(path.relative_to(input_root)).replace("\\", "/")
        if relative in allowlist:
            kept += 1
            continue
        _delete_file(path)
        deleted += 1

    if remove_empty_dirs:
        for directory in sorted((path for path in input_root.rglob("*") if path.is_dir()), key=lambda item: len(item.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass

    return kept, deleted


def main() -> None:
    args = parse_args()
    input_root = (ROOT_DIR / args.input_root).resolve()
    allowlist_path = (ROOT_DIR / args.allowlist).resolve()
    allowlist = read_allowlist(allowlist_path)
    kept, deleted = prune_tree(input_root, allowlist, remove_empty_dirs=args.remove_empty_dirs)
    print(f"kept={kept}")
    print(f"deleted={deleted}")


if __name__ == "__main__":
    main()

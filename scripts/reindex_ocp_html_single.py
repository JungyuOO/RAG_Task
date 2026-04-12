"""Reindex imported html-single markdown sources."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.dependencies import build_container

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rag.reindex_ocp_html_single")


def collect_markdown_sources(root: Path, paths: list[str]) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"html-single root not found: {root}")
    if not paths:
        return sorted(path.resolve() for path in root.rglob("*.md") if path.is_file())
    files: list[Path] = []
    for target in paths:
        resolved = (root / target).resolve()
        if resolved.is_file():
            files.append(resolved)
        else:
            files.extend(sorted(path.resolve() for path in resolved.rglob("*.md") if path.is_file()))
    return sorted(dict.fromkeys(files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/corpus/pdfs")
    parser.add_argument("--path", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    container = build_container(settings)
    source_root = Path(args.root)
    source_files = collect_markdown_sources(source_root, args.path)
    logger.info("html-single reindex start: root=%s file_count=%d", source_root, len(source_files))
    indexed_chunks = 0
    for source_path in source_files:
        result = container.indexing_service.index_single_file(source_path)
        indexed_chunks += int(result.get("indexed_chunks", 0))
        logger.info("reindexed %s -> %d chunks", source_path.name, int(result.get("indexed_chunks", 0)))
    logger.info("html-single reindex done: files=%d indexed_chunks=%d", len(source_files), indexed_chunks)


if __name__ == "__main__":
    main()

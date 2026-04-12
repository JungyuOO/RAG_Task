"""Reindex selected official OCP version folders without wiping the full index.

Usage:
    python scripts/reindex_versions.py --version 4.15
    python scripts/reindex_versions.py --version 4.15 --version 4.21
    python scripts/reindex_versions.py --version 4.15 --clear-embedding-cache
"""
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("rag.reindex_versions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="append",
        required=True,
        help="Target OCP version, for example 4.15. Can be passed multiple times.",
    )
    parser.add_argument(
        "--clear-embedding-cache",
        action="store_true",
        help="Clear the embedding cache before processing the selected versions.",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Warm the vector index cache after reindexing completes.",
    )
    return parser.parse_args()


def collect_version_files(source_dir: Path, versions: list[str]) -> list[Path]:
    files: list[Path] = []
    for version in versions:
        version_dir = source_dir / f"ocp-{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")
        version_files = sorted(
            path for path in version_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        if not version_files:
            raise FileNotFoundError(f"No PDF files found under: {version_dir}")
        files.extend(version_files)
    return files


def main() -> None:
    args = parse_args()
    versions = list(dict.fromkeys(str(version).strip() for version in args.version if str(version).strip()))
    settings = get_settings()
    container = build_container(settings)

    source_files = collect_version_files(settings.rag_source_dir, versions)
    logger.info(
        "version reindex start: versions=%s file_count=%d",
        ",".join(versions),
        len(source_files),
    )

    if args.clear_embedding_cache:
        container.indexing_service.embedding_cache_repository.clear()
        logger.info("embedding cache cleared")

    indexed_files = 0
    indexed_chunks = 0
    skipped_files = 0

    for index, source_path in enumerate(source_files, start=1):
        logger.info(
            "[%d/%d] reindex %s",
            index,
            len(source_files),
            source_path.name,
        )
        result = container.indexing_service.index_single_file(source_path)
        indexed_chunks += int(result.get("indexed_chunks", 0))
        if result.get("skipped"):
            skipped_files += 1
        else:
            indexed_files += 1

    warmed_items = None
    if args.warm_cache:
        warmed_items = container.pipeline.index_repository.warm_cache()
        logger.info("index cache warmed: items=%d", warmed_items)

    logger.info(
        "version reindex done: versions=%s indexed_files=%d indexed_chunks=%d skipped_files=%d%s",
        ",".join(versions),
        indexed_files,
        indexed_chunks,
        skipped_files,
        f" warmed_items={warmed_items}" if warmed_items is not None else "",
    )


if __name__ == "__main__":
    main()

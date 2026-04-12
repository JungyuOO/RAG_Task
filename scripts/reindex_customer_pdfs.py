"""Reindex customer-guide PDFs locally after clearing extracted artifacts.

Usage:
    python scripts/reindex_customer_pdfs.py
    python scripts/reindex_customer_pdfs.py --path ocp-4.20-network-customer-guide.pdf
    python scripts/reindex_customer_pdfs.py --path team-a --clear-embedding-cache --warm-cache
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
from scripts.reindex_official_pdfs import delete_extracted_artifacts, purge_legacy_index_entries


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("rag.reindex_customer_pdfs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="data/corpus/pdfs/generated_pdf",
        help="Customer PDF root directory. Default: data/corpus/pdfs/generated_pdf",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Optional relative path under the customer PDF root. Can be a PDF file or subdirectory. Can be repeated.",
    )
    parser.add_argument(
        "--keep-extracted-artifacts",
        action="store_true",
        help="Do not remove existing extracted markdown/html/json artifacts before reindexing.",
    )
    parser.add_argument(
        "--clear-embedding-cache",
        action="store_true",
        help="Clear the embedding cache before processing.",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Warm the vector index cache after reindexing completes.",
    )
    return parser.parse_args()


def _collect_pdfs_in_path(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Customer PDF path not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Not a PDF file: {path}")
        return [path.resolve()]
    pdfs = sorted(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under: {path}")
    return pdfs


def collect_customer_pdfs(input_root: Path, paths: list[str]) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Customer PDF root not found: {input_root}")

    targets = [str(path).strip() for path in paths if str(path).strip()]
    if not targets:
        return _collect_pdfs_in_path(input_root)

    pdfs: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved_target = (input_root / target).resolve()
        resolved_root = input_root.resolve()
        if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
            raise ValueError(f"Path escapes customer PDF root: {target}")
        for pdf_path in _collect_pdfs_in_path(resolved_target):
            if pdf_path not in seen:
                pdfs.append(pdf_path)
                seen.add(pdf_path)
    return sorted(pdfs)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    container = build_container(settings)

    input_root = (ROOT_DIR / args.input_root).resolve()
    source_files = collect_customer_pdfs(input_root, args.path)
    logger.info(
        "customer pdf reindex start: root=%s targets=%s file_count=%d",
        input_root,
        ",".join(args.path) if args.path else "all",
        len(source_files),
    )

    if not args.keep_extracted_artifacts:
        deleted_artifacts = delete_extracted_artifacts(settings.rag_extract_dir, source_files)
        logger.info("deleted extracted artifacts: %d", deleted_artifacts)

    if args.clear_embedding_cache:
        container.indexing_service.embedding_cache_repository.clear()
        logger.info("embedding cache cleared")

    indexed_files = 0
    indexed_chunks = 0
    skipped_files = 0

    for index, source_path in enumerate(source_files, start=1):
        logger.info("[%d/%d] reindex %s", index, len(source_files), source_path.name)
        purged = purge_legacy_index_entries(container.pipeline.index_repository, settings.rag_source_dir, source_path)
        if purged:
            logger.info("purged legacy index entries: file=%s count=%d", source_path.name, purged)
        result = container.indexing_service.index_single_file(source_path)
        indexed_chunks += int(result.get("indexed_chunks", 0))
        if result.get("skipped"):
            skipped_files += 1
            logger.warning("skipped %s", source_path.name)
        else:
            indexed_files += 1

    warmed_items = None
    if args.warm_cache:
        warmed_items = container.pipeline.index_repository.warm_cache()
        logger.info("index cache warmed: items=%d", warmed_items)

    logger.info(
        "customer pdf reindex done: indexed_files=%d indexed_chunks=%d skipped_files=%d%s",
        indexed_files,
        indexed_chunks,
        skipped_files,
        f" warmed_items={warmed_items}" if warmed_items is not None else "",
    )


if __name__ == "__main__":
    main()

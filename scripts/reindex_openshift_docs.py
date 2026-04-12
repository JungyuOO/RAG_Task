"""Reindex imported openshift-docs Markdown sources locally.

Usage:
    python scripts/reindex_openshift_docs.py
    python scripts/reindex_openshift_docs.py --path installing
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
from scripts.reindex_official_pdfs import purge_legacy_index_entries


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("rag.reindex_openshift_docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="4.20", help="Target OCP version. Default: 4.20")
    parser.add_argument(
        "--input-root",
        default="data/corpus/pdfs",
        help="Base input root. Markdown sources are expected under ocp-<version>-openshift-docs.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Optional relative path inside the openshift-docs markdown root. Can be a file or subdirectory.",
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
    parser.add_argument(
        "--delete-missing",
        action="store_true",
        help="Delete stale indexed markdown documents under the openshift-docs root that no longer exist on disk.",
    )
    return parser.parse_args()


def markdown_root(input_root: Path, version: str) -> Path:
    return input_root / f"ocp-{version}-openshift-docs"


def _collect_markdown(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"openshift-docs markdown path not found: {path}")
    if path.is_file():
        if path.suffix.lower() != ".md":
            raise FileNotFoundError(f"Not a markdown file: {path}")
        return [path.resolve()]
    files = sorted(item.resolve() for item in path.rglob("*.md") if item.is_file())
    if not files:
        raise FileNotFoundError(f"No markdown files found under: {path}")
    return files


def collect_markdown_sources(root: Path, paths: list[str]) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"openshift-docs markdown root not found: {root}")
    targets = [str(path).strip() for path in paths if str(path).strip()]
    if not targets:
        return _collect_markdown(root)

    files: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved_target = (root / target).resolve()
        if resolved_target != root and root not in resolved_target.parents:
            raise ValueError(f"Path escapes openshift-docs markdown root: {target}")
        for file_path in _collect_markdown(resolved_target):
            if file_path not in seen:
                files.append(file_path)
                seen.add(file_path)
    return sorted(files)


def delete_missing_documents(index_repository, source_root: Path) -> int:
    deleted = 0
    source_root = source_root.resolve()
    for doc in index_repository.list_documents():
        source_path = Path(str(doc.get("source_path") or ""))
        try:
            resolved = source_path.resolve()
        except Exception:
            resolved = source_path
        if resolved == source_root or source_root in resolved.parents:
            if not resolved.exists():
                index_repository.delete_document(str(source_path))
                deleted += 1
    return deleted


def main() -> None:
    args = parse_args()
    settings = get_settings()
    container = build_container(settings)

    source_root = markdown_root(Path(args.input_root), args.version)
    source_files = collect_markdown_sources(source_root, args.path)
    logger.info(
        "openshift-docs reindex start: root=%s targets=%s file_count=%d",
        source_root,
        ",".join(args.path) if args.path else "all",
        len(source_files),
    )

    if args.delete_missing:
        deleted = delete_missing_documents(container.pipeline.index_repository, source_root)
        logger.info("deleted stale indexed markdown docs: %d", deleted)

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
        else:
            indexed_files += 1

    warmed_items = None
    if args.warm_cache:
        warmed_items = container.pipeline.index_repository.warm_cache()
        logger.info("index cache warmed: items=%d", warmed_items)

    logger.info(
        "openshift-docs reindex done: indexed_files=%d indexed_chunks=%d skipped_files=%d%s",
        indexed_files,
        indexed_chunks,
        skipped_files,
        f" warmed_items={warmed_items}" if warmed_items is not None else "",
    )


if __name__ == "__main__":
    main()

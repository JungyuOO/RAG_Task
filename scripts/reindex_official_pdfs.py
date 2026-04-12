"""Reindex official OCP PDFs locally after clearing extracted artifacts.

Usage:
    python scripts/reindex_official_pdfs.py
    python scripts/reindex_official_pdfs.py --version 4.20
    python scripts/reindex_official_pdfs.py --version 4.20 --clear-embedding-cache --warm-cache
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
from app.rag.utils import extracted_html_path, extracted_markdown_path, extracted_metadata_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("rag.reindex_official_pdfs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="append",
        default=[],
        help="Target OCP version such as 4.20. Can be passed multiple times. Default: all official OCP versions.",
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


def collect_official_pdfs(source_dir: Path, versions: list[str]) -> list[Path]:
    if versions:
        version_dirs = [source_dir / f"ocp-{version}" for version in versions]
    else:
        version_dirs = sorted(
            path for path in source_dir.iterdir()
            if path.is_dir() and path.name.startswith("ocp-")
        )

    pdfs: list[Path] = []
    for version_dir in version_dirs:
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")
        version_pdfs = sorted(
            path for path in version_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        if not version_pdfs:
            raise FileNotFoundError(f"No PDF files found under: {version_dir}")
        pdfs.extend(version_pdfs)
    return pdfs


def extracted_artifact_candidates(extract_dir: Path, source_path: Path) -> list[Path]:
    candidates: list[Path] = []
    patterns = (
        f"{source_path.stem}-????????.md",
        f"{source_path.stem}-????????.html",
        f"{source_path.stem}-????????.json",
    )
    for pattern in patterns:
        for path in extract_dir.glob(pattern):
            if path not in candidates:
                candidates.append(path)

    for candidate_source in (source_path, source_path.resolve()):
        for path in (
            extracted_markdown_path(extract_dir, candidate_source),
            extracted_html_path(extract_dir, candidate_source),
            extracted_metadata_path(extract_dir, candidate_source),
        ):
            if path not in candidates:
                candidates.append(path)
    return candidates


def delete_extracted_artifacts(extract_dir: Path, source_files: list[Path]) -> int:
    deleted = 0
    seen: set[Path] = set()
    for source_path in source_files:
        for artifact_path in extracted_artifact_candidates(extract_dir, source_path):
            if artifact_path in seen or not artifact_path.exists() or not artifact_path.is_file():
                continue
            artifact_path.unlink()
            seen.add(artifact_path)
            deleted += 1
    return deleted


def source_path_key(source_path: Path | str, source_root: Path) -> str:
    normalized = str(source_path).replace("\\", "/")
    source_root_norm = str(source_root).replace("\\", "/")
    try:
        return str(Path(source_path).relative_to(source_root)).replace("\\", "/")
    except Exception:
        pass
    for sep in (source_root_norm + "/", "pdfs/", "corpus/pdfs/"):
        if sep in normalized:
            return normalized.split(sep, 1)[-1]
    return normalized


def purge_legacy_index_entries(index_repository, source_root: Path, source_path: Path) -> int:
    target_key = source_path_key(source_path, source_root)
    canonical = str(source_path)
    deleted = 0
    for doc in index_repository.list_documents():
        stored = str(doc.get("source_path") or "")
        if not stored or stored == canonical:
            continue
        if source_path_key(stored, source_root) != target_key:
            continue
        index_repository.delete_document(stored)
        deleted += 1
    return deleted


def main() -> None:
    args = parse_args()
    versions = list(dict.fromkeys(str(version).strip() for version in args.version if str(version).strip()))
    settings = get_settings()
    container = build_container(settings)

    source_files = collect_official_pdfs(settings.rag_source_dir, versions)
    logger.info(
        "official reindex start: versions=%s file_count=%d",
        ",".join(versions) if versions else "all",
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
        "official reindex done: versions=%s indexed_files=%d indexed_chunks=%d skipped_files=%d%s",
        ",".join(versions) if versions else "all",
        indexed_files,
        indexed_chunks,
        skipped_files,
        f" warmed_items={warmed_items}" if warmed_items is not None else "",
    )


if __name__ == "__main__":
    main()

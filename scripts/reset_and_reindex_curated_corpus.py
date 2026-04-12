"""Reset caches + DB index, then reindex the curated corpus set.

Curated set:
- official: data/corpus/pdfs/ocp-html-single-4.20-en/*.md
- customer manuals: data/corpus/pdfs/generated/*.md + data/corpus/pdfs/generated_pdf/*.pdf

Usage:
    python scripts/reset_and_reindex_curated_corpus.py
    python scripts/reset_and_reindex_curated_corpus.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
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
logger = logging.getLogger("rag.reset_reindex_curated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-locale", default="en", choices=["en", "ko"])
    parser.add_argument("--version", default="4.20")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_curated_sources(source_root: Path, *, version: str, official_locale: str) -> list[Path]:
    official_dir = source_root / f"ocp-html-single-{version}-{official_locale}"
    if not official_dir.exists():
        raise FileNotFoundError(f"official html-single directory not found: {official_dir}")

    official_files = sorted(path.resolve() for path in official_dir.rglob("*.md") if path.is_file())
    customer_md = sorted(path.resolve() for path in (source_root / "generated").rglob("*.md") if path.is_file())
    customer_pdf = sorted(path.resolve() for path in (source_root / "generated_pdf").rglob("*.pdf") if path.is_file())
    return official_files + customer_md + customer_pdf


def clear_json_cache_dir(path: Path) -> int:
    if not path.exists():
        return 0
    deleted = 0
    for file_path in path.rglob("*.json"):
        try:
            file_path.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted


def main() -> None:
    args = parse_args()
    settings = get_settings()
    container = build_container(settings)
    source_files = collect_curated_sources(
        settings.rag_source_dir,
        version=str(args.version),
        official_locale=str(args.official_locale),
    )

    logger.info(
        "curated corpus selection: official=%d customer_md=%d customer_pdf=%d total=%d",
        sum(1 for path in source_files if f"ocp-html-single-{args.version}-{args.official_locale}" in str(path).replace("\\", "/")),
        sum(1 for path in source_files if "/generated/" in str(path).replace("\\", "/")),
        sum(1 for path in source_files if "/generated_pdf/" in str(path).replace("\\", "/")),
        len(source_files),
    )

    if args.dry_run:
        for path in source_files:
            print(path)
        return

    started = time.perf_counter()
    embed_cache_deleted = clear_json_cache_dir(settings.rag_cache_dir / "embeddings")
    answer_cache_deleted = clear_json_cache_dir(settings.rag_cache_dir / "answers")
    container.pipeline.index_repository.clear_cache()
    container.pipeline.answer_cache_repository.clear()
    container.pipeline.embedding_cache_repository.clear()
    logger.info(
        "cache clear complete: embedding_files=%d answer_files=%d",
        embed_cache_deleted,
        answer_cache_deleted,
    )

    reset_started = time.perf_counter()
    container.pipeline.index_repository.save([], [])
    logger.info("db index reset complete: %.2fs", time.perf_counter() - reset_started)

    result = container.indexing_service.rebuild_index(source_files)
    warmed_items = container.pipeline.index_repository.warm_cache()

    logger.info(
        "curated reindex done: indexed_files=%d indexed_chunks=%d skipped_files=%d warmed_items=%d total=%.2fs",
        int(result.get("indexed_files", 0)),
        int(result.get("indexed_chunks", 0)),
        int(result.get("skipped_files", 0)),
        warmed_items,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()

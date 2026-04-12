"""Delete legacy official corpus files after html-single cutover.

By default this performs a dry run. Pass --apply to delete files and index entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.dependencies import build_container
from app.rag.utils import extracted_html_candidates, extracted_markdown_candidates, extracted_metadata_candidates


KEEP_DIR_MARKERS = ("generated", "generated_pdf", "chat_uploads")


def is_html_single_source(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return "/ocp-html-single-" in normalized and path.suffix.lower() == ".md"


def is_customer_source(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return any(f"/{marker}/" in normalized for marker in KEEP_DIR_MARKERS)


def is_legacy_official_source(path: Path) -> bool:
    if not path.is_file():
        return False
    if is_customer_source(path) or is_html_single_source(path):
        return False
    normalized = str(path).replace("\\", "/").lower()
    if path.suffix.lower() == ".pdf":
        return True
    if path.suffix.lower() == ".md" and ("-openshift-docs/" in normalized or "/ocp-" in normalized):
        return True
    return False


def is_legacy_official_source_ref(source_path: str | Path) -> bool:
    path = Path(str(source_path))
    normalized = str(path).replace("\\", "/").lower()
    if any(f"/{marker}/" in normalized for marker in KEEP_DIR_MARKERS):
        return False
    if "/ocp-html-single-" in normalized and normalized.endswith(".md"):
        return False
    if normalized.endswith(".pdf"):
        return True
    if normalized.endswith(".md") and ("ocp-4.20-openshift-docs/" in normalized or "/ocp-" in normalized):
        return True
    return False


def collect_legacy_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_legacy_official_source(path))


def collect_legacy_index_sources(index_repository) -> list[str]:
    legacy: list[str] = []
    for doc in index_repository.list_documents():
        source_path = str(doc.get("source_path") or "")
        if is_legacy_official_source_ref(source_path):
            legacy.append(source_path)
    return sorted(dict.fromkeys(legacy))


def collect_extracted_artifacts(extract_root: Path, source_path: Path) -> list[Path]:
    artifacts: list[Path] = []
    for candidate in [*extracted_markdown_candidates(extract_root, source_path), *extracted_html_candidates(extract_root, source_path), *extracted_metadata_candidates(extract_root, source_path)]:
        if candidate.exists() and candidate not in artifacts:
            artifacts.append(candidate)
    return artifacts


def delete_legacy_corpus(*, source_root: Path, extract_root: Path, apply: bool) -> dict:
    legacy_sources = collect_legacy_sources(source_root)
    deleted_sources: list[str] = []
    deleted_artifacts: list[str] = []
    deleted_index_docs: list[str] = []

    container = None
    if apply:
        settings = get_settings()
        container = build_container(settings)
    elif legacy_sources:
        settings = get_settings()
        container = build_container(settings)

    legacy_index_sources: list[str] = []
    if container is not None:
        legacy_index_sources = collect_legacy_index_sources(container.pipeline.index_repository)
        if not apply:
            for source_path in legacy_index_sources:
                if source_path not in [str(path) for path in legacy_sources]:
                    deleted_index_docs.append(source_path)

    for source_path in legacy_sources:
        deleted_index_docs.append(str(source_path))
        if apply:
            assert container is not None
            container.pipeline.index_repository.delete_document(str(source_path))
            for artifact in collect_extracted_artifacts(extract_root, source_path):
                artifact.unlink(missing_ok=True)
                deleted_artifacts.append(str(artifact))
            source_path.unlink(missing_ok=True)
            deleted_sources.append(str(source_path))

    if apply and container is not None:
        filesystem_sources = {str(path) for path in legacy_sources}
        for source_path in legacy_index_sources:
            if source_path in filesystem_sources:
                continue
            container.pipeline.index_repository.delete_document(source_path)
            deleted_index_docs.append(source_path)

    return {
        "apply": apply,
        "legacy_source_count": len(legacy_sources),
        "legacy_sources": [str(path) for path in legacy_sources],
        "deleted_sources": deleted_sources,
        "deleted_artifacts": deleted_artifacts,
        "deleted_index_docs": sorted(dict.fromkeys(deleted_index_docs if apply else deleted_index_docs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="")
    parser.add_argument("--extract-root", default="")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    result = delete_legacy_corpus(
        source_root=Path(args.source_root) if args.source_root else settings.rag_source_dir,
        extract_root=Path(args.extract_root) if args.extract_root else settings.rag_extract_dir,
        apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

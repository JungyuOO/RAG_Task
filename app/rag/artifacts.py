from __future__ import annotations

from pathlib import Path

from app.rag.utils import stable_hash


def extracted_markdown_file_name(source_path: Path) -> str:
    return f"{source_path.stem}-{stable_hash(str(source_path))[:8]}.md"


def extracted_markdown_path(extract_dir: Path, source_path: Path) -> Path:
    return extract_dir / extracted_markdown_file_name(source_path)


def extracted_markdown_candidates(extract_dir: Path, source_path: Path) -> list[Path]:
    # First try a stem-based glob so the file can still be found even if the
    # original path string changed across environments such as Docker or local runs.
    stem = source_path.stem
    glob_matches = list(extract_dir.glob(f"{stem}-????????.md"))
    if glob_matches:
        return glob_matches

    # If no exported markdown exists yet, fall back to the current path-based candidates.
    candidates: list[Path] = []
    for candidate_source in (source_path, source_path.resolve()):
        candidate_path = extracted_markdown_path(extract_dir, candidate_source)
        if candidate_path not in candidates:
            candidates.append(candidate_path)
    return candidates

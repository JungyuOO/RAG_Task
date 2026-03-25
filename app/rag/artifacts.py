from __future__ import annotations

from pathlib import Path

from app.rag.utils import stable_hash


def extracted_markdown_file_name(source_path: Path) -> str:
    return f"{source_path.stem}-{stable_hash(str(source_path))[:8]}.md"


def extracted_markdown_path(extract_dir: Path, source_path: Path) -> Path:
    return extract_dir / extracted_markdown_file_name(source_path)


def extracted_markdown_candidates(extract_dir: Path, source_path: Path) -> list[Path]:
    # 파일명 stem 기반 glob으로 후보를 수집한다.
    # 해시는 source_path 절대/상대 경로에 따라 달라지므로,
    # 경로가 바뀌어도 (Docker ↔ 로컬 전환 등) 올바르게 삭제되도록
    # "{stem}-????????.md" 패턴으로 실제 존재하는 파일을 먼저 찾는다.
    stem = source_path.stem
    glob_matches = list(extract_dir.glob(f"{stem}-????????.md"))
    if glob_matches:
        return glob_matches

    # glob 결과가 없으면 (아직 미생성 등) 기존 해시 기반 경로를 후보로 반환.
    candidates: list[Path] = []
    for candidate_source in (source_path, source_path.resolve()):
        candidate_path = extracted_markdown_path(extract_dir, candidate_source)
        if candidate_path not in candidates:
            candidates.append(candidate_path)
    return candidates

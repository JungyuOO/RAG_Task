from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

from app.dependencies import AppContainer
from app.rag.utils import extracted_markdown_candidates

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


logger = logging.getLogger("rag.api")
OWNER_HEADER_NAME = "X-Client-Id"


def resolve_owner_id(request: Request) -> str:
    owner_id = (request.headers.get(OWNER_HEADER_NAME) or "").strip()
    if not owner_id:
        raise HTTPException(status_code=400, detail=f"Missing {OWNER_HEADER_NAME} header.")
    if len(owner_id) > 120:
        raise HTTPException(status_code=400, detail="Invalid client identifier.")
    return owner_id


def scoped_session_id(owner_id: str, session_id: str) -> str:
    return f"{owner_id}::{session_id}"


def unscoped_session_id(owner_id: str, session_id: str) -> str:
    prefix = f"{owner_id}::"
    return session_id[len(prefix):] if session_id.startswith(prefix) else session_id


def resolve_library_pdf(settings, file_name: str) -> Path:
    root_path = settings.rag_source_dir.resolve()
    base_name = Path(file_name).name
    if base_name != file_name or not base_name:
        raise HTTPException(status_code=400, detail="Invalid file path.")

    # 최상위 경로 우선, 없으면 하위 폴더(ocp-x.y/…) 재귀 탐색
    direct = (settings.rag_source_dir / base_name).resolve()
    candidates: list[Path] = []
    if direct.exists() and direct.is_file():
        candidates.append(direct)
    else:
        candidates.extend(
            path for path in settings.rag_source_dir.rglob(base_name)
            if path.is_file()
        )

    for target_path in candidates:
        resolved = target_path.resolve()
        if root_path not in resolved.parents and resolved != root_path:
            continue
        if resolved.suffix.lower() not in {".pdf", ".md"}:
            continue
        return resolved

    raise HTTPException(status_code=404, detail="File not found.")


async def save_library_uploads(settings, files: list[UploadFile]) -> list[str]:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    uploaded_files: list[str] = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".md"}:
            raise HTTPException(status_code=400, detail="Only PDF and Markdown files are allowed.")
        # MD 파일은 generated/ 하위에 저장 (고객사 메뉴얼 분류 유지)
        if suffix == ".md":
            target_dir = settings.rag_source_dir / "generated"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / Path(file.filename).name
            uploaded_files.append(str(Path("generated") / Path(file.filename).name))
        else:
            target_path = settings.rag_source_dir / file.filename
            uploaded_files.append(file.filename)
        contents = await file.read()
        target_path.write_bytes(contents)
    return uploaded_files


def list_source_pdfs(settings) -> list[Path]:
    return [path for path in settings.rag_source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"]


def delete_markdown_artifacts(container: AppContainer, file_name: str) -> tuple[Path, bool]:
    target_path = resolve_library_pdf(container.settings, file_name)
    relative_source_path = container.settings.rag_source_dir / file_name
    markdown_paths = extracted_markdown_candidates(container.settings.rag_extract_dir, relative_source_path)

    target_path.unlink()
    deleted_markdown = False
    for markdown_path in markdown_paths:
        if markdown_path.exists() and markdown_path.is_file():
            markdown_path.unlink()
            deleted_markdown = True
    return target_path, deleted_markdown

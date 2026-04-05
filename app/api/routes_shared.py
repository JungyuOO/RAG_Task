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
    target_path = (settings.rag_source_dir / file_name).resolve()
    root_path = settings.rag_source_dir.resolve()
    if root_path not in target_path.parents and target_path != root_path:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not target_path.exists() or not target_path.is_file() or target_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF file not found.")
    return target_path


async def save_library_uploads(settings, files: list[UploadFile]) -> list[str]:
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files were uploaded.")

    uploaded_files: list[str] = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        target_path = settings.rag_source_dir / file.filename
        contents = await file.read()
        target_path.write_bytes(contents)
        uploaded_files.append(file.filename)
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

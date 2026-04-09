from __future__ import annotations

import logging
from pathlib import Path
import re

from fastapi import HTTPException, Request, UploadFile

from app.dependencies import AppContainer
from app.rag.utils import extracted_markdown_candidates

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


logger = logging.getLogger("rag.api")
OWNER_HEADER_NAME = "X-Client-Id"
CHAT_UPLOAD_DIR_NAME = "chat_uploads"


def customer_generated_target_dir(settings, suffix: str) -> Path:
    return settings.rag_source_dir / ("generated_pdf" if suffix == ".pdf" else "generated")


def _sanitize_upload_scope(scope: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(scope or "").strip())
    return normalized[:80] or "default"


def chat_upload_target_dir(settings, session_id: str) -> Path:
    return settings.rag_source_dir / CHAT_UPLOAD_DIR_NAME / _sanitize_upload_scope(session_id)


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


async def save_library_uploads(
    settings,
    files: list[UploadFile],
    *,
    target_group: str | None = None,
    target_version: str | None = None,
) -> list[str]:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    normalized_group = (target_group or "").strip()
    normalized_version = (target_version or "").strip()
    uploaded_files: list[str] = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".md"}:
            raise HTTPException(status_code=400, detail="Only PDF and Markdown files are allowed.")

        if normalized_group:
            if normalized_group not in {"official_ocp", "customer_generated"}:
                raise HTTPException(status_code=400, detail="Invalid upload target group.")
            if normalized_group == "official_ocp":
                if not re.fullmatch(r"4\.\d+", normalized_version):
                    raise HTTPException(status_code=400, detail="A target version like 4.15 is required.")
                target_dir = settings.rag_source_dir / f"ocp-{normalized_version}"
                relative_path = Path(f"ocp-{normalized_version}") / Path(file.filename).name
            else:
                target_dir = customer_generated_target_dir(settings, suffix)
                relative_path = Path(target_dir.name) / Path(file.filename).name

            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / Path(file.filename).name
            uploaded_files.append(relative_path.as_posix())
        else:
            if suffix == ".md":
                target_dir = customer_generated_target_dir(settings, suffix)
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / Path(file.filename).name
                uploaded_files.append((Path(target_dir.name) / Path(file.filename).name).as_posix())
            else:
                target_path = settings.rag_source_dir / file.filename
                uploaded_files.append(file.filename)

        contents = await file.read()
        target_path.write_bytes(contents)

    return uploaded_files


async def save_chat_uploads(
    settings,
    session_id: str,
    files: list[UploadFile],
) -> list[str]:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    target_dir = chat_upload_target_dir(settings, session_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files: list[str] = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed in chat uploads.")
        target_path = target_dir / Path(file.filename).name
        contents = await file.read()
        target_path.write_bytes(contents)
        uploaded_files.append(str(target_path.relative_to(settings.rag_source_dir)).replace("\\", "/"))

    return uploaded_files


def list_source_pdfs(settings) -> list[Path]:
    chat_upload_root = (settings.rag_source_dir / CHAT_UPLOAD_DIR_NAME).resolve()
    results: list[Path] = []
    for path in settings.rag_source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        resolved = path.resolve()
        if resolved == chat_upload_root or chat_upload_root in resolved.parents:
            continue
        results.append(path)
    return results


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

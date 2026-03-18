from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.api.schemas import (
    BuildIndexResponse,
    ChatRequest,
    DeleteLibraryResponse,
    LibraryStatusResponse,
    RetryChatRequest,
    SessionHistoryResponse,
    UploadLibraryResponse,
)
from app.config import get_settings
from app.rag.artifacts import extracted_markdown_candidates
from app.rag.pipeline import RagPipeline

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


router = APIRouter()
settings = get_settings()
pipeline = RagPipeline(settings)


@router.get("/")
async def index() -> FileResponse:
    return FileResponse("app/web/index.html")


@router.get("/api/library")
async def get_library() -> LibraryStatusResponse:
    return LibraryStatusResponse(**pipeline.list_library_documents())


def resolve_library_pdf(file_name: str) -> Path:
    target_path = (settings.rag_source_dir / file_name).resolve()
    root_path = settings.rag_source_dir.resolve()
    if root_path not in target_path.parents and target_path != root_path:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not target_path.exists() or not target_path.is_file() or target_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF file not found.")
    return target_path


async def save_library_uploads(files: list[UploadFile]) -> list[str]:
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


def list_source_pdfs() -> list[Path]:
    return [
        path
        for path in settings.rag_source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]


@router.get("/api/library/preview")
async def preview_library_file(file_name: str = Query(..., min_length=1)) -> FileResponse:
    target_path = resolve_library_pdf(file_name)
    return FileResponse(
        target_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@router.get("/api/library/page-image")
async def preview_library_page_image(
    file_name: str = Query(..., min_length=1),
    page: int = Query(..., ge=1),
) -> Response:
    if fitz is None:
        raise HTTPException(status_code=500, detail="PyMuPDF is required to render PDF pages.")

    target_path = resolve_library_pdf(file_name)
    pdf = fitz.open(target_path)
    try:
        if page > pdf.page_count:
            raise HTTPException(status_code=404, detail="PDF page not found.")
        pdf_page = pdf.load_page(page - 1)
        pix = pdf_page.get_pixmap(dpi=170, alpha=False)
        return Response(content=pix.tobytes("png"), media_type="image/png")
    finally:
        pdf.close()


@router.get("/api/library/download")
async def download_library_file(file_name: str = Query(..., min_length=1)) -> FileResponse:
    target_path = resolve_library_pdf(file_name)
    return FileResponse(target_path, media_type="application/pdf", filename=target_path.name)


@router.delete("/api/library")
async def delete_library_file(file_name: str = Query(..., min_length=1)) -> DeleteLibraryResponse:
    target_path = resolve_library_pdf(file_name)
    relative_source_path = settings.rag_source_dir / file_name
    markdown_paths = extracted_markdown_candidates(settings.rag_extract_dir, relative_source_path)

    target_path.unlink()
    deleted_markdown = False
    for markdown_path in markdown_paths:
        if markdown_path.exists() and markdown_path.is_file():
            markdown_path.unlink()
            deleted_markdown = True

    result = pipeline.delete_library_document(target_path)
    return DeleteLibraryResponse(
        deleted_file=target_path.name,
        deleted_markdown=deleted_markdown,
        **result,
    )


@router.post("/api/library/upload")
async def upload_to_library(files: list[UploadFile] = File(...)) -> UploadLibraryResponse:
    uploaded_files = await save_library_uploads(files)
    source_files = list_source_pdfs()
    result = pipeline.rebuild_index(source_files)
    return UploadLibraryResponse(uploaded_files=uploaded_files, **result)


@router.post("/api/reindex")
async def reindex_all() -> BuildIndexResponse:
    source_files = list_source_pdfs()
    result = pipeline.rebuild_index(source_files)
    return BuildIndexResponse(**result)


@router.get("/api/sessions")
async def list_sessions() -> SessionHistoryResponse:
    return SessionHistoryResponse(sessions=pipeline.session_store.list_sessions())


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    return pipeline.session_store.export_session(session_id)


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    deleted = pipeline.session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True, "session_id": session_id}


@router.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        async for event in pipeline.stream_chat(session_id=request.session_id, user_message=request.message):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/chat/retry")
async def retry_chat(request: RetryChatRequest) -> StreamingResponse:
    requested_message = (request.message or "").strip()
    pending_message = pipeline.session_store.pending_user_message(request.session_id)
    user_message = pending_message or requested_message
    if not user_message:
        raise HTTPException(status_code=404, detail="No pending user message found for retry.")

    append_user_turn = True
    if pending_message and (not requested_message or pending_message == requested_message):
        append_user_turn = False

    allowed_source_paths: set[str] | None = None
    if request.file_names:
        allowed_source_paths = {
            str(resolve_library_pdf(file_name).resolve())
            for file_name in request.file_names
        }

    async def event_stream():
        async for event in pipeline.stream_chat(
            session_id=request.session_id,
            user_message=user_message,
            allowed_source_paths=allowed_source_paths,
            append_user_turn=append_user_turn,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/chat/upload")
async def chat_with_upload(
    session_id: str = Form(...),
    message: str = Form(...),
    files: list[UploadFile] = File(...),
) -> StreamingResponse:
    async def event_stream():
        uploaded_files = await save_library_uploads(files)
        result = pipeline.rebuild_index(list_source_pdfs())
        uploaded_source_paths = {str((settings.rag_source_dir / file_name).resolve()) for file_name in uploaded_files}
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "upload",
                    "uploaded_files": uploaded_files,
                    "indexed_files": result["indexed_files"],
                    "indexed_chunks": result["indexed_chunks"],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        async for event in pipeline.stream_chat(
            session_id=session_id,
            user_message=message,
            allowed_source_paths=uploaded_source_paths,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

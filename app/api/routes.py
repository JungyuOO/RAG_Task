from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.api.schemas import (
    BuildIndexResponse,
    ChatRequest,
    DeleteLibraryResponse,
    LibraryStatusResponse,
    RetrievalDebugRequest,
    RetryChatRequest,
    SessionHistoryResponse,
    TaskStatusResponse,
)
from app.dependencies import AppContainer, get_container
from app.rag.artifacts import extracted_markdown_candidates

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


router = APIRouter()


@router.get("/")
async def index() -> FileResponse:
    return FileResponse("app/web/index.html")


@router.get("/api/library")
async def get_library(request: Request, container: AppContainer = Depends(get_container)) -> LibraryStatusResponse:
    data = container.pipeline.list_library_documents()
    startup_state = getattr(request.app.state, "startup_indexing", None)
    if startup_state:
        data["startup_indexing"] = startup_state
    return LibraryStatusResponse(**data)


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
    return [
        path
        for path in settings.rag_source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]


@router.get("/api/library/preview")
async def preview_library_file(
    file_name: str = Query(..., min_length=1),
    container: AppContainer = Depends(get_container),
) -> FileResponse:
    target_path = resolve_library_pdf(container.settings, file_name)
    return FileResponse(
        target_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@router.get("/api/library/page-image")
async def preview_library_page_image(
    file_name: str = Query(..., min_length=1),
    page: int = Query(..., ge=1),
    container: AppContainer = Depends(get_container),
) -> Response:
    if fitz is None:
        raise HTTPException(status_code=500, detail="PyMuPDF is required to render PDF pages.")

    target_path = resolve_library_pdf(container.settings, file_name)
    pdf = fitz.open(target_path)
    try:
        if page > pdf.page_count:
            raise HTTPException(status_code=404, detail="PDF page not found.")
        pdf_page = pdf.load_page(page - 1)
        pix = pdf_page.get_pixmap(dpi=container.settings.pdf_render_dpi, alpha=False)
        return Response(content=pix.tobytes("png"), media_type="image/png")
    finally:
        pdf.close()


@router.get("/api/library/download")
async def download_library_file(
    file_name: str = Query(..., min_length=1),
    container: AppContainer = Depends(get_container),
) -> FileResponse:
    target_path = resolve_library_pdf(container.settings, file_name)
    return FileResponse(target_path, media_type="application/pdf", filename=target_path.name)


@router.delete("/api/library")
async def delete_library_file(
    file_name: str = Query(..., min_length=1),
    container: AppContainer = Depends(get_container),
) -> DeleteLibraryResponse:
    target_path = resolve_library_pdf(container.settings, file_name)
    relative_source_path = container.settings.rag_source_dir / file_name
    markdown_paths = extracted_markdown_candidates(container.settings.rag_extract_dir, relative_source_path)

    target_path.unlink()
    deleted_markdown = False
    for markdown_path in markdown_paths:
        if markdown_path.exists() and markdown_path.is_file():
            markdown_path.unlink()
            deleted_markdown = True

    result = container.pipeline.delete_library_document(target_path)
    return DeleteLibraryResponse(
        deleted_file=target_path.name,
        deleted_markdown=deleted_markdown,
        **result,
    )


@router.post("/api/library/upload")
async def upload_to_library(
    files: list[UploadFile] = File(...),
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    uploaded_files = await save_library_uploads(container.settings, files)
    total_files = len(uploaded_files)

    async def event_stream():
        loop = asyncio.get_running_loop()
        total_chunks = 0

        for file_idx, file_name in enumerate(uploaded_files):
            source_path = container.settings.rag_source_dir / file_name
            queue: asyncio.Queue = asyncio.Queue()

            def make_progress_callback(q, ev_loop):
                def callback(stage, current, total):
                    # extract: 0~70%, embed: 70~95%
                    if stage == "extract":
                        pct = int(current / total * 70)
                    else:
                        pct = 70 + int(current / total * 25)
                    ev_loop.call_soon_threadsafe(q.put_nowait, {"type": "progress", "file": file_name, "pct": pct})
                return callback

            progress_cb = make_progress_callback(queue, loop)

            # 스레드풀에서 인덱싱 실행 (진행률 콜백은 queue에 이벤트 적재)
            index_task = asyncio.ensure_future(
                run_in_threadpool(container.pipeline.index_single_file, source_path, progress_cb)
            )

            # 인덱싱 완료까지 queue에서 progress 이벤트를 소비하며 SSE 전송
            while not index_task.done():
                try:
                    event = queue.get_nowait()
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)

            # 남은 queue 이벤트 flush
            while not queue.empty():
                event = queue.get_nowait()
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

            try:
                result = index_task.result()
            except Exception as exc:
                yield "data: " + json.dumps({"type": "file_error", "file": file_name, "error": str(exc)}, ensure_ascii=False) + "\n\n"
                continue

            total_chunks += result.get("indexed_chunks", 0)
            library = container.pipeline.list_library_documents()
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "file_indexed",
                        "file": file_name,
                        "file_idx": file_idx,
                        "total_files": total_files,
                        **result,
                        "library": library,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        yield "data: " + json.dumps({"type": "done", "total_chunks": total_chunks}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/reindex")
async def reindex_all(container: AppContainer = Depends(get_container)) -> BuildIndexResponse:
    source_files = list_source_pdfs(container.settings)
    result = await run_in_threadpool(
        container.task_service.run_inline,
        "full_reindex",
        {"source_count": len(source_files)},
        lambda: container.pipeline.rebuild_index(source_files),
    )
    return BuildIndexResponse(**result)


@router.get("/api/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    container: AppContainer = Depends(get_container),
) -> TaskStatusResponse:
    task = container.task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(**task)


@router.get("/api/sessions")
async def list_sessions(container: AppContainer = Depends(get_container)) -> SessionHistoryResponse:
    return SessionHistoryResponse(sessions=container.pipeline.session_repository.list_sessions())


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, container: AppContainer = Depends(get_container)) -> dict:
    return container.pipeline.session_repository.export_session(session_id)


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, container: AppContainer = Depends(get_container)) -> dict:
    deleted = container.pipeline.session_repository.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True, "session_id": session_id}


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    async def event_stream():
        async for event in container.pipeline.stream_chat(
            session_id=request.session_id,
            user_message=request.message,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/debug/retrieval")
async def debug_retrieval(
    request: RetrievalDebugRequest,
    container: AppContainer = Depends(get_container),
) -> dict:
    allowed_source_paths: set[str] | None = None
    if request.file_names:
        allowed_source_paths = {
            str(resolve_library_pdf(container.settings, file_name).resolve())
            for file_name in request.file_names
        }
    return await container.pipeline.inspect_retrieval(
        session_id=request.session_id,
        user_message=request.message,
        allowed_source_paths=allowed_source_paths,
    )


@router.post("/api/chat/retry")
async def retry_chat(
    request: RetryChatRequest,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    requested_message = (request.message or "").strip()
    pending_message = container.pipeline.session_repository.pending_user_message(request.session_id)
    user_message = pending_message or requested_message
    if not user_message:
        raise HTTPException(status_code=404, detail="No pending user message found for retry.")

    append_user_turn = True
    if pending_message and (not requested_message or pending_message == requested_message):
        append_user_turn = False

    allowed_source_paths: set[str] | None = None
    if request.file_names:
        allowed_source_paths = {
            str(resolve_library_pdf(container.settings, file_name).resolve())
            for file_name in request.file_names
        }

    async def event_stream():
        async for event in container.pipeline.stream_chat(
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
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    async def event_stream():
        uploaded_files = await save_library_uploads(container.settings, files)
        total_chunks = 0
        for file_name in uploaded_files:
            source_path = container.settings.rag_source_dir / file_name
            result = await run_in_threadpool(container.pipeline.index_single_file, source_path)
            total_chunks += result.get("indexed_chunks", 0)
        uploaded_source_paths = {
            str((container.settings.rag_source_dir / file_name).resolve())
            for file_name in uploaded_files
        }
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "upload",
                    "uploaded_files": uploaded_files,
                    "indexed_chunks": total_chunks,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        async for event in container.pipeline.stream_chat(
            session_id=session_id,
            user_message=message,
            allowed_source_paths=uploaded_source_paths,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

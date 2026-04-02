from __future__ import annotations

import asyncio
import json
import logging
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
from app.rag.chat_service import ChatTurnRequest, RetryChatRequestModel
from app.rag.artifacts import extracted_markdown_candidates

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


router = APIRouter()
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


@router.get("/")
async def index() -> FileResponse:
    return FileResponse("app/web/index.html")


@router.get("/api/library")
async def get_library(request: Request, container: AppContainer = Depends(get_container)) -> LibraryStatusResponse:
    data = container.indexing_service.list_library_documents()
    startup_state = getattr(request.app.state, "startup_indexing", None)
    reindex_state = getattr(request.app.state, "reindexing", None)
    if startup_state:
        data["startup_indexing"] = startup_state
    if reindex_state:
        data["reindexing"] = reindex_state
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

    result = container.indexing_service.delete_library_document(target_path)
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
                def callback(stage, current, total, meta=None):
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
                run_in_threadpool(container.indexing_service.index_single_file, source_path, progress_cb)
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
            library = container.indexing_service.list_library_documents()
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
async def reindex_all(request: Request, container: AppContainer = Depends(get_container)) -> BuildIndexResponse:
    startup_state = getattr(request.app.state, "startup_indexing", None) or {}
    reindex_state = getattr(request.app.state, "reindexing", None) or {}
    if startup_state.get("status") == "indexing":
        raise HTTPException(status_code=409, detail="자동 인덱싱이 진행 중입니다. 완료 후 다시 시도해 주세요.")
    if reindex_state.get("status") == "indexing":
        raise HTTPException(status_code=409, detail="재인덱싱이 이미 진행 중입니다.")

    source_files = list_source_pdfs(container.settings)
    file_positions = {str(path): index for index, path in enumerate(source_files)}
    reindex_state.update(
        status="indexing",
        total_files=len(source_files),
        completed_files=0,
        current_file="",
        current_stage="prepare",
        current_chunk=0,
        total_chunks=0,
        progress_pct=0,
    )

    def progress_callback(stage, current, total, meta=None):
        meta = meta or {}
        source_path = str(meta.get("source_path", ""))
        reindex_state.update(
            current_file=meta.get("file_name", ""),
            completed_files=file_positions.get(source_path, 0),
            current_stage=stage,
            current_chunk=current,
            total_chunks=total,
            progress_pct=min(100, int((current / max(total, 1)) * 100)),
        )

    try:
        result = await run_in_threadpool(container.indexing_service.rebuild_index, source_files, progress_callback)
    except Exception:
        reindex_state.update(status="idle", current_stage="error")
        raise

    reindex_state.update(
        status="done",
        completed_files=len(source_files),
        current_file="",
        current_stage="done",
        current_chunk=result.get("indexed_chunks", 0),
        total_chunks=result.get("indexed_chunks", 0),
        progress_pct=100,
    )
    return BuildIndexResponse(**result)


@router.get("/api/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    container: AppContainer = Depends(get_container),
) -> TaskStatusResponse:
    task = container.task_repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(**task)


@router.get("/api/sessions")
async def list_sessions(request: Request, container: AppContainer = Depends(get_container)) -> SessionHistoryResponse:
    owner_id = resolve_owner_id(request)
    logger.info("[SessionList] owner_id=%s", owner_id)
    sessions = container.session_repository.list_sessions(owner_id=owner_id)
    sanitized = [
        {
            **item,
            "session_id": unscoped_session_id(owner_id, str(item.get("session_id", ""))),
        }
        for item in sessions
    ]
    return SessionHistoryResponse(sessions=sanitized)


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request, container: AppContainer = Depends(get_container)) -> dict:
    owner_id = resolve_owner_id(request)
    logger.info("[SessionLoad] owner_id=%s session_id=%s", owner_id, session_id)
    exported = container.session_repository.export_session(scoped_session_id(owner_id, session_id), owner_id)
    if not exported:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        **exported,
        "session_id": session_id,
    }


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request, container: AppContainer = Depends(get_container)) -> dict:
    owner_id = resolve_owner_id(request)
    logger.info("[SessionDelete] owner_id=%s session_id=%s", owner_id, session_id)
    deleted = container.session_repository.delete_session(scoped_session_id(owner_id, session_id), owner_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True, "session_id": session_id}


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    owner_id = resolve_owner_id(http_request)
    effective_session_id = scoped_session_id(owner_id, request.session_id)
    logger.info("[Chat] owner_id=%s session_id=%s", owner_id, request.session_id)

    async def event_stream():
        chat_request = ChatTurnRequest(
            session_id=effective_session_id,
            message=request.message,
        )
        async for event in container.chat_service.stream(chat_request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/debug/retrieval")
async def debug_retrieval(
    request: RetrievalDebugRequest,
    http_request: Request,
    container: AppContainer = Depends(get_container),
) -> dict:
    owner_id = resolve_owner_id(http_request)
    allowed_source_paths: set[str] | None = None
    if request.file_names:
        allowed_source_paths = {
            str(resolve_library_pdf(container.settings, file_name).resolve())
            for file_name in request.file_names
        }
    return await container.pipeline.inspect_retrieval(
        session_id=scoped_session_id(owner_id, request.session_id),
        user_message=request.message,
        allowed_source_paths=allowed_source_paths,
    )


@router.post("/api/chat/retry")
async def retry_chat(
    request: RetryChatRequest,
    http_request: Request,
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    owner_id = resolve_owner_id(http_request)
    effective_session_id = scoped_session_id(owner_id, request.session_id)
    logger.info("[ChatRetry] owner_id=%s session_id=%s", owner_id, request.session_id)

    allowed_source_paths: set[str] | None = None
    if request.file_names:
        allowed_source_paths = {
            str(resolve_library_pdf(container.settings, file_name).resolve())
            for file_name in request.file_names
        }

    async def event_stream():
        retry_request = RetryChatRequestModel(
            session_id=effective_session_id,
            message=request.message or "",
            owner_id=owner_id,
            allowed_source_paths=allowed_source_paths,
            append_user_turn=True,
        )
        try:
            async for event in container.chat_service.retry(retry_request):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/chat/upload")
async def chat_with_upload(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(...),
    files: list[UploadFile] = File(...),
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    owner_id = resolve_owner_id(request)
    effective_session_id = scoped_session_id(owner_id, session_id)
    logger.info("[ChatUpload] owner_id=%s session_id=%s file_count=%d", owner_id, session_id, len(files))

    async def event_stream():
        uploaded_files = await save_library_uploads(container.settings, files)
        total_chunks = 0
        for file_name in uploaded_files:
            source_path = container.settings.rag_source_dir / file_name
            result = await run_in_threadpool(container.indexing_service.index_single_file, source_path)
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
            session_id=effective_session_id,
            user_message=message,
            allowed_source_paths=uploaded_source_paths,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

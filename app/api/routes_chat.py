from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.api.routes_shared import logger, resolve_library_pdf, resolve_owner_id, save_library_uploads, scoped_session_id
from app.api.schemas import ChatRequest, ChatTurnRequest, RetrievalDebugRequest, RetryChatRequest, RetryChatRequestModel
from app.dependencies import AppContainer, get_container

router = APIRouter()


@router.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request, container: AppContainer = Depends(get_container)) -> StreamingResponse:
    owner_id = resolve_owner_id(http_request)
    effective_session_id = scoped_session_id(owner_id, request.session_id)
    logger.info("[Chat] owner_id=%s session_id=%s", owner_id, request.session_id)

    async def event_stream():
        chat_request = ChatTurnRequest(session_id=effective_session_id, message=request.message)
        async for event in container.chat_service.stream(chat_request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/debug/retrieval")
async def debug_retrieval(request: RetrievalDebugRequest, http_request: Request, container: AppContainer = Depends(get_container)) -> dict:
    owner_id = resolve_owner_id(http_request)
    allowed_source_paths: set[str] | None = None
    if request.file_names:
        allowed_source_paths = {str(resolve_library_pdf(container.settings, file_name).resolve()) for file_name in request.file_names}
    return await container.pipeline.inspect_retrieval(
        session_id=scoped_session_id(owner_id, request.session_id),
        user_message=request.message,
        allowed_source_paths=allowed_source_paths,
    )


@router.post("/api/chat/retry")
async def retry_chat(request: RetryChatRequest, http_request: Request, container: AppContainer = Depends(get_container)) -> StreamingResponse:
    owner_id = resolve_owner_id(http_request)
    effective_session_id = scoped_session_id(owner_id, request.session_id)
    logger.info("[ChatRetry] owner_id=%s session_id=%s", owner_id, request.session_id)
    allowed_source_paths: set[str] | None = None
    if request.file_names:
        allowed_source_paths = {str(resolve_library_pdf(container.settings, file_name).resolve()) for file_name in request.file_names}

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
        uploaded_source_paths = {str((container.settings.rag_source_dir / file_name).resolve()) for file_name in uploaded_files}
        yield "data: " + json.dumps({"type": "upload", "uploaded_files": uploaded_files, "indexed_chunks": total_chunks}, ensure_ascii=False) + "\n\n"
        async for event in container.pipeline.stream_chat(
            session_id=effective_session_id,
            user_message=message,
            allowed_source_paths=uploaded_source_paths,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from app.api.routes_shared import delete_markdown_artifacts, fitz, list_source_pdfs, resolve_library_pdf, save_library_uploads
from app.api.schemas import BuildIndexResponse, DeleteLibraryResponse, LibraryProgressResponse, LibraryStatusResponse, TaskStatusResponse
from app.dependencies import AppContainer, get_container
from app.rag.utils import extracted_html_path, extracted_markdown_path

router = APIRouter()
logger = logging.getLogger("rag.startup")


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


@router.get("/api/library/status")
async def get_library_status(request: Request, container: AppContainer = Depends(get_container)) -> LibraryProgressResponse:
    startup_state = getattr(request.app.state, "startup_indexing", None) or {}
    reindex_state = getattr(request.app.state, "reindexing", None) or {}
    total_files = int(
        reindex_state.get("total_files")
        or startup_state.get("total_files")
        or len(list_source_pdfs(container.settings))
    )
    return LibraryProgressResponse(
        source_dir=str(container.settings.rag_source_dir),
        total_files=total_files,
        startup_indexing=startup_state,
        reindexing=reindex_state,
    )


@router.get("/api/library/preview")
async def preview_library_file(file_name: str = Query(..., min_length=1), container: AppContainer = Depends(get_container)) -> FileResponse:
    target_path = resolve_library_pdf(container.settings, file_name)
    return FileResponse(target_path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


@router.get("/api/library/preview-html")
async def preview_library_html(
    file_name: str = Query(..., min_length=1),
    container: AppContainer = Depends(get_container),
) -> Response:
    target_path = resolve_library_pdf(container.settings, file_name)
    html_path = extracted_html_path(container.settings.rag_extract_dir, target_path)
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html; charset=utf-8")

    markdown_path = extracted_markdown_path(container.settings.rag_extract_dir, target_path)
    if markdown_path.exists():
        markdown_text = markdown_path.read_text(encoding="utf-8")
        fallback_html = (
            "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{target_path.name}</title>"
            "<style>body{font-family:'Segoe UI',sans-serif;padding:24px;background:#f6f8fc;color:#182538}"
            "main{max-width:1100px;margin:0 auto;background:#fff;border:1px solid #dde4ef;border-radius:16px;padding:20px}"
            "pre{white-space:pre-wrap;word-break:break-word;font:13px/1.55 'Consolas','Courier New',monospace;margin:0}</style>"
            "</head><body><main><pre>"
            + markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            + "</pre></main></body></html>"
        )
        return HTMLResponse(fallback_html)

    raise HTTPException(status_code=404, detail="HTML preview artifact not found.")


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


@router.get("/api/library/highlight")
async def get_highlighted_page(
    file_name: str = Query(..., min_length=1),
    page: int = Query(..., ge=1),
    line_start: int = Query(..., ge=1),
    line_end: int = Query(..., ge=1),
    container: AppContainer = Depends(get_container),
) -> Response:
    if fitz is None:
        raise HTTPException(status_code=500, detail="PyMuPDF is required.")
    target_path = resolve_library_pdf(container.settings, file_name)
    doc = fitz.open(str(target_path))
    try:
        page_obj = doc[page - 1]
        blocks = page_obj.get_text("dict")["blocks"]
        line_num = 0
        rects_to_highlight = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_num += 1
                if line_start <= line_num <= line_end:
                    rects_to_highlight.append(fitz.Rect(line["bbox"]))
        for rect in rects_to_highlight:
            highlight = page_obj.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 1, 0))
            highlight.update()
        pix = page_obj.get_pixmap(dpi=150)
        return Response(content=pix.tobytes("png"), media_type="image/png")
    finally:
        doc.close()


@router.get("/api/library/download")
async def download_library_file(file_name: str = Query(..., min_length=1), container: AppContainer = Depends(get_container)) -> FileResponse:
    target_path = resolve_library_pdf(container.settings, file_name)
    return FileResponse(target_path, media_type="application/pdf", filename=target_path.name)


@router.delete("/api/library")
async def delete_library_file(file_name: str = Query(..., min_length=1), container: AppContainer = Depends(get_container)) -> DeleteLibraryResponse:
    target_path, deleted_markdown = delete_markdown_artifacts(container, file_name)
    result = container.indexing_service.delete_library_document(target_path)
    return DeleteLibraryResponse(deleted_file=target_path.name, deleted_markdown=deleted_markdown, **result)


@router.post("/api/library/upload")
async def upload_to_library(
    files: list[UploadFile] = File(...),
    target_group: str = Query(..., min_length=1),
    target_version: str | None = Query(None),
    container: AppContainer = Depends(get_container),
) -> StreamingResponse:
    uploaded_files = await save_library_uploads(
        container.settings,
        files,
        target_group=target_group,
        target_version=target_version,
    )
    total_files = len(uploaded_files)

    async def event_stream():
        loop = asyncio.get_running_loop()
        total_chunks = 0
        for file_idx, relative_path in enumerate(uploaded_files):
            source_path = container.settings.rag_source_dir / relative_path
            file_name = Path(relative_path).name
            queue: asyncio.Queue = asyncio.Queue()

            def make_progress_callback(q, ev_loop):
                def callback(stage, current, total, meta=None):
                    pct = int(current / total * 70) if stage == "extract" else 70 + int(current / total * 25)
                    ev_loop.call_soon_threadsafe(q.put_nowait, {"type": "progress", "file": file_name, "pct": pct})

                return callback

            index_task = asyncio.ensure_future(
                run_in_threadpool(
                    container.indexing_service.index_single_file,
                    source_path,
                    make_progress_callback(queue, loop),
                )
            )
            while not index_task.done():
                try:
                    event = queue.get_nowait()
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
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
            yield "data: " + json.dumps(
                {
                    "type": "file_indexed",
                    "file": file_name,
                    "relative_path": relative_path,
                    "file_idx": file_idx,
                    "total_files": total_files,
                    **result,
                    "library": library,
                },
                ensure_ascii=False,
            ) + "\n\n"
        yield "data: " + json.dumps({"type": "done", "total_chunks": total_chunks}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/reindex")
async def reindex_all(request: Request, container: AppContainer = Depends(get_container)) -> BuildIndexResponse:
    startup_state = getattr(request.app.state, "startup_indexing", None) or {}
    reindex_state = getattr(request.app.state, "reindexing", None) or {}
    if startup_state.get("status") == "indexing":
        raise HTTPException(status_code=409, detail="Startup indexing is still in progress.")
    if reindex_state.get("status") == "indexing":
        raise HTTPException(status_code=409, detail="A reindex job is already running.")

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
        reindex_state.update(current_stage="warming_cache", progress_pct=95)
        warm_start = time.perf_counter()
        warmed_items = await run_in_threadpool(container.pipeline.index_repository.warm_cache)
        logger.info("[Timing][Reindex] warm_cache=%.3fs items=%d", time.perf_counter() - warm_start, warmed_items)
    except Exception:
        reindex_state.update(status="idle", current_stage="error")
        raise

    reindex_state.update(
        status="done",
        completed_files=len(source_files),
        current_file="",
        current_stage="done",
        current_chunk=result.get("indexed_chunks", 0),
        total_chunks=warmed_items,
        progress_pct=100,
    )
    return BuildIndexResponse(**result)


@router.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str, container: AppContainer = Depends(get_container)) -> TaskStatusResponse:
    task = container.task_repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(**task)


@router.get("/api/library/{file_name}/chunks")
async def list_chunks(
    file_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source_path: str | None = Query(None),
    container: AppContainer = Depends(get_container),
):
    lookup_source_path = source_path or str(resolve_library_pdf(container.settings, file_name))
    all_items = container.pipeline.index_repository.list_all_chunks(lookup_source_path)
    total = len(all_items)
    offset = max(page - 1, 0) * page_size
    page_items = all_items[offset : offset + page_size]
    chunks = []
    for item in page_items:
        chunk = item["chunk"]
        metadata = chunk.get("metadata", {})
        chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "page_number": chunk.get("page_number") or metadata.get("page_start"),
                "token_count": len(chunk.get("tokens", [])),
                "html_anchor": metadata.get("html_anchor", ""),
                "block_anchor": metadata.get("primary_block_anchor", ""),
                "block_types": metadata.get("block_types", ""),
                "section_title": metadata.get("section_title", ""),
            }
        )
    return {
        "file_name": file_name,
        "source_path": lookup_source_path,
        "chunks": chunks,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/api/library/{file_name}/chunks/{chunk_id}")
async def get_chunk_detail(
    file_name: str,
    chunk_id: str,
    source_path: str | None = Query(None),
    container: AppContainer = Depends(get_container),
):
    lookup_source_path = source_path or str(resolve_library_pdf(container.settings, file_name))
    item = container.pipeline.index_repository.get_chunk(lookup_source_path, chunk_id)
    if item:
        chunk = item["chunk"]
        return {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "page_number": chunk.get("page_number") or chunk.get("metadata", {}).get("page_start"),
            "metadata": chunk.get("metadata", {}),
            "token_count": len(chunk.get("tokens", [])),
            "source_path": chunk.get("source_path", ""),
        }
    raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")

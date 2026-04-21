from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from apps.api.schemas.library import (
    LibraryCatalogResponse,
    LibraryDocumentChunksResponse,
    LibraryDocumentContentResponse,
    LibrarySummaryResponse,
)
from apps.api.rag.library import LibraryDocumentService, LibrarySummaryService
from apps.api.runtime import batch_job_service

router = APIRouter(prefix="/library", tags=["library"])
_service = LibrarySummaryService(batch_job_service=batch_job_service)
_document_service = LibraryDocumentService()


@router.get("/summary", response_model=LibrarySummaryResponse)
async def get_library_summary() -> LibrarySummaryResponse:
    return _service.get_summary()


@router.get("/catalog", response_model=LibraryCatalogResponse)
async def get_library_catalog() -> LibraryCatalogResponse:
    return _document_service.get_catalog()


@router.get("/chunks", response_model=LibraryDocumentChunksResponse)
async def get_library_document_chunks(document_key: str = Query(..., min_length=1)) -> LibraryDocumentChunksResponse:
    try:
        return _document_service.get_chunks(document_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/document-content", response_model=LibraryDocumentContentResponse)
async def get_library_document_content(document_key: str = Query(..., min_length=1)) -> LibraryDocumentContentResponse:
    try:
        return _document_service.get_markdown_content(document_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/document-file")
async def get_library_document_file(document_key: str = Query(..., min_length=1)) -> FileResponse:
    try:
        path = _document_service.get_file_path(document_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name)


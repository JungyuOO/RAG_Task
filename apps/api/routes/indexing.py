from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from apps.api.schemas.indexing import (
    BatchIndexRequest,
    BatchIndexResponse,
    BatchJobListResponse,
    BatchJobStatusResponse,
    IndexResetResponse,
    NewIndexRequest,
    NewIndexResponse,
)
from apps.api.rag.indexing import BatchIndexJobService, IndexAdminService, NewIndexingService
from apps.api.runtime import batch_indexing_service, batch_job_service

router = APIRouter(prefix="/index", tags=["index"])
_index_service = NewIndexingService()
_admin_service = IndexAdminService()
_batch_service = batch_indexing_service
_job_service: BatchIndexJobService = batch_job_service


@router.post("/source", response_model=NewIndexResponse)
async def index_source(request: NewIndexRequest) -> NewIndexResponse:
    try:
        return _index_service.index_source(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/reset", response_model=IndexResetResponse)
async def reset_index(preserve_embedding_cache: bool = Query(True)) -> IndexResetResponse:
    try:
        return _admin_service.reset_all(preserve_embedding_cache=preserve_embedding_cache)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/batch/reindex", response_model=BatchIndexResponse)
async def batch_reindex(request: BatchIndexRequest) -> BatchIndexResponse:
    try:
        return _batch_service.run(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/batch/jobs", response_model=BatchJobStatusResponse)
async def submit_batch_reindex_job(request: BatchIndexRequest, background_tasks: BackgroundTasks) -> BatchJobStatusResponse:
    try:
        job = _job_service.submit(request)
        background_tasks.add_task(_job_service.run_job, job.job_id)
        return job
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/batch/jobs/{job_id}", response_model=BatchJobStatusResponse)
async def get_batch_reindex_job(job_id: str) -> BatchJobStatusResponse:
    job = _job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return job


@router.get("/batch/jobs", response_model=BatchJobListResponse)
async def list_batch_reindex_jobs(limit: int = Query(20, ge=1, le=100)) -> BatchJobListResponse:
    return _job_service.list_recent(limit=limit)


@router.post("/batch/jobs/{job_id}/retry-failed", response_model=BatchJobStatusResponse)
async def retry_failed_batch_items(job_id: str, background_tasks: BackgroundTasks) -> BatchJobStatusResponse:
    try:
        job = _job_service.retry_failed(job_id)
        background_tasks.add_task(_job_service.run_job, job.job_id)
        return job
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/batch/jobs/{job_id}/cancel", response_model=BatchJobStatusResponse)
async def cancel_batch_job(job_id: str) -> BatchJobStatusResponse:
    try:
        return _job_service.cancel(job_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


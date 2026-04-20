from __future__ import annotations

from apps.api.persistence.batch_job_repository import InMemoryBatchJobRepository
from apps.api.api.schemas.indexing import BatchIndexRequest, BatchJobListResponse, BatchJobStatusResponse
from apps.api.rag.indexing.batch_index_service import BatchIndexingService


class BatchIndexJobService:
    def __init__(
        self,
        *,
        batch_service: BatchIndexingService | None = None,
        job_repository: InMemoryBatchJobRepository | None = None,
    ) -> None:
        self.batch_service = batch_service or BatchIndexingService()
        self.job_repository = job_repository or InMemoryBatchJobRepository()

    def submit(self, request: BatchIndexRequest) -> BatchJobStatusResponse:
        return self.job_repository.create(request)

    def run_job(self, job_id: str) -> BatchJobStatusResponse:
        existing = self.job_repository.get(job_id)
        if existing is not None and existing.status == "cancelled":
            return existing
        job = self.job_repository.mark_running(job_id)
        if job.status == "cancelled":
            return job
        try:
            result = self.batch_service.run(
                job.request,
                progress_callback=lambda *, progress_pct, current_file: self.job_repository.mark_progress(
                    job_id,
                    progress_pct=progress_pct,
                    current_file=current_file,
                ),
                log_callback=lambda message: self.job_repository.append_log(job_id, message),
                status_callback=lambda *, step, message, current_file=None: self.job_repository.mark_step(
                    job_id,
                    step=step,
                    message=message,
                    current_file=current_file,
                ),
                should_cancel=lambda: self.job_repository.is_cancelled(job_id),
            )
        except Exception as exc:
            return self.job_repository.mark_failed(job_id, str(exc))
        if self.job_repository.is_cancelled(job_id):
            return self.job_repository.mark_cancelled(job_id, result=result)
        return self.job_repository.mark_completed(job_id, result)

    def get(self, job_id: str) -> BatchJobStatusResponse | None:
        return self.job_repository.get(job_id)

    def list_recent(self, limit: int = 20) -> BatchJobListResponse:
        return BatchJobListResponse(jobs=self.job_repository.list_recent(limit=limit))

    def retry_failed(self, job_id: str) -> BatchJobStatusResponse:
        current = self.job_repository.get(job_id)
        if current is None:
            raise LookupError(f"Unknown batch job: {job_id}")
        if current.result is None:
            raise ValueError("Cannot retry failed items before a batch job has produced a result.")
        failed_paths = [item.source_path for item in current.result.items if item.error]
        if not failed_paths:
            raise ValueError("No failed items available to retry.")

        retry_request = BatchIndexRequest(
            explicit_source_paths=failed_paths,
            source_type=current.request.source_type,
            document_group=current.request.document_group,
            locale=current.request.locale,
            max_files=0,
            include_subdirectories=False,
        )
        return self.submit(retry_request)

    def cancel(self, job_id: str) -> BatchJobStatusResponse:
        return self.job_repository.mark_cancelled(job_id)



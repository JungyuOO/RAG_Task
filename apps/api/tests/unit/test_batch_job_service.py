from __future__ import annotations

import unittest

from apps.api.storage.batch_job_repository import InMemoryBatchJobRepository
from apps.api.schemas.batch_indexing import BatchIndexRequest, BatchIndexResponse
from apps.api.rag.indexing.batch_job_service import BatchIndexJobService

FIXTURE_ROOT = "fixtures/generated-manual"
FIXTURE_FILE = f"{FIXTURE_ROOT}/manual_fixture.md"


class _FakeBatchService:
    def run(
        self,
        request: BatchIndexRequest,
        *,
        progress_callback=None,
        should_cancel=None,
        log_callback=None,
        status_callback=None,
    ) -> BatchIndexResponse:
        if log_callback:
            log_callback("fake batch started")
        if status_callback:
            status_callback(step="RUNNING", message="fake batch running", current_file=FIXTURE_FILE)
        if progress_callback:
            progress_callback(progress_pct=30, current_file=FIXTURE_FILE)
        if should_cancel and should_cancel():
            return BatchIndexResponse(
                discovered_files=1,
                processed_files=0,
                indexed_files=0,
                failed_files=0,
                progress_pct=30,
                current_file="",
                items=[],
            )
        failed = 1 if request.root_path == "force-fail" else 0
        return BatchIndexResponse(
            discovered_files=1,
            processed_files=1,
            indexed_files=0 if failed else 1,
            failed_files=failed,
            progress_pct=100,
            current_file="",
            items=[
                {
                    "source_path": FIXTURE_FILE,
                    "source_type": "generated-manual",
                    "indexed": failed == 0,
                    "chunks": 3 if failed == 0 else 0,
                    "error": "failed" if failed else "",
                }
            ],
        )


class BatchJobServiceTests(unittest.TestCase):
    def test_job_service_lifecycle(self) -> None:
        service = BatchIndexJobService(
            batch_service=_FakeBatchService(),  # type: ignore[arg-type]
            job_repository=InMemoryBatchJobRepository(),
        )
        job = service.submit(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=1))
        self.assertEqual(job.status, "pending")

        completed = service.run_job(job.job_id)
        self.assertEqual(completed.status, "completed")
        self.assertIsNotNone(completed.result)
        self.assertEqual(completed.progress_pct, 100)

    def test_retry_failed_creates_new_job(self) -> None:
        service = BatchIndexJobService(
            batch_service=_FakeBatchService(),  # type: ignore[arg-type]
            job_repository=InMemoryBatchJobRepository(),
        )
        original = service.submit(BatchIndexRequest(root_path="force-fail", max_files=1))
        completed = service.run_job(original.job_id)
        self.assertEqual(completed.result.failed_files, 1)

        retry_job = service.retry_failed(original.job_id)
        self.assertEqual(retry_job.status, "pending")
        self.assertEqual(retry_job.request.explicit_source_paths, [FIXTURE_FILE])

    def test_list_recent_returns_jobs_in_reverse_chronological_order(self) -> None:
        service = BatchIndexJobService(
            batch_service=_FakeBatchService(),  # type: ignore[arg-type]
            job_repository=InMemoryBatchJobRepository(),
        )
        first = service.submit(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=1))
        second = service.submit(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=2))
        jobs = service.list_recent(limit=10).jobs
        self.assertEqual(jobs[0].job_id, second.job_id)
        self.assertEqual(jobs[1].job_id, first.job_id)

    def test_cancel_marks_job_cancelled(self) -> None:
        service = BatchIndexJobService(
            batch_service=_FakeBatchService(),  # type: ignore[arg-type]
            job_repository=InMemoryBatchJobRepository(),
        )
        job = service.submit(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=1))
        cancelled = service.cancel(job.job_id)
        self.assertEqual(cancelled.status, "cancelled")

    def test_run_job_respects_pre_cancelled_state(self) -> None:
        service = BatchIndexJobService(
            batch_service=_FakeBatchService(),  # type: ignore[arg-type]
            job_repository=InMemoryBatchJobRepository(),
        )
        job = service.submit(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=1))
        service.cancel(job.job_id)
        final = service.run_job(job.job_id)
        self.assertEqual(final.status, "cancelled")

    def test_run_job_preserves_partial_result_when_cancelled_during_run(self) -> None:
        class _CancelAfterProgressService(_FakeBatchService):
            def __init__(self, job_service: BatchIndexJobService, job_id_ref: list[str]) -> None:
                self.job_service = job_service
                self.job_id_ref = job_id_ref

            def run(
                self,
                request: BatchIndexRequest,
                *,
                progress_callback=None,
                should_cancel=None,
                log_callback=None,
                status_callback=None,
            ) -> BatchIndexResponse:
                if log_callback:
                    log_callback("cancel-after-progress")
                if status_callback:
                    status_callback(step="RUNNING", message="cancel after progress", current_file=FIXTURE_FILE)
                if progress_callback:
                    progress_callback(progress_pct=30, current_file=FIXTURE_FILE)
                self.job_service.cancel(self.job_id_ref[0])
                if should_cancel and should_cancel():
                    return BatchIndexResponse(
                        discovered_files=2,
                        processed_files=1,
                        indexed_files=1,
                        failed_files=0,
                        progress_pct=50,
                        current_file="",
                        items=[],
                    )
                return super().run(
                    request,
                    progress_callback=progress_callback,
                    should_cancel=should_cancel,
                    log_callback=log_callback,
                    status_callback=status_callback,
                )

        repository = InMemoryBatchJobRepository()
        job_service = BatchIndexJobService(
            batch_service=_FakeBatchService(),  # placeholder replaced below
            job_repository=repository,
        )
        holder: list[str] = []
        job_service.batch_service = _CancelAfterProgressService(job_service, holder)  # type: ignore[assignment]
        job = job_service.submit(BatchIndexRequest(root_path=FIXTURE_ROOT, max_files=2))
        holder.append(job.job_id)
        final = job_service.run_job(job.job_id)
        self.assertEqual(final.status, "cancelled")
        self.assertIsNotNone(final.result)
        assert final.result is not None
        self.assertEqual(final.result.processed_files, 1)


if __name__ == "__main__":
    unittest.main()





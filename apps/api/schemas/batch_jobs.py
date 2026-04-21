from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from apps.api.schemas.batch_indexing import BatchIndexRequest, BatchIndexResponse


class BatchJobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    task_type: str = "batch_reindex"
    status: str
    request: BatchIndexRequest
    result: BatchIndexResponse | None = None
    error: str = ""
    progress_pct: int = 0
    current_file: str = ""
    step: str = "pending"
    message: str = ""
    recent_logs: list[str] = []
    created_at: datetime
    updated_at: datetime




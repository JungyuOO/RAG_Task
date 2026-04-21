from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from apps.api.schemas.batch_jobs import BatchJobStatusResponse


class BatchJobListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[BatchJobStatusResponse] = Field(default_factory=list)




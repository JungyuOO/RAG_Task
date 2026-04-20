from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IndexResetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cleared_tables: list[str] = Field(default_factory=list)
    cleared_files: dict[str, int] = Field(default_factory=dict)


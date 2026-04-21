from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MetricSnapshotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    workspace_id: str
    connection_id: str
    namespace: str = ""
    metric_key: str
    metric_value: float
    unit: str = ""
    collected_at: datetime


class MetricSnapshotListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[MetricSnapshotRecord] = Field(default_factory=list)


class RecommendationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str
    workspace_id: str
    connection_id: str
    namespace: str = ""
    resource_kind: str = ""
    resource_name: str = ""
    recommendation_type: str
    risk_level: str = "info"
    summary: str
    rationale: str = ""
    created_at: datetime


class RecommendationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RecommendationRecord] = Field(default_factory=list)


class RecommendationRefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str = ""

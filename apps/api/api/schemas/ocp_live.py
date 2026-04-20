from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OcpLiveResourceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    namespace: str = ""
    kind: str = ""
    created_at: str = ""
    phase: str = ""
    node_name: str = ""
    ready_replicas: int = 0
    replicas: int = 0
    type: str = ""
    cluster_ip: str = ""
    host: str = ""
    to: str = ""


class OcpLiveNamespaceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    cluster_url: str
    count: int
    items: list[str] = Field(default_factory=list)


class OcpLiveResourceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    cluster_url: str
    resource: str
    namespace: str
    count: int
    items: list[OcpLiveResourceSummary] = Field(default_factory=list)


class OcpLiveResourceDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    cluster_url: str
    resource: str
    namespace: str
    name: str
    kind: str
    manifest_yaml: str
    manifest_json: dict[str, Any] = Field(default_factory=dict)


class OcpOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    cluster_url: str
    default_namespace: str = ""
    namespace_count: int = 0
    namespace_sample: list[str] = Field(default_factory=list)
    resource_counts: dict[str, int] = Field(default_factory=dict)
    message: str = ""


class OcpMetricPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: int
    value: float


class OcpMetricSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str
    label: str
    unit: str = ""
    current_value: float = 0.0
    capacity_value: float = 0.0
    available_value: float = 0.0
    points: list[OcpMetricPoint] = Field(default_factory=list)


class OcpDashboardMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    cluster_url: str
    window: str
    step: str
    series: list[OcpMetricSeries] = Field(default_factory=list)


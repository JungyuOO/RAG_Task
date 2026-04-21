from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from apps.api.schemas.ocp import (
    OcpDashboardMetricsResponse,
    OcpLiveNamespaceListResponse,
    OcpLiveResourceDetailResponse,
    OcpLiveResourceListResponse,
    OcpLiveResourceSummary,
    OcpMetricPoint,
    OcpMetricSeries,
    OcpOverviewResponse,
)
from apps.api.ocp.auth import OcpConnectionBroker


class ConnectedOcpService:
    RESOURCE_CONFIG = {
        "pods": {"path": "/api/v1/namespaces/{namespace}/pods", "kind": "Pod"},
        "deployments": {"path": "/apis/apps/v1/namespaces/{namespace}/deployments", "kind": "Deployment"},
        "services": {"path": "/api/v1/namespaces/{namespace}/services", "kind": "Service"},
        "routes": {"path": "/apis/route.openshift.io/v1/namespaces/{namespace}/routes", "kind": "Route"},
        "events": {"path": "/api/v1/namespaces/{namespace}/events", "kind": "Event"},
    }

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 15.0) -> None:
        self.transport = transport
        self.timeout = timeout
        self.prometheus_proxy_path = "/api/v1/namespaces/openshift-monitoring/services/https:thanos-querier:9092/proxy/api/v1/query_range"

    async def list_namespaces(
        self,
        connection_id: str,
        broker: OcpConnectionBroker,
    ) -> OcpLiveNamespaceListResponse:
        profile = self._require_profile(connection_id, broker)
        runtime = broker.build_runtime_config(profile)
        self._require_token_mode(runtime)
        headers = self._headers(runtime)

        async with httpx.AsyncClient(
            verify=profile.verify_ssl,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = await client.get(f"{profile.cluster_url}/api/v1/namespaces", headers=headers)

        if response.status_code == 403:
            items = [profile.default_namespace] if profile.default_namespace else []
            return OcpLiveNamespaceListResponse(
                connection_id=profile.connection_id,
                cluster_url=profile.cluster_url,
                count=len(items),
                items=items,
            )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        items = [
            str((item.get("metadata", {}) or {}).get("name") or "").strip()
            for item in payload.get("items", []) or []
        ]
        items = [name for name in items if name]
        return OcpLiveNamespaceListResponse(
            connection_id=profile.connection_id,
            cluster_url=profile.cluster_url,
            count=len(items),
            items=items,
        )

    async def list_resources(
        self,
        connection_id: str,
        *,
        resource: str,
        namespace: str = "",
        broker: OcpConnectionBroker,
    ) -> OcpLiveResourceListResponse:
        profile = self._require_profile(connection_id, broker)
        runtime = broker.build_runtime_config(profile)
        self._require_token_mode(runtime)
        config = self.RESOURCE_CONFIG.get(resource)
        if config is None:
            raise ValueError(f"unsupported resource: {resource}")
        resolved_namespace = namespace or profile.default_namespace
        if not resolved_namespace:
            raise ValueError("namespace is required")

        path = config["path"].format(namespace=resolved_namespace)
        headers = self._headers(runtime)
        async with httpx.AsyncClient(
            verify=profile.verify_ssl,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = await client.get(f"{profile.cluster_url}{path}", headers=headers)
        response.raise_for_status()
        payload = response.json() if response.content else {}
        items = [self._summarize_resource(resource, item) for item in payload.get("items", []) or []]
        return OcpLiveResourceListResponse(
            connection_id=profile.connection_id,
            cluster_url=profile.cluster_url,
            resource=resource,
            namespace=resolved_namespace,
            count=len(items),
            items=items,
        )

    async def get_resource_detail(
        self,
        connection_id: str,
        *,
        resource: str,
        namespace: str = "",
        name: str,
        broker: OcpConnectionBroker,
    ) -> OcpLiveResourceDetailResponse:
        profile = self._require_profile(connection_id, broker)
        runtime = broker.build_runtime_config(profile)
        self._require_token_mode(runtime)
        config = self.RESOURCE_CONFIG.get(resource)
        if config is None:
            raise ValueError(f"unsupported resource: {resource}")
        resolved_namespace = namespace or profile.default_namespace
        if not resolved_namespace:
            raise ValueError("namespace is required")
        if not name.strip():
            raise ValueError("resource name is required")

        path = f"{config['path'].format(namespace=resolved_namespace)}/{name}"
        headers = self._headers(runtime)
        async with httpx.AsyncClient(
            verify=profile.verify_ssl,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            response = await client.get(f"{profile.cluster_url}{path}", headers=headers)
        response.raise_for_status()
        payload = response.json() if response.content else {}
        summary = self._summarize_resource(resource, payload)
        return OcpLiveResourceDetailResponse(
            connection_id=profile.connection_id,
            cluster_url=profile.cluster_url,
            resource=resource,
            namespace=resolved_namespace,
            name=summary.name,
            kind=summary.kind,
            manifest_yaml=self._dump_yaml(payload),
            manifest_json=payload,
        )

    async def resolve_resource_detail_by_name(
        self,
        connection_id: str,
        *,
        namespace: str = "",
        name: str,
        broker: OcpConnectionBroker,
        preferred_resource: str = "",
    ) -> OcpLiveResourceDetailResponse:
        search_resources = [preferred_resource] if preferred_resource in self.RESOURCE_CONFIG else []
        for resource in self.RESOURCE_CONFIG:
            if resource not in search_resources:
                search_resources.append(resource)

        for resource in search_resources:
            try:
                result = await self.list_resources(
                    connection_id,
                    resource=resource,
                    namespace=namespace,
                    broker=broker,
                )
            except Exception:
                continue
            matched = next((item for item in result.items if item.name.casefold() == name.casefold()), None)
            if matched is None:
                continue
            return await self.get_resource_detail(
                connection_id,
                resource=resource,
                namespace=result.namespace,
                name=matched.name,
                broker=broker,
            )
        raise LookupError(f"Resource named '{name}' was not found in the current namespace scope.")

    async def get_overview(self, connection_id: str, broker: OcpConnectionBroker) -> OcpOverviewResponse:
        profile = self._require_profile(connection_id, broker)
        namespaces_response = await self.list_namespaces(connection_id, broker)
        namespace = profile.default_namespace or (namespaces_response.items[0] if namespaces_response.items else "")

        resource_counts: dict[str, int] = {}
        message = "No namespace available for resource overview."
        if namespace:
            for resource in ("pods", "deployments", "services", "routes", "events"):
                try:
                    result = await self.list_resources(
                        connection_id,
                        resource=resource,
                        namespace=namespace,
                        broker=broker,
                    )
                    resource_counts[resource] = result.count
                except httpx.HTTPStatusError:
                    resource_counts[resource] = -1
            message = "Overview loaded from live cluster."

        return OcpOverviewResponse(
            connection_id=profile.connection_id,
            cluster_url=profile.cluster_url,
            default_namespace=namespace,
            namespace_count=namespaces_response.count,
            namespace_sample=namespaces_response.items[:8],
            resource_counts=resource_counts,
            message=message,
        )

    async def get_dashboard_metrics(
        self,
        connection_id: str,
        broker: OcpConnectionBroker,
        *,
        window: str = "1h",
        step: str = "5m",
    ) -> OcpDashboardMetricsResponse:
        profile = self._require_profile(connection_id, broker)
        runtime = broker.build_runtime_config(profile)
        self._require_token_mode(runtime)
        headers = self._headers(runtime)
        start_at, end_at = self._resolve_time_window(window)
        resolved_namespace = profile.default_namespace.strip()
        if not resolved_namespace:
            namespaces_response = await self.list_namespaces(connection_id, broker)
            resolved_namespace = namespaces_response.items[0] if namespaces_response.items else ""
        if not resolved_namespace:
            raise ValueError("default namespace is required for dashboard metrics")

        series_specs = [
            {
                "metric_id": "cpu_usage",
                "label": "CPU",
                "unit": "cores",
                "usage_query": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{resolved_namespace}",container!="",pod!=""}}[5m]))',
                "capacity_query": "",
            },
            {
                "metric_id": "memory_usage",
                "label": "Memory",
                "unit": "bytes",
                "usage_query": f'sum(container_memory_working_set_bytes{{namespace="{resolved_namespace}",container!="",pod!=""}})',
                "capacity_query": "",
            },
            {
                "metric_id": "filesystem_usage",
                "label": "Filesystem",
                "unit": "bytes",
                "usage_query": f'sum(container_fs_usage_bytes{{namespace="{resolved_namespace}",container!="",pod!=""}})',
                "capacity_query": "",
            },
            {
                "metric_id": "network_in",
                "label": "Network In",
                "unit": "bytes_per_second",
                "usage_query": f'sum(rate(container_network_receive_bytes_total{{namespace="{resolved_namespace}",pod!=""}}[5m]))',
                "capacity_query": "",
            },
            {
                "metric_id": "network_out",
                "label": "Network Out",
                "unit": "bytes_per_second",
                "usage_query": f'sum(rate(container_network_transmit_bytes_total{{namespace="{resolved_namespace}",pod!=""}}[5m]))',
                "capacity_query": "",
            },
            {
                "metric_id": "pod_count",
                "label": "Pods",
                "unit": "count",
                "usage_query": f'count(kube_pod_info{{namespace="{resolved_namespace}"}})',
                "capacity_query": "",
            },
        ]

        async with httpx.AsyncClient(
            verify=profile.verify_ssl,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            built_series: list[OcpMetricSeries] = []
            for spec in series_specs:
                usage_result = await self._query_range(
                    client=client,
                    cluster_url=profile.cluster_url,
                    headers=headers,
                    query=spec["usage_query"],
                    start_at=start_at,
                    end_at=end_at,
                    step=step,
                )
                capacity_result = None
                if spec["capacity_query"]:
                    capacity_result = await self._query_range(
                        client=client,
                        cluster_url=profile.cluster_url,
                        headers=headers,
                        query=spec["capacity_query"],
                        start_at=start_at,
                        end_at=end_at,
                        step=step,
                    )
                points = self._prometheus_to_points(usage_result)
                current_value = points[-1].value if points else 0.0
                capacity_points = self._prometheus_to_points(capacity_result) if capacity_result else []
                capacity_value = capacity_points[-1].value if capacity_points else 0.0
                built_series.append(
                    OcpMetricSeries(
                        metric_id=spec["metric_id"],
                        label=spec["label"],
                        unit=spec["unit"],
                        current_value=current_value,
                        capacity_value=capacity_value,
                        available_value=max(capacity_value - current_value, 0.0) if capacity_value else 0.0,
                        points=points,
                    )
                )

        return OcpDashboardMetricsResponse(
            connection_id=profile.connection_id,
            cluster_url=profile.cluster_url,
            window=window,
            step=step,
            series=built_series,
        )

    async def _query_range(
        self,
        *,
        client: httpx.AsyncClient,
        cluster_url: str,
        headers: dict[str, str],
        query: str,
        start_at: datetime,
        end_at: datetime,
        step: str,
    ) -> dict[str, Any] | None:
        params = {
            "query": query,
            "start": str(int(start_at.timestamp())),
            "end": str(int(end_at.timestamp())),
            "step": step,
        }
        response = await client.get(f"{cluster_url}{self.prometheus_proxy_path}", headers=headers, params=params)
        response.raise_for_status()
        return response.json() if response.content else {}

    @staticmethod
    def _prometheus_to_points(payload: dict[str, Any] | None) -> list[OcpMetricPoint]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data") or {}
        results = data.get("result") or []
        if not isinstance(results, list) or not results:
            return []
        values = results[0].get("values") or []
        points: list[OcpMetricPoint] = []
        for item in values:
            if not isinstance(item, list) or len(item) < 2:
                continue
            try:
                points.append(OcpMetricPoint(timestamp=int(float(item[0])), value=float(item[1])))
            except (TypeError, ValueError):
                continue
        return points

    @staticmethod
    def _resolve_time_window(window: str) -> tuple[datetime, datetime]:
        mapping = {"1h": timedelta(hours=1), "6h": timedelta(hours=6), "24h": timedelta(hours=24)}
        delta = mapping.get(str(window or "").strip(), timedelta(hours=1))
        end_at = datetime.now(UTC)
        return end_at - delta, end_at

    async def resolve_resource_relations(
        self,
        connection_id: str,
        *,
        namespace: str,
        broker: OcpConnectionBroker,
        resource: str = "",
        name: str = "",
    ) -> dict[str, Any]:
        resolved_namespace = namespace or self._require_profile(connection_id, broker).default_namespace
        pods = await self.list_resources(connection_id, resource="pods", namespace=resolved_namespace, broker=broker)
        services = await self.list_resources(connection_id, resource="services", namespace=resolved_namespace, broker=broker)
        pod_details: dict[str, dict[str, Any]] = {}
        for item in pods.items:
            try:
                detail = await self.get_resource_detail(
                    connection_id,
                    resource="pods",
                    namespace=resolved_namespace,
                    name=item.name,
                    broker=broker,
                )
                pod_details[item.name] = detail.manifest_json
            except Exception:
                continue

        service_details: dict[str, dict[str, Any]] = {}
        for item in services.items:
            try:
                detail = await self.get_resource_detail(
                    connection_id,
                    resource="services",
                    namespace=resolved_namespace,
                    name=item.name,
                    broker=broker,
                )
                service_details[item.name] = detail.manifest_json
            except Exception:
                continue

        relations: list[dict[str, Any]] = []
        for pod in pods.items:
            if resource == "pods" and name and pod.name.casefold() != name.casefold():
                continue
            labels = (((pod_details.get(pod.name) or {}).get("metadata") or {}).get("labels") or {})
            if not isinstance(labels, dict):
                labels = {}
            matched_services: list[str] = []
            for service in services.items:
                selector = ((((service_details.get(service.name) or {}).get("spec") or {}).get("selector")) or {})
                if not isinstance(selector, dict) or not selector:
                    continue
                if all(str(labels.get(key) or "") == str(value) for key, value in selector.items()):
                    matched_services.append(service.name)
            if matched_services:
                relations.append(
                    {
                        "pod": pod.name,
                        "services": matched_services,
                    }
                )
        return {
            "namespace": resolved_namespace,
            "relations": relations,
        }

    async def get_resource_health_summary(
        self,
        connection_id: str,
        *,
        namespace: str,
        name: str,
        broker: OcpConnectionBroker,
        preferred_resource: str = "",
    ) -> dict[str, Any]:
        detail = await self.resolve_resource_detail_by_name(
            connection_id,
            namespace=namespace,
            name=name,
            broker=broker,
            preferred_resource=preferred_resource,
        )
        events = await self.list_resources(connection_id, resource="events", namespace=detail.namespace, broker=broker)
        related_events = [
            item for item in events.items if item.to.casefold() == detail.name.casefold()
        ]
        spec = detail.manifest_json.get("spec") if isinstance(detail.manifest_json, dict) else {}
        status = detail.manifest_json.get("status") if isinstance(detail.manifest_json, dict) else {}
        desired_replicas = int((spec or {}).get("replicas") or 0) if isinstance(spec, dict) else 0
        ready_replicas = int((status or {}).get("readyReplicas") or 0) if isinstance(status, dict) else 0
        phase = str((status or {}).get("phase") or "") if isinstance(status, dict) else ""
        issues: list[str] = []
        if desired_replicas and ready_replicas < desired_replicas:
            issues.append(f"ready replicas {ready_replicas}/{desired_replicas}")
        if phase and phase.casefold() not in {"running", "active", "succeeded"}:
            issues.append(f"phase={phase}")
        if related_events:
            issues.append(f"related warning/error events={len(related_events)}")
        return {
            "resource": detail.resource,
            "name": detail.name,
            "namespace": detail.namespace,
            "kind": detail.kind,
            "issues": issues,
            "events": [event.model_dump() for event in related_events[:8]],
            "manifest_yaml": detail.manifest_yaml,
            "manifest_json": detail.manifest_json,
        }

    @staticmethod
    def _headers(runtime: dict[str, Any]) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {runtime.get('token', '')}",
            "Accept": "application/json",
        }

    @staticmethod
    def _require_token_mode(runtime: dict[str, Any]) -> None:
        if runtime.get("exchange_required"):
            raise ValueError("This connection requires token exchange and is not available for live resource access yet.")

    @staticmethod
    def _require_profile(connection_id: str, broker: OcpConnectionBroker):
        profile = broker.get_profile(connection_id)
        if profile is None:
            raise LookupError(f"Unknown connection_id={connection_id}")
        return profile

    @classmethod
    def _summarize_resource(cls, resource: str, item: dict) -> OcpLiveResourceSummary:
        metadata = item.get("metadata", {}) or {}
        status = item.get("status", {}) or {}
        spec = item.get("spec", {}) or {}
        summary = OcpLiveResourceSummary(
            name=str(metadata.get("name") or ""),
            namespace=str(metadata.get("namespace") or ""),
            kind=cls.RESOURCE_CONFIG[resource]["kind"],
            created_at=str(metadata.get("creationTimestamp") or ""),
        )
        if resource == "pods":
            summary.phase = str(status.get("phase") or "")
            summary.node_name = str(spec.get("nodeName") or "")
        elif resource == "deployments":
            summary.ready_replicas = int(status.get("readyReplicas") or 0)
            summary.replicas = int(spec.get("replicas") or 0)
        elif resource == "services":
            summary.type = str(spec.get("type") or "")
            summary.cluster_ip = str(spec.get("clusterIP") or "")
        elif resource == "routes":
            summary.host = str(spec.get("host") or "")
            summary.to = str((spec.get("to") or {}).get("name") or "")
        elif resource == "events":
            involved = item.get("involvedObject", {}) or {}
            summary.type = str(item.get("type") or "")
            summary.phase = str(item.get("reason") or "")
            summary.to = str(involved.get("name") or "")
            summary.host = str(involved.get("kind") or "")
        return summary

    @classmethod
    def _dump_yaml(cls, value: Any, indent: int = 0) -> str:
        return "\n".join(cls._dump_yaml_lines(value, indent=indent))

    @classmethod
    def _dump_yaml_lines(cls, value: Any, *, indent: int) -> list[str]:
        prefix = " " * indent
        if isinstance(value, dict):
            if not value:
                return [f"{prefix}{{}}"]
            lines: list[str] = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    if not item:
                        empty_value = "{}" if isinstance(item, dict) else "[]"
                        lines.append(f"{prefix}{key}: {empty_value}")
                    else:
                        lines.append(f"{prefix}{key}:")
                        lines.extend(cls._dump_yaml_lines(item, indent=indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {cls._yaml_scalar(item)}")
            return lines
        if isinstance(value, list):
            if not value:
                return [f"{prefix}[]"]
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    if not item:
                        empty_value = "{}" if isinstance(item, dict) else "[]"
                        lines.append(f"{prefix}- {empty_value}")
                    else:
                        lines.append(f"{prefix}-")
                        lines.extend(cls._dump_yaml_lines(item, indent=indent + 2))
                else:
                    lines.append(f"{prefix}- {cls._yaml_scalar(item)}")
            return lines
        return [f"{prefix}{cls._yaml_scalar(value)}"]

    @staticmethod
    def _yaml_scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        if not text:
            return '""'
        if "\n" in text:
            return json.dumps(text, ensure_ascii=False)
        if all(char.isalnum() or char in "-_./:@" for char in text):
            return text
        return json.dumps(text, ensure_ascii=False)



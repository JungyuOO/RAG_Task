from __future__ import annotations

import httpx


class OcpApiClient:
    RESOURCE_CONFIG = {
        "pods": {"api_path": "/api/v1/namespaces/{namespace}/pods", "kind": "Pod"},
        "deployments": {"api_path": "/apis/apps/v1/namespaces/{namespace}/deployments", "kind": "Deployment"},
        "services": {"api_path": "/api/v1/namespaces/{namespace}/services", "kind": "Service"},
        "routes": {"api_path": "/apis/route.openshift.io/v1/namespaces/{namespace}/routes", "kind": "Route"},
        "events": {"api_path": "/api/v1/namespaces/{namespace}/events", "kind": "Event"},
    }

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        verify_ssl: bool = True,
        default_namespace: str = "",
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.verify_ssl = verify_ssl
        self.default_namespace = default_namespace.strip()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.token)

    def resolve_namespace(self, namespace: str | None = None) -> str:
        resolved = (namespace or self.default_namespace).strip()
        if not resolved:
            raise ValueError("namespace is required")
        return resolved

    def build_headers(self) -> dict[str, str]:
        if not self.enabled:
            raise RuntimeError("OCP API is not configured")
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def build_resource_path(self, resource: str, namespace: str | None = None, name: str | None = None) -> str:
        config = self.RESOURCE_CONFIG.get(resource)
        if config is None:
            raise ValueError(f"unsupported resource: {resource}")
        resolved_namespace = self.resolve_namespace(namespace)
        path = config["api_path"].format(namespace=resolved_namespace)
        if name:
            path += f"/{name}"
        return path

    async def get_json(self, path: str) -> dict:
        async with httpx.AsyncClient(base_url=self.base_url, verify=self.verify_ssl, timeout=self.timeout) as client:
            response = await client.get(path, headers=self.build_headers())
            response.raise_for_status()
            return response.json()

    async def list_resources(self, resource: str, namespace: str | None = None) -> dict:
        path = self.build_resource_path(resource, namespace=namespace)
        payload = await self.get_json(path)
        items = payload.get("items", []) or []
        return {
            "resource": resource,
            "namespace": self.resolve_namespace(namespace),
            "count": len(items),
            "items": [self._summarize_resource(resource, item) for item in items],
        }

    async def list_namespaces(self) -> dict:
        try:
            payload = await self.get_json("/api/v1/namespaces")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                fallback_names = [self.default_namespace] if self.default_namespace else []
                return {
                    "count": len(fallback_names),
                    "items": fallback_names,
                }
            raise
        items = payload.get("items", []) or []
        names = [
            str((item.get("metadata", {}) or {}).get("name") or "").strip()
            for item in items
        ]
        names = [name for name in names if name]
        names.sort()
        return {
            "count": len(names),
            "items": names,
        }

    async def get_resource_yaml(self, resource: str, name: str, namespace: str | None = None) -> dict:
        path = self.build_resource_path(resource, namespace=namespace, name=name)
        payload = await self.get_json(path)
        return {
            "resource": resource,
            "namespace": self.resolve_namespace(namespace),
            "name": name,
            "object": payload,
        }

    def _summarize_resource(self, resource: str, item: dict) -> dict:
        metadata = item.get("metadata", {}) or {}
        status = item.get("status", {}) or {}
        spec = item.get("spec", {}) or {}
        summary = {
            "name": str(metadata.get("name") or ""),
            "namespace": str(metadata.get("namespace") or ""),
            "kind": self.RESOURCE_CONFIG[resource]["kind"],
            "created_at": str(metadata.get("creationTimestamp") or ""),
        }
        if resource == "pods":
            summary["phase"] = str(status.get("phase") or "")
            summary["node_name"] = str(spec.get("nodeName") or "")
        elif resource == "deployments":
            summary["ready_replicas"] = int(status.get("readyReplicas") or 0)
            summary["replicas"] = int(spec.get("replicas") or 0)
        elif resource == "services":
            summary["type"] = str(spec.get("type") or "")
            summary["cluster_ip"] = str(spec.get("clusterIP") or "")
        elif resource == "routes":
            summary["host"] = str(spec.get("host") or "")
            summary["to"] = str((spec.get("to") or {}).get("name") or "")
        elif resource == "events":
            involved = item.get("involvedObject", {}) or {}
            summary["type"] = str(item.get("type") or "")
            summary["phase"] = str(item.get("reason") or "")
            summary["to"] = str(involved.get("name") or "")
            summary["host"] = str(involved.get("kind") or "")
        return summary

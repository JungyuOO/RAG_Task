from __future__ import annotations

import httpx
import unittest

from app.ocp_client import OcpApiClient


class OcpApiClientTests(unittest.TestCase):
    def test_enabled_when_base_url_and_token_present(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="demo")
        self.assertTrue(client.enabled)

    def test_build_headers_sets_bearer_token(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="demo")
        headers = client.build_headers()
        self.assertEqual(headers["Authorization"], "Bearer token")
        self.assertEqual(headers["Accept"], "application/json")

    def test_build_resource_path_for_routes(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="demo")
        self.assertEqual(
            client.build_resource_path("routes", namespace="demo", name="frontend"),
            "/apis/route.openshift.io/v1/namespaces/demo/routes/frontend",
        )

    def test_build_resource_path_for_events(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="demo")
        self.assertEqual(
            client.build_resource_path("events", namespace="demo"),
            "/api/v1/namespaces/demo/events",
        )

    def test_resolve_namespace_uses_default(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="demo")
        self.assertEqual(client.resolve_namespace(), "demo")

    def test_resolve_namespace_requires_value(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="")
        with self.assertRaises(ValueError):
            client.resolve_namespace()

    def test_list_namespaces_sorts_and_filters_empty_names(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="demo")

        async def fake_get_json(_path: str) -> dict:
            return {
                "items": [
                    {"metadata": {"name": "zeta"}},
                    {"metadata": {"name": ""}},
                    {"metadata": {"name": "alpha"}},
                ]
            }

        client.get_json = fake_get_json  # type: ignore[method-assign]
        import asyncio

        payload = asyncio.run(client.list_namespaces())
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["items"], ["alpha", "zeta"])

    def test_list_namespaces_falls_back_to_default_namespace_on_403(self) -> None:
        client = OcpApiClient(base_url="https://api.example.com", token="token", default_namespace="demo")

        async def fake_get_json(_path: str) -> dict:
            request = httpx.Request("GET", "https://api.example.com/api/v1/namespaces")
            response = httpx.Response(status_code=403, request=request)
            raise httpx.HTTPStatusError("forbidden", request=request, response=response)

        client.get_json = fake_get_json  # type: ignore[method-assign]
        import asyncio

        payload = asyncio.run(client.list_namespaces())
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"], ["demo"])


if __name__ == "__main__":
    unittest.main()

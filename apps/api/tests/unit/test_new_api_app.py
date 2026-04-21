from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from apps.api.main import app


class NewApiAppTests(unittest.TestCase):
    def test_healthz_returns_ok(self) -> None:
        client = TestClient(app)
        response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_openapi_contains_ocp_auth_routes(self) -> None:
        client = TestClient(app)
        response = client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("/api/v1/actions/preview", payload["paths"])
        self.assertIn("/api/v1/actions/requests", payload["paths"])
        self.assertIn("/api/v1/actions/requests/{request_id}/approve", payload["paths"])
        self.assertIn("/api/v1/actions/requests/{request_id}/reject", payload["paths"])
        self.assertIn("/api/v1/actions/requests/{request_id}/execute", payload["paths"])
        self.assertIn("/api/v1/actions/executions", payload["paths"])
        self.assertIn("/api/v1/actions/audit", payload["paths"])
        self.assertIn("/api/v1/auth/ocp/connect", payload["paths"])
        self.assertIn("/api/v1/auth/ocp/test", payload["paths"])
        self.assertIn("/api/v1/auth/ocp/lease/refresh", payload["paths"])
        self.assertIn("/api/v1/auth/ocp/lease/status", payload["paths"])
        self.assertIn("/api/v1/auth/ocp/disconnect", payload["paths"])
        self.assertIn("/api/v1/chat/query", payload["paths"])
        self.assertIn("/api/v1/chat/query/stream", payload["paths"])
        self.assertIn("/api/v1/chat/live", payload["paths"])
        self.assertIn("/api/v1/docs-preview/snippet", payload["paths"])
        self.assertIn("/api/v1/index/source", payload["paths"])
        self.assertIn("/api/v1/index/batch/reindex", payload["paths"])
        self.assertIn("/api/v1/index/batch/jobs", payload["paths"])
        self.assertIn("/api/v1/index/batch/jobs/{job_id}", payload["paths"])
        self.assertIn("/api/v1/index/batch/jobs/{job_id}/retry-failed", payload["paths"])
        self.assertIn("/api/v1/index/batch/jobs/{job_id}/cancel", payload["paths"])
        self.assertIn("/api/v1/ocp/overview/{connection_id}", payload["paths"])
        self.assertIn("/api/v1/ocp/namespaces/{connection_id}", payload["paths"])
        self.assertIn("/api/v1/ocp/resources/{connection_id}", payload["paths"])


if __name__ == "__main__":
    unittest.main()


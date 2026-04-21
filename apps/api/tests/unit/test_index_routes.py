from __future__ import annotations

import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.indexing import router

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class IndexRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        self.client = TestClient(self.app)

    def test_index_route_returns_404_for_missing_source(self) -> None:
        response = self.client.post(
            "/api/v1/index/source",
            json={
                "source_type": "generated-manual",
                "source_path": str(DATA_DIR / "missing.md"),
            },
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()



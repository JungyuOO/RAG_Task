from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.workspaces import router
from apps.api.runtime import workspace_model_profile_repository, workspace_repository


class WorkspaceRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router, prefix="/api/v1")
        self.client = TestClient(self.app)
        workspace_repository.clear()
        workspace_model_profile_repository.clear()

    def test_workspace_create_list_and_model_profile_update(self) -> None:
        create_response = self.client.post(
            "/api/v1/workspaces",
            json={
                "name": "Customer A",
                "slug": "customer-a",
                "industry": "finance",
                "environment": "prod",
            },
        )
        self.assertEqual(create_response.status_code, 200)
        workspace = create_response.json()
        workspace_id = workspace["workspace_id"]

        list_response = self.client.get("/api/v1/workspaces")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()["items"]), 1)

        model_response = self.client.put(
            f"/api/v1/workspaces/{workspace_id}/models/default",
            json={
                "chat_provider": "openai-compatible",
                "chat_base_url": "https://llm.example.com/v1",
                "chat_model": "gpt-4o-mini",
                "chat_api_key_mode": "byok",
                "embedding_provider": "tei",
                "embedding_base_url": "https://embed.example.com",
                "embedding_model": "bge-m3",
                "embedding_api_key_mode": "managed",
            },
        )
        self.assertEqual(model_response.status_code, 200)
        model_payload = model_response.json()
        self.assertEqual(model_payload["workspace_id"], workspace_id)
        self.assertEqual(model_payload["chat_model"], "gpt-4o-mini")

        get_model_response = self.client.get(f"/api/v1/workspaces/{workspace_id}/models/default")
        self.assertEqual(get_model_response.status_code, 200)
        self.assertEqual(get_model_response.json()["chat_base_url"], "https://llm.example.com/v1")


if __name__ == "__main__":
    unittest.main()

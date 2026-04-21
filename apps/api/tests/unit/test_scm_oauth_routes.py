from __future__ import annotations

import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes.oauth import router as oauth_router
from apps.api.routes.workspaces import router as workspaces_router
from apps.api.runtime import (
    scm_connection_repository,
    scm_oauth_service,
    workspace_repository,
)


class ScmOauthRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(workspaces_router, prefix="/api/v1")
        self.app.include_router(oauth_router, prefix="/api/v1")
        self.client = TestClient(self.app)
        scm_connection_repository.clear()
        workspace_repository.clear()
        scm_oauth_service.state_store.clear()
        scm_oauth_service.secret_store.clear()
        scm_oauth_service.settings.scm_github_client_id = "github-client"
        scm_oauth_service.settings.scm_github_client_secret = "github-secret"
        scm_oauth_service.settings.scm_gitlab_client_id = "gitlab-client"
        scm_oauth_service.settings.scm_gitlab_client_secret = "gitlab-secret"

    def test_start_route_returns_authorize_url(self) -> None:
        workspace = self.client.post("/api/v1/workspaces", json={"name": "OAuth", "slug": "oauth"}).json()
        response = self.client.post(f"/api/v1/oauth/github/start?workspace_id={workspace['workspace_id']}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("authorize_url", payload)
        self.assertIn("client_id=github-client", payload["authorize_url"])
        self.assertTrue(payload["state"].startswith("scm-oauth-"))

    def test_callback_creates_connection_record(self) -> None:
        workspace = self.client.post("/api/v1/workspaces", json={"name": "OAuth", "slug": "oauth"}).json()
        start_response = self.client.post(f"/api/v1/oauth/github/start?workspace_id={workspace['workspace_id']}")
        state = start_response.json()["state"]

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == scm_oauth_service.settings.scm_github_token_url:
                return httpx.Response(
                    200,
                    json={
                        "access_token": "gho_test_token",
                        "token_type": "bearer",
                        "scope": "read:user repo",
                    },
                )
            if str(request.url) == scm_oauth_service.settings.scm_github_user_url:
                return httpx.Response(
                    200,
                    json={
                        "login": "octo-user",
                        "name": "Octo User",
                    },
                )
            return httpx.Response(404, json={"message": "not found"})

        scm_oauth_service.transport = httpx.MockTransport(handler)
        callback_response = self.client.get(
            f"/api/v1/oauth/github/callback?code=test-code&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(callback_response.status_code, 303)
        self.assertIn("/scm?oauth_status=connected", callback_response.headers["location"])

        connections = scm_connection_repository.list_by_workspace(workspace["workspace_id"]).items
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0].auth_type, "oauth")
        self.assertEqual(connections[0].login_name, "octo-user")
        self.assertTrue(connections[0].secret_ref)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode
from uuid import uuid4

import httpx

from apps.api.core.scm_oauth_settings import ScmOauthSettings
from apps.api.ocp.auth.broker import InMemoryConnectionSecretStore
from apps.api.schemas.auth import OcpAuthMode
from apps.api.schemas.scm import ScmConnectionCreateRequest, ScmConnectionRecord

ScmProvider = Literal["github", "gitlab"]


@dataclass
class PendingScmOauthState:
    workspace_id: str
    provider: ScmProvider


class InMemoryScmOauthStateStore:
    def __init__(self) -> None:
        self._store: dict[str, PendingScmOauthState] = {}

    def create(self, *, workspace_id: str, provider: ScmProvider) -> str:
        state = f"scm-oauth-{uuid4().hex}"
        self._store[state] = PendingScmOauthState(workspace_id=workspace_id, provider=provider)
        return state

    def pop(self, state: str) -> PendingScmOauthState | None:
        return self._store.pop(state, None)

    def clear(self) -> None:
        self._store.clear()


class ScmOauthService:
    def __init__(
        self,
        *,
        settings: ScmOauthSettings,
        secret_store: InMemoryConnectionSecretStore,
        transport: httpx.BaseTransport | None = None,
        state_store: InMemoryScmOauthStateStore | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.settings = settings
        self.secret_store = secret_store
        self.transport = transport
        self.state_store = state_store or InMemoryScmOauthStateStore()
        self.timeout = timeout

    def build_authorize_url(self, *, provider: ScmProvider, workspace_id: str, callback_url: str) -> tuple[str, str]:
        config = self._provider_config(provider)
        state = self.state_store.create(workspace_id=workspace_id, provider=provider)
        query = urlencode(
            {
                "client_id": config["client_id"],
                "redirect_uri": callback_url,
                "response_type": "code",
                "scope": config["scope"],
                "state": state,
            }
        )
        return f"{config['authorize_url']}?{query}", state

    def complete_callback(
        self,
        *,
        provider: ScmProvider,
        code: str,
        state: str,
        callback_url: str,
        connection_repository,
    ) -> ScmConnectionRecord:
        pending = self.state_store.pop(state)
        if pending is None or pending.provider != provider:
            raise ValueError("OAuth state is invalid or expired.")

        config = self._provider_config(provider)
        token_payload = self._exchange_code(
            provider=provider,
            code=code,
            callback_url=callback_url,
            config=config,
        )
        user_payload = self._fetch_user(provider=provider, access_token=str(token_payload.get("access_token") or ""), config=config)

        secret_ref = self.secret_store.put(
            auth_mode=OcpAuthMode.TOKEN,
            payload={
                "provider": provider,
                "access_token": str(token_payload.get("access_token") or ""),
                "refresh_token": str(token_payload.get("refresh_token") or ""),
                "token_type": str(token_payload.get("token_type") or "Bearer"),
                "scope": str(token_payload.get("scope") or config["scope"]),
            },
        )
        login_name = str(user_payload.get("login") or user_payload.get("username") or "").strip()
        account_label = str(user_payload.get("name") or login_name or f"{provider} account").strip()
        scopes = [scope for scope in str(token_payload.get("scope") or config["scope"]).replace(",", " ").split() if scope]
        return connection_repository.create(
            pending.workspace_id,
            ScmConnectionCreateRequest(
                provider=provider,
                host_url=config["host_url"],
                auth_type="oauth",
                account_label=account_label,
                login_name=login_name,
                scopes=scopes,
                secret_ref=secret_ref,
            ),
        )

    def _exchange_code(self, *, provider: ScmProvider, code: str, callback_url: str, config: dict[str, str]) -> dict:
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            response = client.post(
                config["token_url"],
                headers={"Accept": "application/json"},
                data={
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"],
                    "code": code,
                    "redirect_uri": callback_url,
                    "grant_type": "authorization_code",
                },
            )
            response.raise_for_status()
            return response.json()

    def _fetch_user(self, *, provider: ScmProvider, access_token: str, config: dict[str, str]) -> dict:
        if not access_token:
            raise ValueError("OAuth token exchange did not return an access token.")
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            response = client.get(
                config["user_url"],
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()

    def _provider_config(self, provider: ScmProvider) -> dict[str, str]:
        if provider == "github":
            config = {
                "client_id": self.settings.scm_github_client_id,
                "client_secret": self.settings.scm_github_client_secret,
                "scope": self.settings.scm_github_scope,
                "authorize_url": self.settings.scm_github_authorize_url,
                "token_url": self.settings.scm_github_token_url,
                "user_url": self.settings.scm_github_user_url,
                "host_url": self.settings.scm_github_host_url,
            }
        else:
            config = {
                "client_id": self.settings.scm_gitlab_client_id,
                "client_secret": self.settings.scm_gitlab_client_secret,
                "scope": self.settings.scm_gitlab_scope,
                "authorize_url": self.settings.scm_gitlab_authorize_url,
                "token_url": self.settings.scm_gitlab_token_url,
                "user_url": self.settings.scm_gitlab_user_url,
                "host_url": self.settings.scm_gitlab_host_url,
            }
        if not config["client_id"] or not config["client_secret"]:
            raise ValueError(f"{provider} OAuth is not configured.")
        return config

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import httpx

from apps.api.storage.json_persistence import load_json_file, save_json_file
from apps.api.schemas.auth import OcpAuthMode
from apps.api.ocp.auth.secret_store_types import StoredConnectionSecret


class VaultSecretClient(Protocol):
    def put_secret(self, secret_ref: str, payload: dict) -> dict[str, object]: ...

    def get_secret(self, secret_ref: str) -> dict | None: ...

    def delete_secret(self, secret_ref: str) -> None: ...

    def describe_secret(self, secret_ref: str) -> dict[str, object]: ...

    def renew_secret_lease(self, secret_ref: str, *, increment: str = "") -> dict[str, object]: ...


class VaultHttpSecretClient:
    """Minimal generic HTTP vault client for secret envelopes.

    Expected contract:
    - POST   {base_url}/v1/secrets/{secret_ref} with JSON body
    - GET    {base_url}/v1/secrets/{secret_ref} -> JSON or 404
    - DELETE {base_url}/v1/secrets/{secret_ref}
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.token = token
        self.transport = transport
        self.timeout = timeout

    def put_secret(self, secret_ref: str, payload: dict) -> None:
        response = self._request("POST", secret_ref, json_body=payload)
        response.raise_for_status()
        return {
            "secret_backend": "vault_http",
            "rotation_supported": False,
            "lease_renewable": False,
        }

    def get_secret(self, secret_ref: str) -> dict | None:
        response = self._request("GET", secret_ref)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json() if response.content else {}

    def delete_secret(self, secret_ref: str) -> None:
        response = self._request("DELETE", secret_ref)
        if response.status_code not in {200, 204, 404}:
            response.raise_for_status()

    def describe_secret(self, secret_ref: str) -> dict[str, object]:
        return {
            "secret_backend": "vault_http",
            "rotation_supported": False,
            "lease_renewable": False,
        }

    def renew_secret_lease(self, secret_ref: str, *, increment: str = "") -> dict[str, object]:
        return {
            "secret_backend": "vault_http",
            "rotation_supported": False,
            "lease_renewable": False,
            "renewed": False,
            "renew_message": "Generic vault_http backend does not support lease renewal.",
        }

    def _request(self, method: str, secret_ref: str, *, json_body: dict | None = None) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            return client.request(
                method=method,
                url=f"{self.base_url}/v1/secrets/{secret_ref}",
                headers=headers,
                json=json_body,
            )


class HashiCorpVaultKvV2SecretClient:
    """HashiCorp Vault KV v2 HTTP client.

    Uses the official KV v2 API layout:
    - POST   /v1/{mount}/data/{path}
    - GET    /v1/{mount}/data/{path}
    - DELETE /v1/{mount}/metadata/{path}
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        mount_path: str = "secret",
        path_prefix: str = "rag-task/ocp-connections",
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.token = token
        self.mount_path = str(mount_path or "secret").strip("/")
        self.path_prefix = str(path_prefix or "rag-task/ocp-connections").strip("/")
        self.transport = transport
        self.timeout = timeout

    def put_secret(self, secret_ref: str, payload: dict) -> None:
        response = self._request(
            "POST",
            self._data_path(secret_ref),
            json_body={"data": payload},
        )
        response.raise_for_status()
        metadata = ((response.json().get("data") or {}) if response.content else {}) if response.content else {}
        described = self.describe_secret(secret_ref)
        if "version" in metadata:
            described["secret_version"] = metadata.get("version")
        return described

    def get_secret(self, secret_ref: str) -> dict | None:
        response = self._request("GET", self._data_path(secret_ref))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json() if response.content else {}
        data = ((payload.get("data") or {}).get("data")) if isinstance(payload, dict) else None
        return dict(data or {})

    def delete_secret(self, secret_ref: str) -> None:
        response = self._request("DELETE", self._metadata_path(secret_ref))
        if response.status_code not in {200, 204, 404}:
            response.raise_for_status()

    def describe_secret(self, secret_ref: str) -> dict[str, object]:
        metadata: dict[str, object] = {
            "secret_backend": "vault_hashicorp",
            "rotation_supported": True,
        }

        secret_response = self._request("GET", self._data_path(secret_ref))
        if secret_response.status_code != 404:
            secret_response.raise_for_status()
            payload = secret_response.json() if secret_response.content else {}
            secret_metadata = ((payload.get("data") or {}).get("metadata")) if isinstance(payload, dict) else {}
            if isinstance(secret_metadata, dict):
                metadata["secret_version"] = secret_metadata.get("version")
                metadata["secret_created_at"] = secret_metadata.get("created_time") or ""
                metadata["secret_destroyed"] = bool(secret_metadata.get("destroyed") or False)

        token_response = self._request("GET", "/v1/auth/token/lookup-self")
        token_response.raise_for_status()
        token_payload = token_response.json() if token_response.content else {}
        token_data = token_payload.get("data", {}) if isinstance(token_payload, dict) else {}
        metadata["lease_renewable"] = bool(token_data.get("renewable") or False)
        metadata["lease_ttl_seconds"] = int(token_data.get("ttl") or 0)
        metadata["lease_expires_at"] = str(token_data.get("expire_time") or "")
        metadata["lease_id"] = str(token_data.get("id") or "")
        metadata["lease_last_checked_at"] = str(token_data.get("last_renewal_time") or "")
        return metadata

    def renew_secret_lease(self, secret_ref: str, *, increment: str = "") -> dict[str, object]:
        payload = {"increment": increment} if increment else None
        response = self._request("POST", "/v1/auth/token/renew-self", json_body=payload)
        response.raise_for_status()
        metadata = self.describe_secret(secret_ref)
        metadata["renewed"] = True
        auth_payload = response.json().get("auth", {}) if response.content else {}
        if isinstance(auth_payload, dict):
            metadata["lease_ttl_seconds"] = int(auth_payload.get("lease_duration") or metadata.get("lease_ttl_seconds") or 0)
            metadata["lease_renewable"] = bool(auth_payload.get("renewable") or metadata.get("lease_renewable") or False)
        metadata["renew_message"] = "Vault token lease renewed via renew-self."
        return metadata

    def _data_path(self, secret_ref: str) -> str:
        return f"/v1/{self.mount_path}/data/{self.path_prefix}/{secret_ref}"

    def _metadata_path(self, secret_ref: str) -> str:
        return f"/v1/{self.mount_path}/metadata/{self.path_prefix}/{secret_ref}"

    def _request(self, method: str, path: str, *, json_body: dict | None = None) -> httpx.Response:
        headers = {
            "X-Vault-Token": self.token,
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            return client.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=headers,
                json=json_body,
            )


class VaultHttpConnectionSecretStore:
    """Connection secret store backed by an external HTTP vault service."""

    def __init__(
        self,
        *,
        client: VaultSecretClient,
        refs_path: Path | None = None,
    ) -> None:
        self.client = client
        self.refs_path = refs_path
        self._refs, self._metadata_cache = self._load_refs()

    def put(self, *, auth_mode: OcpAuthMode, payload: dict) -> str:
        secret_ref = f"ocp-secret-{uuid4().hex}"
        metadata = self.client.put_secret(
            secret_ref,
            {
                "auth_mode": auth_mode.value,
                "payload": dict(payload),
            },
        )
        self._refs.add(secret_ref)
        self._metadata_cache[secret_ref] = dict(metadata or {})
        self._save_refs()
        return secret_ref

    def get(self, secret_ref: str) -> StoredConnectionSecret | None:
        payload = self.client.get_secret(secret_ref)
        if payload is None:
            return None
        return StoredConnectionSecret(
            auth_mode=OcpAuthMode(payload["auth_mode"]),
            payload=dict(payload.get("payload") or {}),
            metadata=dict(self._metadata_cache.get(secret_ref) or {}),
        )

    def describe(
        self,
        secret_ref: str,
        *,
        refresh: bool = False,
        renew: bool = False,
        auto_renew: bool = False,
        renew_threshold_seconds: int = 0,
        renew_increment: str = "",
    ) -> dict[str, object]:
        if renew:
            metadata = self.client.renew_secret_lease(secret_ref, increment=renew_increment)
            self._metadata_cache[secret_ref] = dict(metadata or {})
            self._save_refs()
            return dict(self._metadata_cache[secret_ref])
        if refresh or secret_ref not in self._metadata_cache:
            metadata = self.client.describe_secret(secret_ref)
            self._metadata_cache[secret_ref] = dict(metadata or {})
            self._save_refs()
        if auto_renew and self._should_auto_renew(self._metadata_cache.get(secret_ref) or {}, renew_threshold_seconds):
            metadata = self.client.renew_secret_lease(secret_ref, increment=renew_increment)
            metadata["auto_renew_applied"] = True
            metadata["auto_renew_threshold_seconds"] = renew_threshold_seconds
            self._metadata_cache[secret_ref] = dict(metadata or {})
            self._save_refs()
        return dict(self._metadata_cache.get(secret_ref) or {})

    def delete(self, secret_ref: str) -> None:
        self.client.delete_secret(secret_ref)
        self._refs.discard(secret_ref)
        self._metadata_cache.pop(secret_ref, None)
        self._save_refs()

    def clear(self) -> None:
        for secret_ref in list(self._refs):
            self.client.delete_secret(secret_ref)
        self._refs.clear()
        self._metadata_cache.clear()
        self._save_refs()

    def _load_refs(self) -> tuple[set[str], dict[str, dict[str, object]]]:
        if self.refs_path is None:
            return set(), {}
        payload = load_json_file(self.refs_path, default={"refs": []})
        refs = {str(item) for item in payload.get("refs", []) if str(item).strip()}
        metadata = {
            str(key): dict(value or {})
            for key, value in dict(payload.get("metadata") or {}).items()
        }
        return refs, metadata

    def _save_refs(self) -> None:
        if self.refs_path is None:
            return
        save_json_file(
            self.refs_path,
            {
                "refs": sorted(self._refs),
                "metadata": self._metadata_cache,
            },
        )

    @staticmethod
    def _should_auto_renew(metadata: dict[str, object], renew_threshold_seconds: int) -> bool:
        if renew_threshold_seconds <= 0:
            return False
        lease_renewable = bool(metadata.get("lease_renewable") or False)
        ttl_seconds = int(metadata.get("lease_ttl_seconds") or 0)
        return lease_renewable and 0 < ttl_seconds <= renew_threshold_seconds


def build_vault_http_secret_store(*, refs_path: Path | None = None) -> VaultHttpConnectionSecretStore:
    base_url = str(os.environ.get("RAG_TASK_VAULT_URL") or "").strip()
    token = str(os.environ.get("RAG_TASK_VAULT_TOKEN") or "").strip()
    if not base_url:
        raise ValueError("RAG_TASK_VAULT_URL is required when RAG_TASK_SECRET_BACKEND=vault_http.")
    if not token:
        raise ValueError("RAG_TASK_VAULT_TOKEN is required when RAG_TASK_SECRET_BACKEND=vault_http.")
    return VaultHttpConnectionSecretStore(
        client=VaultHttpSecretClient(base_url=base_url, token=token),
        refs_path=refs_path,
    )


def build_hashicorp_vault_secret_store(*, refs_path: Path | None = None) -> VaultHttpConnectionSecretStore:
    base_url = str(os.environ.get("RAG_TASK_VAULT_URL") or "").strip()
    token = str(os.environ.get("RAG_TASK_VAULT_TOKEN") or "").strip()
    mount_path = str(os.environ.get("RAG_TASK_VAULT_MOUNT") or "secret").strip()
    path_prefix = str(os.environ.get("RAG_TASK_VAULT_PREFIX") or "rag-task/ocp-connections").strip()
    if not base_url:
        raise ValueError("RAG_TASK_VAULT_URL is required when RAG_TASK_SECRET_BACKEND=vault_hashicorp.")
    if not token:
        raise ValueError("RAG_TASK_VAULT_TOKEN is required when RAG_TASK_SECRET_BACKEND=vault_hashicorp.")
    return VaultHttpConnectionSecretStore(
        client=HashiCorpVaultKvV2SecretClient(
            base_url=base_url,
            token=token,
            mount_path=mount_path,
            path_prefix=path_prefix,
        ),
        refs_path=refs_path,
    )





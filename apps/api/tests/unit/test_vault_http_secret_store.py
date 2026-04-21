from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from apps.api.schemas.ocp_auth import OcpAuthMode
from apps.api.ocp.auth import (
    HashiCorpVaultKvV2SecretClient,
    VaultHttpConnectionSecretStore,
    VaultHttpSecretClient,
    build_default_connection_secret_store,
)

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "vault_http_secret_store"


class VaultHttpSecretStoreTests(unittest.TestCase):
    def tearDown(self) -> None:
        if RESULT_DIR.exists():
            shutil.rmtree(RESULT_DIR, ignore_errors=True)

    def test_vault_http_store_round_trip_and_clear(self) -> None:
        state: dict[str, dict] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            secret_ref = request.url.path.rsplit("/", 1)[-1]
            if request.method == "POST":
                state[secret_ref] = request.content.decode("utf-8")
                return httpx.Response(200, json={"stored": True})
            if request.method == "GET":
                if secret_ref not in state:
                    return httpx.Response(404)
                return httpx.Response(200, content=state[secret_ref], headers={"Content-Type": "application/json"})
            if request.method == "DELETE":
                state.pop(secret_ref, None)
                return httpx.Response(204)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        store = VaultHttpConnectionSecretStore(
            client=VaultHttpSecretClient(
                base_url="https://vault.example.com",
                token="vault-token",
                transport=httpx.MockTransport(handler),
            ),
            refs_path=RESULT_DIR / "refs.json",
        )

        secret_ref = store.put(auth_mode=OcpAuthMode.TOKEN, payload={"token": "sha256~vault"})
        loaded = store.get(secret_ref)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.payload["token"], "sha256~vault")

        reloaded = VaultHttpConnectionSecretStore(
            client=VaultHttpSecretClient(
                base_url="https://vault.example.com",
                token="vault-token",
                transport=httpx.MockTransport(handler),
            ),
            refs_path=RESULT_DIR / "refs.json",
        )
        again = reloaded.get(secret_ref)
        self.assertIsNotNone(again)
        reloaded.clear()
        self.assertEqual(state, {})

    def test_default_secret_store_factory_selects_vault_http(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_TASK_SECRET_BACKEND": "vault_http",
                "RAG_TASK_VAULT_URL": "https://vault.example.com",
                "RAG_TASK_VAULT_TOKEN": "vault-token",
            },
            clear=False,
        ):
            store = build_default_connection_secret_store(refs_path=RESULT_DIR / "refs.json")
        self.assertIsInstance(store, VaultHttpConnectionSecretStore)

    def test_hashicorp_vault_kv_v2_store_round_trip(self) -> None:
        state: dict[str, dict] = {}
        renew_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-Vault-Token"], "vault-token")
            secret_ref = request.url.path.rsplit("/", 1)[-1]
            nonlocal renew_calls
            if request.method == "POST":
                if request.url.path.endswith("/renew-self"):
                    renew_calls += 1
                    return httpx.Response(200, json={"auth": {"renewable": True, "lease_duration": 1800}})
                payload = json.loads(request.content.decode("utf-8"))
                state[secret_ref] = payload["data"]
                return httpx.Response(200, json={"data": {"version": 1}})
            if request.method == "GET":
                if request.url.path.endswith("/lookup-self"):
                    return httpx.Response(
                        200,
                        json={"data": {"ttl": 1800, "renewable": True, "expire_time": "2026-04-14T12:00:00Z", "id": "token-1"}},
                    )
                if secret_ref not in state:
                    return httpx.Response(404)
                return httpx.Response(
                    200,
                    json={"data": {"data": state[secret_ref], "metadata": {"version": 1, "created_time": "2026-04-14T11:30:00Z"}}},
                )
            if request.method == "DELETE":
                state.pop(secret_ref, None)
                return httpx.Response(204)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        store = VaultHttpConnectionSecretStore(
            client=HashiCorpVaultKvV2SecretClient(
                base_url="https://vault.example.com",
                token="vault-token",
                mount_path="secret",
                path_prefix="rag-task/ocp-connections",
                transport=httpx.MockTransport(handler),
            ),
            refs_path=RESULT_DIR / "hashi-refs.json",
        )

        secret_ref = store.put(auth_mode=OcpAuthMode.TOKEN, payload={"token": "sha256~hashi"})
        loaded = store.get(secret_ref)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.payload["token"], "sha256~hashi")

        metadata = store.describe(secret_ref, refresh=True)
        self.assertEqual(metadata["secret_backend"], "vault_hashicorp")
        self.assertTrue(metadata["lease_renewable"])
        self.assertEqual(metadata["secret_version"], 1)

        renewed = store.describe(secret_ref, renew=True)
        self.assertTrue(renewed["renewed"])
        self.assertEqual(renew_calls, 1)

        store.delete(secret_ref)
        self.assertEqual(state, {})

    def test_hashicorp_vault_kv_v2_auto_renews_when_ttl_is_low(self) -> None:
        state: dict[str, dict] = {}
        renew_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-Vault-Token"], "vault-token")
            secret_ref = request.url.path.rsplit("/", 1)[-1]
            nonlocal renew_calls
            if request.method == "POST":
                if request.url.path.endswith("/renew-self"):
                    renew_calls += 1
                    return httpx.Response(200, json={"auth": {"renewable": True, "lease_duration": 1800}})
                payload = json.loads(request.content.decode("utf-8"))
                state[secret_ref] = payload["data"]
                return httpx.Response(200, json={"data": {"version": 2}})
            if request.method == "GET":
                if request.url.path.endswith("/lookup-self"):
                    ttl = 300 if renew_calls == 0 else 1800
                    return httpx.Response(
                        200,
                        json={"data": {"ttl": ttl, "renewable": True, "expire_time": "2026-04-14T12:00:00Z", "id": "token-1"}},
                    )
                if secret_ref not in state:
                    return httpx.Response(404)
                return httpx.Response(
                    200,
                    json={"data": {"data": state[secret_ref], "metadata": {"version": 2, "created_time": "2026-04-14T11:30:00Z"}}},
                )
            if request.method == "DELETE":
                state.pop(secret_ref, None)
                return httpx.Response(204)
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        store = VaultHttpConnectionSecretStore(
            client=HashiCorpVaultKvV2SecretClient(
                base_url="https://vault.example.com",
                token="vault-token",
                mount_path="secret",
                path_prefix="rag-task/ocp-connections",
                transport=httpx.MockTransport(handler),
            ),
            refs_path=RESULT_DIR / "hashi-auto-refs.json",
        )

        secret_ref = store.put(auth_mode=OcpAuthMode.TOKEN, payload={"token": "sha256~hashi"})
        metadata = store.describe(secret_ref, auto_renew=True, renew_threshold_seconds=600)
        self.assertTrue(metadata["auto_renew_applied"])
        self.assertEqual(metadata["auto_renew_threshold_seconds"], 600)
        self.assertEqual(renew_calls, 1)

    def test_default_secret_store_factory_selects_hashicorp_vault(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_TASK_SECRET_BACKEND": "vault_hashicorp",
                "RAG_TASK_VAULT_URL": "https://vault.example.com",
                "RAG_TASK_VAULT_TOKEN": "vault-token",
                "RAG_TASK_VAULT_MOUNT": "secret",
                "RAG_TASK_VAULT_PREFIX": "rag-task/ocp-connections",
            },
            clear=False,
        ):
            store = build_default_connection_secret_store(refs_path=RESULT_DIR / "hashi-refs.json")
        self.assertIsInstance(store, VaultHttpConnectionSecretStore)


if __name__ == "__main__":
    unittest.main()





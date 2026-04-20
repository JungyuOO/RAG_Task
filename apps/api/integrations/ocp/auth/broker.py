from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from apps.api.persistence.json_persistence import load_json_file, save_json_file
from apps.api.persistence.protected_secret_persistence import (
    SecretProtectionError,
    SecretProtector,
    load_protected_json_file,
    save_protected_json_file,
)
from apps.api.api.schemas.auth import (
    OcpAuthMode,
    OcpConnectionProfile,
    OcpConnectionRequest,
    OcpConnectionTestResult,
)
from apps.api.integrations.ocp.auth.secret_store_types import StoredConnectionSecret
from apps.api.integrations.ocp.auth.vault_store import build_hashicorp_vault_secret_store, build_vault_http_secret_store


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name) or "").strip().casefold()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


class InMemoryConnectionSecretStore:
    """Protected-at-rest connection secret store."""

    def __init__(
        self,
        *,
        storage_path: Path | None = None,
        protector: SecretProtector | None = None,
    ) -> None:
        self._store: dict[str, StoredConnectionSecret] = {}
        self.storage_path = storage_path
        self.protector = protector or SecretProtector()
        self._load()

    def put(self, *, auth_mode: OcpAuthMode, payload: dict[str, Any]) -> str:
        secret_ref = f"ocp-secret-{uuid4().hex}"
        self._store[secret_ref] = StoredConnectionSecret(
            auth_mode=auth_mode,
            payload=dict(payload),
            metadata={
                "secret_backend": "protected_file",
                "secret_format": self.protector.format_name,
                "rotation_supported": False,
                "lease_renewable": False,
            },
        )
        self._save()
        return secret_ref

    def get(self, secret_ref: str) -> StoredConnectionSecret | None:
        return self._store.get(secret_ref)

    def delete(self, secret_ref: str) -> None:
        self._store.pop(secret_ref, None)
        self._save()

    def clear(self) -> None:
        self._store.clear()
        self._save()

    def describe(
        self,
        secret_ref: str,
        *,
        refresh: bool = False,
        renew: bool = False,
        auto_renew: bool = False,
        renew_threshold_seconds: int = 0,
        renew_increment: str = "",
    ) -> dict[str, Any]:
        secret = self._store.get(secret_ref)
        if secret is None:
            raise LookupError(f"Missing secret for secret_ref={secret_ref}")
        return dict(secret.metadata)

    def _load(self) -> None:
        if self.storage_path is None:
            return
        payload, rewrite_required = load_protected_json_file(
            self.storage_path,
            protector=self.protector,
            default={},
        )
        self._store = {
            secret_ref: StoredConnectionSecret(
                auth_mode=OcpAuthMode(item["auth_mode"]),
                payload=dict(item.get("payload") or {}),
                metadata=dict(item.get("metadata") or {}),
            )
            for secret_ref, item in dict(payload).items()
        }
        if rewrite_required:
            self._save()

    def _save(self) -> None:
        if self.storage_path is None:
            return
        save_protected_json_file(
            self.storage_path,
            {
                secret_ref: {
                    "auth_mode": secret.auth_mode.value,
                    "payload": secret.payload,
                    "metadata": secret.metadata,
                }
                for secret_ref, secret in self._store.items()
            },
            protector=self.protector,
        )


class InMemoryConnectionProfileStore:
    """Ephemeral connection profile registry for scaffold/testing."""

    def __init__(self, *, storage_path: Path | None = None) -> None:
        self._store: dict[str, OcpConnectionProfile] = {}
        self.storage_path = storage_path
        self._load()

    def put(self, profile: OcpConnectionProfile) -> None:
        self._store[profile.connection_id] = profile
        self._save()

    def get(self, connection_id: str) -> OcpConnectionProfile | None:
        return self._store.get(connection_id)

    def list_profiles(self) -> list[OcpConnectionProfile]:
        return list(self._store.values())

    def delete(self, connection_id: str) -> OcpConnectionProfile | None:
        removed = self._store.pop(connection_id, None)
        self._save()
        return removed

    def clear(self) -> None:
        self._store.clear()
        self._save()

    def _load(self) -> None:
        if self.storage_path is None:
            return
        payload = load_json_file(self.storage_path, default={})
        self._store = {
            connection_id: OcpConnectionProfile.model_validate(item)
            for connection_id, item in dict(payload).items()
        }

    def _save(self) -> None:
        if self.storage_path is None:
            return
        save_json_file(
            self.storage_path,
            {
                connection_id: profile.model_dump(mode="json")
                for connection_id, profile in self._store.items()
            },
        )


class OcpConnectionBroker:
    """Builds session-scoped OCP connection profiles without exposing secrets."""

    def __init__(
        self,
        secret_store: InMemoryConnectionSecretStore | None = None,
        profile_store: InMemoryConnectionProfileStore | None = None,
    ) -> None:
        self.secret_store = secret_store or build_default_connection_secret_store()
        self.profile_store = profile_store or InMemoryConnectionProfileStore()

    def create_profile(self, request: OcpConnectionRequest) -> OcpConnectionProfile:
        secret_payload = self._extract_secret_payload(request)
        secret_ref = self.secret_store.put(auth_mode=request.auth_mode, payload=secret_payload)
        display_name = request.display_name or request.cluster_url
        profile = OcpConnectionProfile(
            connection_id=f"ocp-conn-{uuid4().hex}",
            display_name=display_name,
            cluster_url=request.cluster_url,
            auth_mode=request.auth_mode,
            verify_ssl=request.verify_ssl,
            default_namespace=request.default_namespace,
            username_hint=request.username,
            secret_ref=secret_ref,
            save_profile=request.save_profile,
            status="connected",
            last_verified_at=_utc_now_iso(),
            metadata=dict(request.metadata),
        )
        self.profile_store.put(profile)
        return profile

    def get_profile(self, connection_id: str) -> OcpConnectionProfile | None:
        return self.profile_store.get(connection_id)

    def build_runtime_config(self, profile: OcpConnectionProfile) -> dict[str, Any]:
        stored = self.secret_store.get(profile.secret_ref)
        if stored is None:
            raise LookupError(f"Missing secret for secret_ref={profile.secret_ref}")
        secret_metadata = self.describe_secret(profile, auto_renew=True)

        runtime = {
            "base_url": profile.cluster_url,
            "verify_ssl": profile.verify_ssl,
            "default_namespace": profile.default_namespace,
            "auth_mode": profile.auth_mode.value,
            "secret_metadata": secret_metadata or dict(stored.metadata or {}),
        }
        if stored.auth_mode == OcpAuthMode.TOKEN:
            runtime["token"] = str(stored.payload.get("token") or "")
            runtime["exchange_required"] = False
        elif stored.auth_mode == OcpAuthMode.PASSWORD:
            runtime["username"] = str(stored.payload.get("username") or "")
            runtime["password"] = str(stored.payload.get("password") or "")
            runtime["exchange_required"] = True
        else:
            runtime["exchange_required"] = True
        return runtime

    def describe_secret(
        self,
        profile: OcpConnectionProfile,
        *,
        refresh: bool = False,
        renew: bool = False,
        auto_renew: bool = False,
    ) -> dict[str, Any]:
        if hasattr(self.secret_store, "describe"):
            return dict(
                self.secret_store.describe(
                    profile.secret_ref,
                    refresh=refresh,
                    renew=renew,
                    auto_renew=auto_renew and _env_flag("RAG_TASK_VAULT_AUTO_RENEW", default=True),
                    renew_threshold_seconds=int(os.environ.get("RAG_TASK_VAULT_RENEW_THRESHOLD_SECONDS") or 600),
                    renew_increment=str(os.environ.get("RAG_TASK_VAULT_RENEW_INCREMENT") or "").strip(),
                )
            )
        return {}

    def build_success_result(
        self,
        profile: OcpConnectionProfile,
        *,
        resolved_user: str = "",
        resolved_groups: list[str] | None = None,
        resolved_roles: list[str] | None = None,
        identity_source: str = "",
        permission_hints: dict[str, bool] | None = None,
        rbac_evidence: list[str] | None = None,
        rbac_rules_incomplete: bool = False,
        rbac_evaluation_error: str = "",
        secret_status: dict[str, Any] | None = None,
        resolved_namespace: str = "",
        expires_at: str = "",
        message: str = "Connection verified.",
    ) -> OcpConnectionTestResult:
        secret_status = dict(secret_status or {})
        return OcpConnectionTestResult(
            success=True,
            connection_id=profile.connection_id,
            cluster_url=profile.cluster_url,
            auth_mode=profile.auth_mode,
            resolved_user=resolved_user,
            resolved_groups=list(resolved_groups or []),
            resolved_roles=list(resolved_roles or []),
            identity_source=identity_source,
            permission_hints=dict(permission_hints or {}),
            rbac_evidence=list(rbac_evidence or []),
            rbac_rules_incomplete=rbac_rules_incomplete,
            rbac_evaluation_error=rbac_evaluation_error,
            secret_backend=str(secret_status.get("secret_backend") or ""),
            secret_version=str(secret_status.get("secret_version") or ""),
            secret_created_at=str(secret_status.get("secret_created_at") or ""),
            secret_lease_renewable=bool(secret_status.get("lease_renewable") or False),
            secret_lease_ttl_seconds=int(secret_status.get("lease_ttl_seconds") or 0),
            secret_lease_expires_at=str(secret_status.get("lease_expires_at") or ""),
            secret_rotation_supported=bool(secret_status.get("rotation_supported") or False),
            secret_auto_renew_applied=bool(secret_status.get("auto_renew_applied") or False),
            secret_auto_renew_threshold_seconds=int(secret_status.get("auto_renew_threshold_seconds") or 0),
            secret_renew_message=str(secret_status.get("renew_message") or ""),
            resolved_namespace=resolved_namespace or profile.default_namespace,
            expires_at=expires_at,
            message=message,
        )

    def build_failure_result(self, profile: OcpConnectionProfile, *, error: str) -> OcpConnectionTestResult:
        return OcpConnectionTestResult(
            success=False,
            connection_id=profile.connection_id,
            cluster_url=profile.cluster_url,
            auth_mode=profile.auth_mode,
            resolved_namespace=profile.default_namespace,
            error=error,
            message="Connection failed.",
        )

    def disconnect(self, profile: OcpConnectionProfile) -> None:
        self.secret_store.delete(profile.secret_ref)
        self.profile_store.delete(profile.connection_id)

    def disconnect_by_id(self, connection_id: str) -> OcpConnectionProfile | None:
        profile = self.profile_store.get(connection_id)
        if profile is None:
            return None
        self.disconnect(profile)
        return profile

    @staticmethod
    def _extract_secret_payload(request: OcpConnectionRequest) -> dict[str, Any]:
        if request.auth_mode == OcpAuthMode.TOKEN:
            return {
                "token": request.token.get_secret_value() if request.token else "",
            }
        if request.auth_mode == OcpAuthMode.PASSWORD:
            return {
                "username": request.username,
                "password": request.password.get_secret_value() if request.password else "",
            }
        return {}


def build_default_connection_secret_store(
    *,
    storage_path: Path | None = None,
    refs_path: Path | None = None,
    backend: str | None = None,
):
    backend = str(backend or os.environ.get("RAG_TASK_SECRET_BACKEND") or "").strip().casefold()
    if backend == "vault_http":
        return build_vault_http_secret_store(refs_path=refs_path)
    if backend == "vault_hashicorp":
        return build_hashicorp_vault_secret_store(refs_path=refs_path)
    protector = SecretProtector(backend=backend) if backend else None
    return InMemoryConnectionSecretStore(storage_path=storage_path, protector=protector)



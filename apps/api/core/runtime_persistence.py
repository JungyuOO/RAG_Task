from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Literal


PersistenceMode = Literal["development", "production"]
MANAGED_SECRET_BACKENDS = {"env_key", "vault_http", "vault_hashicorp"}


def _env_flag(environ: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = str(environ.get(name) or "").strip().casefold()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RuntimePersistenceProfile:
    mode: PersistenceMode
    state_root: Path
    state_db_path: Path
    secret_storage_path: Path
    secret_refs_path: Path
    legacy_migration_enabled: bool
    require_managed_secret_backend: bool
    secret_backend: str
    secret_backend_source: str

    def legacy_json_path(self, filename: str) -> Path | None:
        if not self.legacy_migration_enabled:
            return None
        return self.state_root / filename


def load_runtime_persistence_profile(
    *,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimePersistenceProfile:
    env = environ or os.environ
    base_cwd = (cwd or Path.cwd()).resolve()

    mode_raw = str(env.get("RAG_TASK_RUNTIME_PERSISTENCE_MODE") or "").strip().casefold()
    mode: PersistenceMode = "production" if mode_raw in {"prod", "production"} else "development"

    state_root_raw = str(env.get("RAG_TASK_RUNTIME_STATE_DIR") or "").strip()
    state_root = Path(state_root_raw).expanduser() if state_root_raw else (base_cwd / "data" / "runtime_state")

    state_db_raw = str(env.get("RAG_TASK_RUNTIME_DB_PATH") or "").strip()
    state_db_path = Path(state_db_raw).expanduser() if state_db_raw else (state_root / "runtime_state.sqlite3")

    secret_storage_raw = str(env.get("RAG_TASK_RUNTIME_SECRET_PATH") or "").strip()
    secret_storage_path = (
        Path(secret_storage_raw).expanduser()
        if secret_storage_raw
        else (state_root / "connection_secrets.protected.json")
    )

    secret_refs_raw = str(env.get("RAG_TASK_RUNTIME_SECRET_REFS_PATH") or "").strip()
    secret_refs_path = (
        Path(secret_refs_raw).expanduser()
        if secret_refs_raw
        else (state_root / "connection_secret_refs.json")
    )

    legacy_migration_enabled = _env_flag(
        env,
        "RAG_TASK_RUNTIME_ENABLE_LEGACY_MIGRATION",
        default=mode == "development",
    )
    require_managed_secret_backend = _env_flag(
        env,
        "RAG_TASK_RUNTIME_REQUIRE_MANAGED_SECRETS",
        default=mode == "production",
    )

    explicit_secret_backend = str(env.get("RAG_TASK_SECRET_BACKEND") or "").strip().casefold()
    secret_backend = explicit_secret_backend
    secret_backend_source = "env" if explicit_secret_backend else "default"
    if not secret_backend and require_managed_secret_backend:
        if str(env.get("RAG_TASK_SECRET_MASTER_KEY") or "").strip():
            secret_backend = "env_key"
            secret_backend_source = "managed-default"
        else:
            raise RuntimeError(
                "Production persistence semantics require a managed secret backend. "
                "Set RAG_TASK_SECRET_BACKEND to vault_http/vault_hashicorp/env_key or provide "
                "RAG_TASK_SECRET_MASTER_KEY for env_key."
            )

    if require_managed_secret_backend and secret_backend and secret_backend not in MANAGED_SECRET_BACKENDS:
        raise RuntimeError(
            f"Managed secret backend is required in production semantics, but got unsupported backend '{secret_backend}'."
        )

    return RuntimePersistenceProfile(
        mode=mode,
        state_root=state_root,
        state_db_path=state_db_path,
        secret_storage_path=secret_storage_path,
        secret_refs_path=secret_refs_path,
        legacy_migration_enabled=legacy_migration_enabled,
        require_managed_secret_backend=require_managed_secret_backend,
        secret_backend=secret_backend,
        secret_backend_source=secret_backend_source,
    )

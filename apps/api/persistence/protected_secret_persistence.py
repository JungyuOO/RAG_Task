from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import hashlib
import hmac
import json
import os
import platform
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import logging

from apps.api.persistence.json_persistence import load_json_file, save_json_file

logger = logging.getLogger("rag.secrets")


class SecretProtectionError(RuntimeError):
    """Raised when protected secret storage is unavailable or fails."""


class _ProtectorBackend(Protocol):
    format_name: str

    def is_available(self) -> bool: ...

    def encrypt_json(self, payload: dict) -> dict: ...

    def decrypt_json(self, payload: dict) -> dict: ...


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


@dataclass(slots=True)
class ProtectedSecretsPayload:
    format: str
    ciphertext: str


class _DpapiSecretProtector:
    format_name = "dpapi-json-v1"

    def __init__(self) -> None:
        self._enabled = platform.system().lower().startswith("windows")
        if self._enabled:
            self._crypt32 = ctypes.windll.crypt32
            self._kernel32 = ctypes.windll.kernel32
            self._crypt32.CryptProtectData.argtypes = [
                ctypes.POINTER(_DataBlob),
                wintypes.LPCWSTR,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(_DataBlob),
            ]
            self._crypt32.CryptProtectData.restype = wintypes.BOOL
            self._crypt32.CryptUnprotectData.argtypes = [
                ctypes.POINTER(_DataBlob),
                ctypes.POINTER(wintypes.LPWSTR),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(_DataBlob),
            ]
            self._crypt32.CryptUnprotectData.restype = wintypes.BOOL
            self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
            self._kernel32.LocalFree.restype = ctypes.c_void_p

    def is_available(self) -> bool:
        return self._enabled

    def encrypt_json(self, payload: dict) -> dict:
        self._ensure_available()
        plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ciphertext = self._protect(plaintext)
        protected = ProtectedSecretsPayload(
            format=self.format_name,
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
        )
        return {
            "format": protected.format,
            "ciphertext": protected.ciphertext,
        }

    def decrypt_json(self, payload: dict) -> dict:
        self._ensure_available()
        if payload.get("format") != self.format_name:
            raise SecretProtectionError(f"Unsupported protected secrets format: {payload.get('format')}")
        ciphertext = base64.b64decode(str(payload.get("ciphertext") or "").encode("ascii"))
        plaintext = self._unprotect(ciphertext)
        return json.loads(plaintext.decode("utf-8"))

    def _ensure_available(self) -> None:
        if not self._enabled:
            raise SecretProtectionError("DPAPI protected secret persistence is available only on Windows.")

    def _protect(self, data: bytes) -> bytes:
        in_blob, _in_buffer = self._to_blob(data)
        out_blob = _DataBlob()
        if not self._crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "RAG Task OCP Secret",
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        ):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            self._kernel32.LocalFree(out_blob.pbData)

    def _unprotect(self, data: bytes) -> bytes:
        in_blob, _in_buffer = self._to_blob(data)
        out_blob = _DataBlob()
        description = wintypes.LPWSTR()
        if not self._crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            ctypes.byref(description),
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        ):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            if description:
                self._kernel32.LocalFree(description)
            self._kernel32.LocalFree(out_blob.pbData)

    @staticmethod
    def _to_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
        buffer = ctypes.create_string_buffer(data)
        blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
        return blob, buffer


class _EnvKeySecretProtector:
    format_name = "envkey-json-v1"

    def __init__(self, *, env_var_name: str = "RAG_TASK_SECRET_MASTER_KEY") -> None:
        self.env_var_name = env_var_name

    def is_available(self) -> bool:
        return bool(os.environ.get(self.env_var_name))

    def encrypt_json(self, payload: dict) -> dict:
        key = self._derive_key()
        plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        nonce = secrets.token_bytes(16)
        ciphertext = self._xor_stream(plaintext, key=key, nonce=nonce)
        tag = hmac.new(key, b"tag:" + nonce + ciphertext, hashlib.sha256).digest()
        return {
            "format": self.format_name,
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "tag": base64.b64encode(tag).decode("ascii"),
            "key_source": self.env_var_name,
        }

    def decrypt_json(self, payload: dict) -> dict:
        if payload.get("format") != self.format_name:
            raise SecretProtectionError(f"Unsupported protected secrets format: {payload.get('format')}")
        key = self._derive_key()
        nonce = base64.b64decode(str(payload.get("nonce") or "").encode("ascii"))
        ciphertext = base64.b64decode(str(payload.get("ciphertext") or "").encode("ascii"))
        tag = base64.b64decode(str(payload.get("tag") or "").encode("ascii"))
        expected_tag = hmac.new(key, b"tag:" + nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            raise SecretProtectionError("Secret payload authentication failed.")
        plaintext = self._xor_stream(ciphertext, key=key, nonce=nonce)
        return json.loads(plaintext.decode("utf-8"))

    def _derive_key(self) -> bytes:
        raw = os.environ.get(self.env_var_name, "")
        if not raw:
            raise SecretProtectionError(
                f"Environment-key secret backend requires {self.env_var_name} to be set."
            )
        return hashlib.sha256(raw.encode("utf-8")).digest()

    @staticmethod
    def _xor_stream(payload: bytes, *, key: bytes, nonce: bytes) -> bytes:
        stream = bytearray()
        counter = 0
        while len(stream) < len(payload):
            counter_bytes = counter.to_bytes(8, byteorder="big", signed=False)
            stream.extend(hmac.new(key, b"stream:" + nonce + counter_bytes, hashlib.sha256).digest())
            counter += 1
        return bytes(source ^ mask for source, mask in zip(payload, stream))


class SecretProtector:
    """Configurable secret protector with DPAPI and environment-key backends.

    Preferred backend selection:
    - explicit `backend`
    - `RAG_TASK_SECRET_BACKEND`
    - default to DPAPI on Windows
    - otherwise default to env_key when `RAG_TASK_SECRET_MASTER_KEY` exists
    """

    def __init__(self, *, backend: str | None = None, env_key_var_name: str = "RAG_TASK_SECRET_MASTER_KEY") -> None:
        self._backends: dict[str, _ProtectorBackend] = {
            "dpapi": _DpapiSecretProtector(),
            "env_key": _EnvKeySecretProtector(env_var_name=env_key_var_name),
        }
        preferred = (backend or os.environ.get("RAG_TASK_SECRET_BACKEND") or "").strip().casefold()
        if not preferred:
            preferred = "dpapi" if self._backends["dpapi"].is_available() else "env_key"
        selected = self._backends.get(preferred)
        if selected is None:
            raise SecretProtectionError(f"Unsupported secret backend: {preferred}")
        self._preferred_backend_name = preferred
        self._preferred_backend = selected
        self._FORMAT = self._preferred_backend.format_name

    @property
    def format_name(self) -> str:
        return self._preferred_backend.format_name

    def ensure_available(self) -> None:
        if self._preferred_backend.is_available():
            return
        if self._preferred_backend_name == "env_key":
            raise SecretProtectionError(
                "Environment-key secret backend is selected but RAG_TASK_SECRET_MASTER_KEY is not configured."
            )
        raise SecretProtectionError("Protected secret persistence is unavailable for the selected backend.")

    def encrypt_json(self, payload: dict) -> dict:
        self.ensure_available()
        return self._preferred_backend.encrypt_json(payload)

    def decrypt_json(self, payload: dict) -> dict:
        protector = self._resolve_backend_for_payload(payload)
        if protector is None:
            raise SecretProtectionError(f"Unsupported protected secrets format: {payload.get('format')}")
        if not protector.is_available():
            raise SecretProtectionError(
                f"Secrets are stored with {payload.get('format')} but that backend is not currently available."
            )
        return protector.decrypt_json(payload)

    def needs_rewrite(self, payload: dict) -> bool:
        return payload.get("format") != self.format_name

    def _resolve_backend_for_payload(self, payload: dict) -> _ProtectorBackend | None:
        fmt = str(payload.get("format") or "")
        for backend in self._backends.values():
            if backend.format_name == fmt:
                return backend
        return None


def load_protected_json_file(path: Path, *, protector: SecretProtector, default: dict) -> tuple[dict, bool]:
    payload = load_json_file(path, default=default)
    if not payload:
        return default, False
    if isinstance(payload, dict) and payload.get("format"):
        try:
            decrypted = protector.decrypt_json(payload)
        except (SecretProtectionError, OSError, ValueError) as exc:
            mode = str(os.environ.get("RAG_TASK_RUNTIME_PERSISTENCE_MODE") or "").strip().casefold()
            allow_fallback = str(os.environ.get("RAG_TASK_ALLOW_SECRET_LOAD_FAILURE") or "").strip().casefold()
            if allow_fallback in {"1", "true", "yes", "on"} or (not allow_fallback and mode != "production"):
                logger.warning("protected secret store could not be loaded from %s; continuing with empty store: %s", path, exc)
                return default, False
            raise
        return decrypted, protector.needs_rewrite(payload)
    if isinstance(payload, dict):
        # Legacy plaintext format: return payload and ask caller to rewrite in protected form.
        return payload, True
    return default, False


def save_protected_json_file(path: Path, payload: dict, *, protector: SecretProtector) -> None:
    save_json_file(path, protector.encrypt_json(payload))


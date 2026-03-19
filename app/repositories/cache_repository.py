from __future__ import annotations

from app.rag.cache import JsonFileCache


class CacheRepository:
    def __init__(self, backend: JsonFileCache) -> None:
        self.backend = backend

    def get(self, key: str) -> dict | None:
        return self.backend.get(key)

    def set(self, key: str, value: dict) -> None:
        self.backend.set(key, value)


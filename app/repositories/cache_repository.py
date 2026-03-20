from __future__ import annotations

from app.rag.cache import JsonFileCache


class CacheRepository:
    """캐시 데이터 접근 계층 — JsonFileCache와 서비스 계층 사이의 추상 경계.

    캐시 백엔드를 파일→Redis 등으로 교체할 때 이 인터페이스만 재구현하면 된다.
    """

    def __init__(self, backend: JsonFileCache) -> None:
        self.backend = backend

    def get(self, key: str) -> dict | None:
        return self.backend.get(key)

    def set(self, key: str, value: dict) -> None:
        self.backend.set(key, value)


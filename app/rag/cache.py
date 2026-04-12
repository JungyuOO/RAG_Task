from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class JsonFileCache:
    """Simple JSON file cache with TTL and bounded entry count."""

    def __init__(
        self,
        cache_dir: Path,
        max_entries: int,
        ttl_hours: int,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            return None

        created_at = raw.get("_created_at")
        if created_at is not None and (time.time() - created_at) > self.ttl_seconds:
            self._safe_unlink(path)
            self._misses += 1
            return None

        self._hits += 1
        return {key: value for key, value in raw.items() if not key.startswith("_")}

    def set(self, key: str, value: dict) -> None:
        self._evict_if_needed()
        path = self.cache_dir / f"{key}.json"
        payload = {**value, "_created_at": time.time()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        for cache_file in self.cache_dir.glob("*.json"):
            self._safe_unlink(cache_file)
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }

    def _evict_if_needed(self) -> None:
        with self._lock:
            try:
                cache_files = sorted(
                    self.cache_dir.glob("*.json"),
                    key=lambda file_path: file_path.stat().st_mtime,
                )
            except OSError:
                return

            excess = len(cache_files) - self.max_entries + 1
            if excess <= 0:
                return
            for old_file in cache_files[:excess]:
                self._safe_unlink(old_file)

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return

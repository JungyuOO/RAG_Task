from __future__ import annotations

import json
from pathlib import Path


class JsonFileCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

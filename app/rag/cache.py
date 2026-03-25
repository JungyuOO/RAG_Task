from __future__ import annotations

import json
import time
from pathlib import Path


class JsonFileCache:
    """JSON 파일 기반 캐시 — TTL 만료와 최대 항목 수 제한을 지원한다.

    각 캐시 항목은 {key}.json 파일로 저장되며, 파일 내부에 생성 타임스탬프를
    포함하여 TTL 기반 만료 여부를 판단한다.
    """

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

    def get(self, key: str) -> dict | None:
        """캐시 항목을 조회한다. TTL이 만료된 항목은 삭제 후 None을 반환한다."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        # TTL 확인: _created_at이 없는 레거시 항목은 만료하지 않음
        created_at = raw.get("_created_at")
        if created_at is not None and (time.time() - created_at) > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return None

        # 내부 메타 필드를 제외하고 반환
        return {k: v for k, v in raw.items() if not k.startswith("_")}

    def set(self, key: str, value: dict) -> None:
        """캐시 항목을 저장한다. 최대 항목 수를 초과하면 가장 오래된 항목을 삭제한다."""
        self._evict_if_needed()
        path = self.cache_dir / f"{key}.json"
        payload = {**value, "_created_at": time.time()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _evict_if_needed(self) -> None:
        """최대 항목 수 초과 시 가장 오래된 파일부터 삭제한다."""
        try:
            cache_files = sorted(
                self.cache_dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime,
            )
        except OSError:
            return

        excess = len(cache_files) - self.max_entries + 1
        if excess <= 0:
            return
        for old_file in cache_files[:excess]:
            old_file.unlink(missing_ok=True)

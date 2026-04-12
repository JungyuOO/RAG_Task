from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

from app.rag.cache import JsonFileCache


class JsonFileCacheTests(unittest.TestCase):
    def test_evict_if_needed_ignores_unlink_errors(self) -> None:
        cache_dir = Path("tests/results/_json_cache_test")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(cache_dir, ignore_errors=True))

        cache = JsonFileCache(cache_dir, max_entries=10, ttl_hours=24)
        cache.set("a", {"value": 1})

        target = cache_dir / "a.json"
        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            cache._safe_unlink(target)

        self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.api.core.text import stable_hash


class NormalizedArtifactRepository:
    """Writes parsed/normalized/chunk artifacts to disk for the new pipeline."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Path.cwd() / "data" / "normalized")

    def save_parsed(self, source_path: str, payload: dict[str, Any]) -> str:
        return self._write_json("documents", source_path, payload)

    def save_normalized(self, source_path: str, payload: list[dict[str, Any]]) -> str:
        return self._write_json("blocks", source_path, payload)

    def save_chunks(self, source_path: str, payload: list[dict[str, Any]]) -> str:
        return self._write_json("chunks", source_path, payload)

    def _write_json(self, category: str, source_path: str, payload: Any) -> str:
        file_path = self.root / category / f"{stable_hash(source_path)}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(file_path)



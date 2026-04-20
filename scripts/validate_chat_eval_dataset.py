from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests" / "data" / "chat_eval_dataset_v1.json"
API_BASE = "http://localhost:8000"


def api_get(path: str) -> Any:
    response = httpx.get(f"{API_BASE}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def api_get_retry(path: str, *, attempts: int = 5, delay_sec: float = 2.0) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return api_get(path)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(delay_sec * attempt)
    assert last_error is not None
    raise last_error


def main() -> int:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    connection_id = str(dataset["connection_id"])
    namespace = str(dataset["live_selection"]["namespace"])
    catalog = api_get("/api/v1/library/catalog")

    doc_paths = [str(item.get("relative_path") or "").casefold() for item in catalog.get("documents", [])]
    problems: list[str] = []
    warnings: list[str] = []
    checked_live_targets: set[tuple[str, str]] = set()

    for scenario in dataset["scenarios"]:
      for step in scenario["turns"]:
        expect = step.get("expect") or {}
        for token in expect.get("source_path_tokens") or []:
          token_text = str(token).casefold()
          if not any(token_text in path for path in doc_paths):
            problems.append(f"missing_doc_token:{scenario['id']}:{token}")

        if step["step_type"] == "yaml_apply":
          resource = step["target_resource"]
          name = step["target_name"]
          target_key = (str(resource), str(name))
          if target_key in checked_live_targets:
            continue
          checked_live_targets.add(target_key)
          try:
            payload = api_get_retry(
              f"/api/v1/ocp/resources/{connection_id}?resource={resource}&namespace={namespace}"
            )
            items = payload.get("items") or []
            if not any(str(item.get("name") or "") == str(name) for item in items):
              problems.append(f"missing_live_target:{scenario['id']}:{name}:not_listed")
          except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "502 Bad Gateway" in message:
              warnings.append(f"live_target_backend_warning:{scenario['id']}:{name}:{message}")
            else:
              problems.append(f"missing_live_target:{scenario['id']}:{name}:{message}")

    print(json.dumps({
      "dataset": str(DATASET_PATH),
      "doc_count": len(doc_paths),
      "problem_count": len(problems),
      "warning_count": len(warnings),
      "problems": problems,
      "warnings": warnings,
    }, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

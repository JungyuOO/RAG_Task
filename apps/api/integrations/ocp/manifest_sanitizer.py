from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

_SERVER_MANAGED_METADATA_FIELDS = {
    "creationTimestamp",
    "deletionGracePeriodSeconds",
    "deletionTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
}


def sanitize_manifest_for_apply(manifest: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(manifest)
    sanitized.pop("status", None)

    metadata = sanitized.get("metadata")
    if isinstance(metadata, dict):
        for field in _SERVER_MANAGED_METADATA_FIELDS:
            metadata.pop(field, None)

    _normalize_string_maps_and_env(sanitized)

    return sanitized


def _normalize_string_maps_and_env(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key in {"annotations", "labels"} and isinstance(item, dict):
                for nested_key, nested_value in list(item.items()):
                    if nested_value is None:
                        item[nested_key] = ""
                    elif not isinstance(nested_value, str):
                        item[nested_key] = str(nested_value)
            elif key == "env" and isinstance(item, list):
                for env_item in item:
                    if not isinstance(env_item, dict):
                        continue
                    if "value" in env_item and "valueFrom" not in env_item:
                        env_value = env_item.get("value")
                        if env_value is None:
                            env_item["value"] = ""
                        elif not isinstance(env_value, str):
                            env_item["value"] = str(env_value)
                    _normalize_string_maps_and_env(env_item)
            else:
                _normalize_string_maps_and_env(item)
    elif isinstance(value, list):
        for item in value:
            _normalize_string_maps_and_env(item)


def dump_manifest_yaml(manifest: dict[str, Any]) -> str:
    return yaml.safe_dump(
        manifest,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

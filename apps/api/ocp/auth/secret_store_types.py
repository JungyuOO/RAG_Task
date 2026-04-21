from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.schemas.auth import OcpAuthMode


@dataclass(slots=True)
class StoredConnectionSecret:
    auth_mode: OcpAuthMode
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)



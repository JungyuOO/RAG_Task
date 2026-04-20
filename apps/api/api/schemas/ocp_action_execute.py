from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OcpActionExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    actor_id: str = "ui"
    actor_roles: list[str] = Field(default_factory=list)
    execution_note: str = ""
    force: bool = False


from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DocumentPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    relative_source_path: str = ""
    repo_relative_path: str = ""
    repo_locator: str = ""
    workspace_root_name: str = ""
    file_name: str = ""
    chunk_id: str = ""
    source_type: str
    title: str = ""
    section_title: str = ""
    section_path: list[str] = Field(default_factory=list)
    anchor: str = ""
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_locator: str = ""
    file_uri: str = ""
    shell_open_command: str = ""
    shell_open_label: str = ""
    vscode_uri: str = ""
    vscode_uri_with_line: str = ""
    code_command: str = ""
    snippet: str
    lines: list[str] = Field(default_factory=list)


from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query

from apps.api.schemas.chat import DocumentPreviewResponse

router = APIRouter(prefix="/docs-preview", tags=["docs-preview"])


@router.get("/snippet", response_model=DocumentPreviewResponse)
async def preview_document_snippet(
    source_path: str = Query(..., min_length=1),
    chunk_id: str = Query("", min_length=0),
) -> DocumentPreviewResponse:
    path = Path(source_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Source file not found.")

    text = path.read_text(encoding="utf-8", errors="ignore")
    source_type = path.suffix.lower().lstrip(".") or "text"
    title = _extract_title(path.name, text)
    chunk_preview = _lookup_chunk_preview(path, chunk_id)
    if chunk_preview is not None:
        section_title = str(chunk_preview.get("section_title") or "")
        section_path = list(chunk_preview.get("section_path") or _extract_section_path(title, section_title))
        anchor = str(chunk_preview.get("anchor") or "")
        page_number = chunk_preview.get("page_number")
        snippet_lines = list(chunk_preview.get("snippet_lines") or [])
        line_start = chunk_preview.get("line_start")
        line_end = chunk_preview.get("line_end")
    else:
        section_title = _extract_section_title(text)
        section_path = _extract_section_path(title, section_title)
        anchor = _extract_anchor(text)
        page_number = _extract_page_number(text)
        snippet_lines, line_start, line_end = _extract_relevant_lines(text)
    snippet = "\n".join(snippet_lines)[:4000]
    return DocumentPreviewResponse(
        source_path=str(path),
        relative_source_path=_relative_source_path(path),
        repo_relative_path=_repo_relative_path(path),
        repo_locator=_build_repo_locator(path, line_start=line_start),
        workspace_root_name=Path.cwd().resolve().name,
        file_name=path.name,
        chunk_id=chunk_id,
        source_type=source_type,
        title=title,
        section_title=section_title,
        section_path=section_path,
        anchor=anchor,
        page_number=page_number,
        line_start=line_start,
        line_end=line_end,
        source_locator=_build_source_locator(anchor=anchor, page_number=page_number, line_start=line_start, line_end=line_end),
        file_uri=_build_file_uri(path),
        shell_open_command=_build_shell_open_command(path),
        shell_open_label=_build_shell_open_label(),
        vscode_uri=_build_vscode_uri(path),
        vscode_uri_with_line=_build_vscode_uri(path, line=line_start or 1),
        code_command=_build_code_command(path, line=line_start or 1),
        snippet=snippet,
        lines=snippet_lines[:12],
    )


def _extract_title(file_name: str, text: str) -> str:
    html_title = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if html_title:
        return re.sub(r"\s+", " ", html_title.group(1)).strip()
    md_title = re.search(r"^\s*#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if md_title:
        return md_title.group(1).strip()
    return file_name


def _extract_section_title(text: str) -> str:
    md_heading = re.search(r"^\s*##\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if md_heading:
        return md_heading.group(1).strip()
    html_heading = re.search(r"<h[1-6][^>]*>(.*?)</h[1-6]>", text, flags=re.IGNORECASE | re.DOTALL)
    if html_heading:
        return re.sub(r"\s+", " ", html_heading.group(1)).strip()
    return ""


def _extract_section_path(title: str, section_title: str) -> list[str]:
    path = []
    if title:
        path.append(title)
    if section_title and section_title != title:
        path.append(section_title)
    return path


def _extract_anchor(text: str) -> str:
    html_id = re.search(r"<h[1-6][^>]*id=[\"']([^\"']+)[\"']", text, flags=re.IGNORECASE)
    if html_id:
        return html_id.group(1).strip()
    return ""


def _extract_page_number(text: str) -> int | None:
    page_match = re.search(r"(?im)^##\s*Page\s+(\d+)\s*$", text)
    if page_match:
        try:
            return int(page_match.group(1))
        except ValueError:
            return None
    return None


def _extract_relevant_lines(text: str) -> tuple[list[str], int | None, int | None]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    selected: list[str] = []
    line_numbers: list[int] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        selected.append(line)
        line_numbers.append(index)
        if len(selected) >= 20:
            break
    if not selected:
        return [], None, None
    return selected, line_numbers[0], line_numbers[-1]


def _lookup_chunk_preview(path: Path, chunk_id: str) -> dict | None:
    if not chunk_id.strip():
        return None
    try:
        from apps.api.runtime import pgvector_bridge

        runtime = pgvector_bridge._get_runtime()
        chunks = runtime.index_repository.list_chunks()
    except Exception:
        return None

    target_path = str(path.resolve()).replace("\\", "/")
    for item in chunks:
        if str(item.get("chunk_id") or "") != chunk_id:
            continue
        source_path = str(item.get("source_path") or "").replace("\\", "/")
        if source_path != target_path and not source_path.endswith(path.name):
            continue
        metadata = dict(item.get("metadata") or {})
        snippet_text = str(item.get("text") or "").strip()
        snippet_lines = [line.rstrip() for line in snippet_text.splitlines() if line.strip()]
        line_start, line_end = _locate_snippet_lines(path, snippet_text)
        return {
            "section_title": metadata.get("section_title") or "",
            "section_path": metadata.get("section_path") or [],
            "anchor": metadata.get("html_anchor") or "",
            "page_number": item.get("page_number"),
            "snippet_lines": snippet_lines[:20],
            "line_start": line_start,
            "line_end": line_end,
        }
    return None


def _locate_snippet_lines(path: Path, snippet_text: str) -> tuple[int | None, int | None]:
    if not snippet_text.strip():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return None, None
    needle = snippet_text.strip().replace("\r\n", "\n").replace("\r", "\n")
    exact_index = text.find(needle)
    if exact_index != -1:
        line_start = text.count("\n", 0, exact_index) + 1
        line_end = line_start + max(len(needle.splitlines()) - 1, 0)
        return line_start, line_end
    first_line = next((line.strip() for line in needle.splitlines() if line.strip()), "")
    if not first_line:
        return None, None
    lines = text.split("\n")
    for index, line in enumerate(lines, start=1):
        if first_line in line:
            return index, min(index + max(len(needle.splitlines()) - 1, 0), len(lines))
    return None, None


def _build_source_locator(*, anchor: str, page_number: int | None, line_start: int | None, line_end: int | None) -> str:
    if anchor:
        return f"#{anchor}"
    if page_number is not None:
        return f"page-{page_number}"
    if line_start is not None and line_end is not None:
        return f"lines {line_start}-{line_end}"
    return ""


def _relative_source_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _repo_relative_path(path: Path) -> str:
    return _relative_source_path(path).replace("\\", "/")


def _build_file_uri(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except ValueError:
        return ""


def _build_shell_open_label() -> str:
    if sys.platform.startswith("win"):
        return "explorer"
    if sys.platform == "darwin":
        return "open"
    return "xdg-open"


def _build_shell_open_command(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win"):
        return f'Start-Process "{resolved}"'
    if sys.platform == "darwin":
        return f'open "{resolved}"'
    return f'xdg-open "{resolved}"'


def _build_vscode_uri(path: Path, *, line: int | None = None) -> str:
    resolved = path.resolve()
    encoded = quote(resolved.as_posix(), safe="/:")
    if line is not None and line > 0:
        return f"vscode://file/{encoded}:{line}:1"
    return f"vscode://file/{encoded}"


def _build_repo_locator(path: Path, *, line_start: int | None) -> str:
    repo_relative = _repo_relative_path(path)
    if line_start is not None and line_start > 0:
        return f"{repo_relative}:{line_start}"
    return repo_relative


def _build_code_command(path: Path, *, line: int) -> str:
    resolved = str(path.resolve())
    safe_line = line if line > 0 else 1
    return f'code -g "{resolved}:{safe_line}:1"'


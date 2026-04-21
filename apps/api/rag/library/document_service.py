from __future__ import annotations

from pathlib import Path

from apps.api.schemas.library import (
    LibraryCatalogResponse,
    LibraryChunkRecord,
    LibraryDocumentChunksResponse,
    LibraryDocumentContentResponse,
    LibraryDocumentRecord,
)
from apps.api.rag.retrieval.legacy_pgvector_runtime import LegacyPgvectorRuntime


class LibraryDocumentService:
    def __init__(self, *, runtime: LegacyPgvectorRuntime | None = None) -> None:
        self._runtime = runtime or LegacyPgvectorRuntime()

    def get_catalog(self, *, workspace_id: str = "") -> LibraryCatalogResponse:
        runtime = self._runtime.get()
        source_root = Path(runtime.settings.rag_source_dir).resolve()
        files = [
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", ".gitkeep"} and not path.name.startswith(".")
        ]

        chunk_count_map = {
            self._document_key_from_path(str(item.get("source_path") or ""), source_root): int(item.get("chunk_count") or 0)
            for item in runtime.index_repository.get_document_chunk_counts()
        }

        documents: list[LibraryDocumentRecord] = []
        for path in sorted(files, key=lambda item: str(item).casefold()):
            if self._is_legacy_file(path, source_root):
                continue
            if self._should_skip_catalog_file(path, source_root):
                continue
            document = self._build_document_record(path, source_root, chunk_count_map, workspace_id=workspace_id)
            documents.append(document)

        message = (
            "Official docs render as markdown and each document exposes indexed status and chunk counts."
        )
        return LibraryCatalogResponse(
            workspace_id=workspace_id,
            documents=documents,
            message=message,
        )

    def get_chunks(self, document_key: str, *, workspace_id: str = "") -> LibraryDocumentChunksResponse:
        runtime = self._runtime.get()
        source_root = Path(runtime.settings.rag_source_dir).resolve()
        path = self._resolve_document_key(document_key, source_root)
        title = self._extract_title(path)

        chunks = []
        for item in runtime.index_repository.list_chunks():
            key = self._document_key_from_path(str(item.get("source_path") or ""), source_root)
            if key != document_key:
                continue
            metadata = dict(item.get("metadata") or {})
            block_types = metadata.get("block_types") or []
            if isinstance(block_types, str):
                block_types = [block_types] if block_types else []
            chunks.append(
                LibraryChunkRecord(
                    chunk_id=str(item.get("chunk_id") or ""),
                    chunk_order=item.get("chunk_order"),
                    page_number=item.get("page_number"),
                    section_title=str(metadata.get("section_title") or ""),
                    block_types=[str(block_type) for block_type in block_types],
                    preview_text=str(item.get("text") or "")[:600],
                )
            )

        return LibraryDocumentChunksResponse(
            workspace_id=workspace_id,
            document_key=document_key,
            title=title,
            chunk_count=len(chunks),
            chunks=chunks[:120],
        )

    def get_markdown_content(self, document_key: str, *, workspace_id: str = "") -> LibraryDocumentContentResponse:
        runtime = self._runtime.get()
        source_root = Path(runtime.settings.rag_source_dir).resolve()
        path = self._resolve_document_key(document_key, source_root)
        if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            raise ValueError("Markdown content view is only available for markdown/text documents.")
        content = path.read_text(encoding="utf-8", errors="ignore")
        return LibraryDocumentContentResponse(
            workspace_id=workspace_id,
            document_key=document_key,
            title=self._extract_title(path),
            content=content,
        )

    def get_file_path(self, document_key: str, *, workspace_id: str = "") -> Path:
        runtime = self._runtime.get()
        source_root = Path(runtime.settings.rag_source_dir).resolve()
        return self._resolve_document_key(document_key, source_root)

    def _build_document_record(
        self,
        path: Path,
        source_root: Path,
        chunk_count_map: dict[str, int],
        *,
        workspace_id: str,
    ) -> LibraryDocumentRecord:
        document_key = path.relative_to(source_root).as_posix()
        group = self._classify_group(document_key)
        original_kind, original_key = self._resolve_original(path, source_root, group)
        chunk_count = int(chunk_count_map.get(document_key, 0))
        return LibraryDocumentRecord(
            workspace_id=workspace_id,
            document_key=document_key,
            title=self._extract_title(path),
            relative_path=document_key,
            source_type=path.suffix.lower().lstrip(".") or "text",
            group=group,
            indexed=chunk_count > 0,
            chunk_count=chunk_count,
            original_kind=original_kind,
            original_key=original_key,
            description=self._describe_document(document_key, group),
        )

    @staticmethod
    def _classify_group(document_key: str) -> str:
        lowered = document_key.casefold()
        return "official"

    @staticmethod
    def _describe_document(document_key: str, group: str) -> str:
        if "ocp-html-single" in document_key.casefold():
            return "OpenShift official documentation normalized into markdown."
        return "Official product documentation."

    def _resolve_original(self, path: Path, source_root: Path, group: str) -> tuple[str, str]:
        document_key = path.relative_to(source_root).as_posix()
        return "markdown", document_key

    @staticmethod
    def _should_skip_catalog_file(path: Path, source_root: Path) -> bool:
        relative_path = path.relative_to(source_root).as_posix()
        if not (relative_path.startswith("generated_pdf/") or relative_path.startswith("customer_pdf/")):
            return False
        sibling_candidates = [
            source_root / "customer" / f"{path.stem}.md",
            source_root / "generated" / f"{path.stem}.md",
        ]
        return any(candidate.exists() for candidate in sibling_candidates)

    @staticmethod
    def _is_legacy_file(path: Path, source_root: Path) -> bool:
        return path.relative_to(source_root).as_posix().startswith("legacy/")

    @staticmethod
    def _extract_title(path: Path) -> str:
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            try:
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("# "):
                        return stripped[2:].strip()
                    if stripped:
                        break
            except Exception:
                pass
        return path.stem.replace("_", " ").replace("-", " ").strip() or path.name

    @staticmethod
    def _document_key_from_path(value: str, source_root: Path) -> str:
        if not value:
            return ""
        normalized = value.replace("\\", "/")
        root = source_root.as_posix()
        if normalized.startswith(root):
            return normalized[len(root) + 1 :]
        for anchor in ("pdfs/", "corpus/pdfs/"):
            if anchor in normalized:
                return normalized.split(anchor, 1)[-1]
        return ""

    @staticmethod
    def _resolve_document_key(document_key: str, source_root: Path) -> Path:
        normalized = document_key.replace("\\", "/").lstrip("/")
        path = (source_root / normalized).resolve()
        if source_root not in path.parents and path != source_root:
            raise ValueError("Requested document is outside the allowed source root.")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError("Requested document was not found.")
        return path


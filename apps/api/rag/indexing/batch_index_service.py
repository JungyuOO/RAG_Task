from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path

from apps.api.api.schemas.indexing import (
    BatchIndexItem,
    BatchIndexRequest,
    BatchIndexResponse,
    DocumentGroup,
    NewIndexRequest,
    SourceType,
)
from apps.api.rag.indexing.index_service import NewIndexingService

logger = logging.getLogger("rag.batch_index")


class BatchIndexingService:
    def __init__(self, *, indexing_service: NewIndexingService | None = None) -> None:
        self.indexing_service = indexing_service or NewIndexingService()

    def run(self, request: BatchIndexRequest, *, progress_callback=None, log_callback=None, status_callback=None, should_cancel=None) -> BatchIndexResponse:
        sources = self._discover_sources(request)
        items: list[BatchIndexItem] = []
        indexed_files = 0
        failed_files = 0

        self._emit_log(log_callback, f"discovered {len(sources)} source files for batch reindex")
        self._emit_status(status_callback, step="RUNNING", message=f"{len(sources)} files discovered")

        for index, source in enumerate(sources, start=1):
            if should_cancel and should_cancel():
                self._emit_log(log_callback, "batch reindex cancelled")
                return BatchIndexResponse(
                    discovered_files=len(sources),
                    processed_files=len(items),
                    indexed_files=indexed_files,
                    failed_files=failed_files,
                    progress_pct=int(((index - 1) / max(len(sources), 1)) * 100),
                    current_file="",
                    items=items,
                )
            if progress_callback:
                progress_callback(
                    progress_pct=int(((index - 1) / max(len(sources), 1)) * 100),
                    current_file=source["source_path"],
                )
            self._emit_log(log_callback, f"[{index}/{len(sources)}] start {source['source_path']}")
            self._emit_status(status_callback, step="RUNNING", message=f"[{index}/{len(sources)}] started", current_file=source["source_path"])
            started = time.perf_counter()
            try:
                index_request = NewIndexRequest(
                    source_type=source["source_type"],
                    source_path=source["source_path"],
                    file_name=Path(source["source_path"]).name,
                    locale=source["locale"],
                    document_group=source["document_group"],
                )
                kwargs = {}
                signature = inspect.signature(self.indexing_service.index_source)
                if "stage_callback" in signature.parameters:
                    kwargs["stage_callback"] = log_callback
                if "status_callback" in signature.parameters:
                    kwargs["status_callback"] = status_callback
                if "current_file" in signature.parameters:
                    kwargs["current_file"] = source["source_path"]
                result = self.indexing_service.index_source(index_request, **kwargs)
                indexed_files += 1 if result.indexed else 0
                elapsed = time.perf_counter() - started
                self._emit_log(log_callback, f"[{index}/{len(sources)}] indexed chunks={result.chunks} source={result.source_path} elapsed={elapsed:.2f}s")
                items.append(
                    BatchIndexItem(
                        source_path=result.source_path,
                        source_type=result.source_type,
                        indexed=result.indexed,
                        chunks=result.chunks,
                    )
                )
            except Exception as exc:
                failed_files += 1
                elapsed = time.perf_counter() - started
                logger.exception("[batch-index] failed source=%s elapsed=%.2fs", source["source_path"], elapsed)
                self._emit_log(log_callback, f"[{index}/{len(sources)}] failed source={source['source_path']} error={exc} elapsed={elapsed:.2f}s")
                items.append(
                    BatchIndexItem(
                        source_path=source["source_path"],
                        source_type=source["source_type"],
                        indexed=False,
                        chunks=0,
                        error=str(exc),
                    )
                )

        self._emit_log(log_callback, f"batch reindex complete indexed={indexed_files} failed={failed_files}")
        self._emit_status(status_callback, step="COMPLETED", message=f"indexed={indexed_files} failed={failed_files}")
        return BatchIndexResponse(
            discovered_files=len(sources),
            processed_files=len(items),
            indexed_files=indexed_files,
            failed_files=failed_files,
            progress_pct=100 if sources else 0,
            current_file="",
            items=items,
        )

    @staticmethod
    def _emit_log(log_callback, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        logger.info(text)
        if log_callback:
            log_callback(text)

    @staticmethod
    def _emit_status(status_callback, *, step: str, message: str, current_file: str | None = None) -> None:
        if status_callback:
            status_callback(step=step, message=message, current_file=current_file)

    def _discover_sources(self, request: BatchIndexRequest) -> list[dict]:
        if request.explicit_source_paths:
            return self._build_explicit_sources(request)

        root = Path(request.root_path) if request.root_path else Path.cwd() / "data"
        patterns = self._patterns_for_source_type(request.source_type)
        matches: list[dict] = []

        for pattern, source_type in patterns:
            iterator = root.rglob(pattern) if request.include_subdirectories else root.glob(pattern)
            for path in sorted(iterator):
                if not path.is_file():
                    continue
                document_group = self._infer_document_group(path)
                if request.document_group and document_group != request.document_group:
                    continue
                locale = self._extract_locale(path.name)
                if request.locale and locale != request.locale:
                    continue
                matches.append(
                    {
                        "source_path": str(path),
                        "source_type": source_type,
                        "document_group": document_group,
                        "locale": locale,
                    }
                )

        if request.max_files > 0:
            matches = matches[: request.max_files]
        return matches

    def _build_explicit_sources(self, request: BatchIndexRequest) -> list[dict]:
        matches: list[dict] = []
        for raw_path in request.explicit_source_paths:
            path = Path(raw_path)
            if not path.is_file():
                continue
            source_type = request.source_type or self._infer_source_type_from_path(path)
            if source_type is None:
                continue
            document_group = request.document_group or self._infer_document_group(path)
            locale = request.locale or self._extract_locale(path.name)
            matches.append(
                {
                    "source_path": str(path),
                    "source_type": source_type,
                    "document_group": document_group,
                    "locale": locale,
                }
            )
        if request.max_files > 0:
            matches = matches[: request.max_files]
        return matches

    @staticmethod
    def _patterns_for_source_type(source_type: SourceType | None) -> list[tuple[str, SourceType]]:
        if source_type == SourceType.HTML_SINGLE:
            return [("*.html", SourceType.HTML_SINGLE)]
        if source_type == SourceType.HTML:
            return [("*.html", SourceType.HTML)]
        if source_type == SourceType.GENERATED_MANUAL:
            return [("*.md", SourceType.GENERATED_MANUAL)]
        if source_type == SourceType.PDF:
            return [("*.pdf", SourceType.PDF)]
        if source_type == SourceType.STRUCTURED_JSON:
            return [("*.json", SourceType.STRUCTURED_JSON)]
        return [
            ("*.html", SourceType.HTML),
            ("*.md", SourceType.GENERATED_MANUAL),
        ]

    @staticmethod
    def _infer_source_type_from_path(path: Path) -> SourceType | None:
        suffix = path.suffix.casefold()
        if suffix == ".html":
            normalized = str(path).replace("\\", "/").casefold()
            if "html-single" in normalized:
                return SourceType.HTML_SINGLE
            return SourceType.HTML
        if suffix == ".md":
            return SourceType.GENERATED_MANUAL
        if suffix == ".pdf":
            return SourceType.PDF
        if suffix == ".json":
            return SourceType.STRUCTURED_JSON
        return None

    @staticmethod
    def _infer_document_group(path: Path) -> DocumentGroup:
        normalized = str(path).replace("\\", "/").casefold()
        if any(marker in normalized for marker in ("customer-guide", "/generated", "/customer/", "/customer_pdf/")):
            return DocumentGroup.CUSTOMER_GENERATED
        return DocumentGroup.OFFICIAL_OCP

    @staticmethod
    def _extract_locale(name: str) -> str:
        lowered = name.casefold()
        if "-ko" in lowered or "_ko" in lowered:
            return "ko"
        if "-en" in lowered or "_en" in lowered:
            return "en"
        return ""

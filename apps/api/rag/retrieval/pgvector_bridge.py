from __future__ import annotations

from pathlib import Path

from apps.api.api.schemas.chat import CopilotChatResponse, CopilotChatSourceItem
from apps.api.rag.query.query_features import answer_source_budget
from apps.api.rag.retrieval.legacy_pgvector_runtime import LegacyPgvectorRuntime


class PgvectorRetrievalBridge:
    """Thin adapter over the currently indexed pgvector-backed legacy corpus."""

    DENSE_SEARCH_LIMIT = 20
    MAX_COSINE_DISTANCE = 0.85

    def __init__(self, *, runtime: LegacyPgvectorRuntime | None = None) -> None:
        self._runtime = runtime or LegacyPgvectorRuntime()

    def _get_runtime(self):
        return self._runtime.get()

    async def answer(
        self,
        *,
        message: str,
        allowed_source_paths: list[str] | None = None,
        additional_query: str | None = None,
    ) -> CopilotChatResponse | None:
        normalized_allowed = self._normalize_allowed_source_paths(allowed_source_paths)
        try:
            runtime = self._get_runtime()
            use_asym = bool(getattr(self._runtime, "_asymmetric_enabled", True))
            query_vector = runtime.embedder.encode(message, is_query=use_asym)
            candidates = runtime.index_repository.search_dense_candidates(
                query_vector,
                limit=self.DENSE_SEARCH_LIMIT,
                source_paths=allowed_source_paths,
            )
        except Exception:
            return None

        if additional_query and additional_query.strip() != message.strip():
            try:
                alt_vector = runtime.embedder.encode(additional_query, is_query=use_asym)
                alt_candidates = runtime.index_repository.search_dense_candidates(
                    alt_vector,
                    limit=self.DENSE_SEARCH_LIMIT,
                    source_paths=allowed_source_paths,
                )
                candidates = self._merge_candidates(candidates, alt_candidates)
            except Exception:
                pass

        candidates = [c for c in candidates if float(c.get("distance", 1.0)) <= self.MAX_COSINE_DISTANCE]
        if normalized_allowed:
            candidates = [
                candidate
                for candidate in candidates
                if self._source_path_allowed(str((candidate.get("chunk") or {}).get("source_path") or ""), normalized_allowed)
            ]
        if not candidates:
            return None

        top = candidates[: self._answer_source_limit(message)]
        answer_lines: list[str] = []
        sources: list[CopilotChatSourceItem] = []
        for candidate in top:
            chunk = candidate.get("chunk", {})
            metadata = chunk.get("metadata", {}) or {}
            source_path = str(chunk.get("source_path") or "")
            section_title = str(metadata.get("section_title") or Path(source_path).name or "document")
            display_text = str(chunk.get("text") or "")[:220]
            answer_lines.append(f"{section_title}\n{display_text}")
            sources.append(
                CopilotChatSourceItem(
                    source_type="doc",
                    label=f"{Path(source_path).name} · {section_title}" if source_path else section_title,
                    source_path=source_path,
                    relative_source_path=self._relative_source_path(source_path),
                    page_number=chunk.get("page_number"),
                    chunk_id=str(chunk.get("chunk_id") or ""),
                    score=round(
                        max(0.0, 1.0 - float(candidate.get("distance") or 1.0))
                        - (0.08 if "/legacy/" in source_path.replace("\\", "/").casefold() else 0.0),
                        3,
                    ),
                    provenance=["doc_pgvector"],
                    metadata={
                        "section_title": section_title,
                        "section_path": metadata.get("section_path", []),
                        "html_anchor": metadata.get("html_anchor", ""),
                        "source_block_type": metadata.get("block_types", ""),
                        "retrieval_backend": "pgvector_bridge",
                        "relative_source_path": self._relative_source_path(source_path),
                        "file_name": Path(source_path).name if source_path else "",
                        "preview_text": display_text,
                        "synthesis_text": str(chunk.get("text") or "")[:500],
                    },
                )
            )

        return CopilotChatResponse(
            lane="doc_pgvector",
            mode="pgvector_dense",
            fallback_used=False,
            preview_ready=True,
            answer="\n\n".join(answer_lines).strip(),
            sources=sources,
        )

    @staticmethod
    def _normalize_allowed_source_paths(allowed_source_paths: list[str] | None) -> set[str]:
        return {
            str(path or "").replace("\\", "/").casefold()
            for path in (allowed_source_paths or [])
            if str(path or "").strip()
        }

    @staticmethod
    def _source_path_allowed(source_path: str, normalized_allowed: set[str]) -> bool:
        normalized_source = str(source_path or "").replace("\\", "/").casefold()
        if normalized_source in normalized_allowed:
            return True
        file_name = Path(source_path).name.casefold()
        return any(Path(path).name.casefold() == file_name for path in normalized_allowed)

    @staticmethod
    def _merge_candidates(primary: list[dict], secondary: list[dict]) -> list[dict]:
        seen: set[str] = set()
        merged: list[dict] = []
        for candidate in primary:
            chunk_id = candidate.get("chunk", {}).get("chunk_id", "")
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append(candidate)
        for candidate in secondary:
            chunk_id = candidate.get("chunk", {}).get("chunk_id", "")
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append(candidate)
        merged.sort(key=lambda c: float(c.get("distance", 1.0)))
        return merged

    @staticmethod
    def _relative_source_path(source_path: str) -> str:
        if not source_path:
            return ""
        path = Path(source_path)
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _answer_source_limit(message: str) -> int:
        return answer_source_budget(message)

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


class AnswerCitationMixin:
    def build_answer_citation_payload(
        self,
        answer: str,
        context_items: list[dict],
        grounded_pages: list[dict],
        preferred_preview_source: str | None,
    ) -> list[dict]:
        if self.should_suppress_citations(answer):
            return []

        payload: list[dict] = []
        seen: set[tuple[str, int]] = set()
        items_by_name: dict[str, list[dict]] = defaultdict(list)
        for item in context_items:
            source_path = item["chunk"]["source_path"]
            items_by_name[Path(source_path).name].append(item)

        for file_name, cited_start, cited_end in self.extract_answer_citations(answer):
            for item in items_by_name.get(file_name, []):
                chunk = item["chunk"]
                source_path = chunk["source_path"]
                page_start = int(chunk["metadata"].get("page_start") or chunk["page_number"] or 1)
                page_end = int(chunk["metadata"].get("page_end") or page_start)
                if page_end < page_start:
                    page_end = page_start
                overlap_start = max(page_start, cited_start)
                overlap_end = min(page_end, cited_end)
                for page_number in range(overlap_start, overlap_end + 1):
                    self.append_citation_entry(
                        payload,
                        seen,
                        source_path=source_path,
                        page_number=page_number,
                        score=float(item["rerank_score"]),
                        chunk_id=chunk["chunk_id"],
                        origin="answer_text",
                    )

        if payload:
            return payload

        fallback_pages = [item for item in grounded_pages if item["source_path"] == preferred_preview_source] or grounded_pages
        for item in fallback_pages[:3]:
            self.append_citation_entry(
                payload,
                seen,
                source_path=item["source_path"],
                page_number=int(item["page_number"]),
                score=float(item["score"]),
                chunk_id=None,
                origin="grounded_page",
            )
        return payload

    def should_suppress_citations(self, answer: str) -> bool:
        normalized = (answer or "").strip().casefold()
        if not normalized:
            return True
        negative_markers = (
            "업로드된 문서에서 관련 내용을 찾을 수 없습니다",
            "관련 내용을 찾기 어렵습니다",
            "제공된 문서에는",
            "포함되어 있지 않습니다",
            "명시적인 목록은 없습니다",
            "다른 질문을 해주시거나",
            "관련 문서를 업로드해 주세요",
            "찾을 수 없습니다",
            "unable to find relevant content",
        )
        return any(marker.casefold() in normalized for marker in negative_markers)

    def append_citation_entry(
        self,
        payload: list[dict],
        seen: set[tuple[str, int]],
        source_path: str,
        page_number: int,
        score: float,
        chunk_id: str | None,
        origin: str,
    ) -> None:
        key = (source_path, page_number)
        if key in seen:
            return
        seen.add(key)
        payload.append(
            {
                "source_path": source_path,
                "file_name": Path(source_path).name,
                "page_number": page_number,
                "score": round(score, 4),
                "chunk_id": chunk_id,
                "origin": origin,
            }
        )

    def extract_answer_citations(self, answer: str) -> list[tuple[str, int, int]]:
        if not answer:
            return []

        citations: list[tuple[str, int, int]] = []
        grouped_pattern = re.compile(
            r"\[([^\[\]\n]+?\.pdf)\]\s*((?:p\.\d+(?:-\d+)?)(?:\s*,\s*p\.\d+(?:-\d+)?)*)",
            flags=re.IGNORECASE,
        )
        for match in grouped_pattern.finditer(answer):
            file_name = Path(match.group(1).strip()).name
            page_tokens = re.findall(r"p\.(\d+)(?:-(\d+))?", match.group(2), flags=re.IGNORECASE)
            for page_start_raw, page_end_raw in page_tokens:
                page_start = int(page_start_raw)
                page_end = int(page_end_raw or page_start)
                if page_end < page_start:
                    page_end = page_start
                citations.append((file_name, page_start, page_end))

        patterns = (
            r"\[([^\[\]\n]+?\.pdf)\s+p\.(\d+)(?:-(\d+))?\]",
            r"\[([^\[\]\n]+?\.pdf)\]\s*p\.(\d+)(?:-(\d+))?",
            r"\[([^\[\]\n]+?)\]\s*p\.(\d+)(?:-(\d+))?",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
                file_name = Path(match.group(1).strip()).name
                page_start = int(match.group(2))
                page_end = int(match.group(3) or page_start)
                if page_end < page_start:
                    page_end = page_start
                citations.append((file_name, page_start, page_end))
        deduped: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int, int]] = set()
        for citation in citations:
            if citation in seen:
                continue
            seen.add(citation)
            deduped.append(citation)
        return deduped

    def build_answer_aligned_preview_pages(
        self,
        answer_citations: list[dict],
        context_items: list[dict],
        preferred_preview_source: str | None,
        grounded_pages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        fallback_pages = self.retrieval_service.build_grounded_preview_pages(preferred_preview_source, grounded_pages)
        if not context_items or not answer_citations:
            return preferred_preview_source, fallback_pages

        preview_pages: list[dict] = []
        seen_pages: set[tuple[str, int]] = set()
        chosen_source: str | None = None
        for citation in answer_citations:
            source_path = citation["source_path"]
            if chosen_source is None:
                chosen_source = source_path
            if source_path != chosen_source:
                continue
            page_key = (source_path, int(citation["page_number"]))
            if page_key in seen_pages:
                continue
            preview_pages.append(
                {
                    "source_path": source_path,
                    "page_number": int(citation["page_number"]),
                    "score": round(float(citation["score"]), 4),
                }
            )
            seen_pages.add(page_key)
            if len(preview_pages) >= 3:
                return chosen_source, preview_pages

        if preview_pages:
            return chosen_source, preview_pages
        if fallback_pages:
            return preferred_preview_source, fallback_pages
        if grounded_pages:
            top_grounded = sorted(grounded_pages, key=lambda page: float(page.get("score", 0)), reverse=True)[:3]
            source = top_grounded[0].get("source_path") if top_grounded else preferred_preview_source
            return source, top_grounded
        return preferred_preview_source, []

    def build_source_line(self, answer_citations: list[dict]) -> str:
        citations_by_source: dict[str, list[int]] = defaultdict(list)
        for citation in answer_citations:
            citations_by_source[citation["file_name"]].append(int(citation["page_number"]))

        parts: list[str] = []
        for file_name, pages in citations_by_source.items():
            unique_pages = sorted(set(pages))
            page_ranges: list[str] = []
            start = unique_pages[0]
            end = unique_pages[0]
            for page_number in unique_pages[1:]:
                if page_number == end + 1:
                    end = page_number
                    continue
                page_ranges.append(f"p.{start}" if start == end else f"p.{start}-{end}")
                start = page_number
                end = page_number
            page_ranges.append(f"p.{start}" if start == end else f"p.{start}-{end}")
            parts.append(f"[{file_name}] " + ", ".join(page_ranges))
        return "\nSources: " + " | ".join(parts) if parts else ""

    def ensure_answer_source_line(
        self,
        answer: str,
        answer_citations: list[dict],
        use_retrieved_context: bool,
    ) -> str:
        if not answer or not use_retrieved_context or not answer_citations:
            return answer
        if self.should_suppress_citations(answer):
            return answer
        if "[source:" in answer:
            return answer
        if self.extract_answer_citations(answer):
            return answer
        unique_files = {citation["file_name"] for citation in answer_citations if citation.get("file_name")}
        if len(unique_files) <= 1:
            return answer
        return answer.rstrip() + self.build_source_line(answer_citations)

    def build_context_payload(
        self,
        rewritten_query: str,
        response_mode: str,
        top_score: float,
        preferred_preview_source: str | None,
        preview_pages: list[dict],
        context_items: list[dict],
        grounded_pages: list[dict],
        answer_citations: list[dict],
        preview_finalized: bool = False,
    ) -> dict:
        return {
            "query": rewritten_query,
            "mode": response_mode,
            "top_score": round(top_score, 4),
            "preferred_preview_source": preferred_preview_source,
            "preview_pages": preview_pages,
            "source_grounding": self.retrieval_service.aggregate_source_grounding(grounded_pages),
            "grounded_pages": grounded_pages,
            "answer_citations": answer_citations,
            "preview_finalized": preview_finalized,
            "items": self.retrieval_service.build_context_items_payload(context_items),
        }


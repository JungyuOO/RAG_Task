from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from app.services.retrieval_service import RetrievalService


class AnswerService:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def build_extractive_code_answer(self, context_items: list[dict]) -> str | None:
        snippets: list[dict[str, str]] = []
        seen_blocks: set[str] = set()

        for item in context_items:
            chunk = item["chunk"]
            text = str(chunk.get("text", "") or "")
            if not text.strip():
                continue
            for block in self._extract_code_candidates(text):
                normalized = re.sub(r"\s+", " ", block).strip().casefold()
                if len(normalized) < 24 or normalized in seen_blocks:
                    continue
                seen_blocks.add(normalized)
                snippets.append(
                    {
                        "file_name": Path(chunk["source_path"]).name,
                        "page_start": str(chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                        "page_end": str(chunk["metadata"].get("page_end") or chunk["metadata"].get("page_start") or chunk.get("page_number") or 1),
                        "code": block.strip(),
                    }
                )
                if len(snippets) >= 3:
                    break
            if len(snippets) >= 3:
                break

        if not snippets:
            return None

        parts = ["문서에 나온 예시 코드를 그대로 정리하면 아래와 같습니다."]
        for snippet in snippets:
            page_label = (
                f"p.{snippet['page_start']}"
                if snippet["page_start"] == snippet["page_end"]
                else f"p.{snippet['page_start']}-{snippet['page_end']}"
            )
            parts.append(f"[{snippet['file_name']}] {page_label}")
            parts.append("```yaml")
            parts.append(snippet["code"])
            parts.append("```")
        return "\n\n".join(parts).strip()

    def sanitize_answer(self, answer: str, use_retrieved_context: bool) -> str:
        if not answer:
            return answer
        if use_retrieved_context:
            return self._sanitize_retrieved_answer(answer)
        return self._sanitize_general_answer(answer)

    def _sanitize_retrieved_answer(self, answer: str) -> str:
        sanitized = answer.replace("\r\n", "\n").strip()
        lines: list[str] = []
        for raw_line in sanitized.split("\n"):
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            if self._is_retrieved_mode_negative_artifact(line):
                continue
            lines.append(raw_line.rstrip())

        compacted: list[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            compacted.append(line)
            previous_blank = is_blank
        return "\n".join(compacted).strip()

    def _sanitize_general_answer(self, answer: str) -> str:
        sanitized = answer.replace("\r\n", "\n").strip()
        lines: list[str] = []
        for raw_line in sanitized.split("\n"):
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            if self._is_general_mode_source_artifact(line):
                continue
            lines.append(raw_line.rstrip())

        compacted: list[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            compacted.append(line)
            previous_blank = is_blank
        return "\n".join(compacted).strip()

    def _is_general_mode_source_artifact(self, line: str) -> bool:
        normalized = line.casefold()
        if normalized in {"[general knowledge]", "general knowledge"}:
            return True
        if "uploaded library" in normalized:
            return True
        if "no specific content was retrieved" in normalized:
            return True
        if "general knowledge" in normalized and line.strip().startswith("[") and line.strip().endswith("]"):
            return True
        if "uploaded documents" in normalized and "supporting evidence" in normalized:
            return True
        if "retrieval mode:" in normalized:
            return True
        if re.fullmatch(r"\[[^\]\n]+\]\s*p\.\d+(?:-\d+)?(?:\s*\(.*\))?", line, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"sources:\s*.+", line, flags=re.IGNORECASE):
            return True
        return False

    def _is_retrieved_mode_negative_artifact(self, line: str) -> bool:
        normalized = line.casefold()
        patterns = (
            "제공된 문서에는",
            "문서에는",
            "포함되어 있지 않습니다",
            "구체적인 정의나 설명이 포함되어 있지 않습니다",
            "is not included in the provided document",
            "is not specifically described in the provided document",
        )
        if any(pattern.casefold() in normalized for pattern in patterns):
            return True
        return False

    def _extract_code_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        for fenced in re.findall(r"```(?:[\w+-]+)?\n(.*?)```", text, flags=re.DOTALL):
            block = fenced.strip()
            if self._looks_like_code_block(block):
                candidates.append(block)

        lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
        current: list[str] = []
        in_code = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if current:
                    block = "\n".join(current).strip()
                    if self._looks_like_code_block(block):
                        candidates.append(block)
                    current = []
                    in_code = False
                continue

            if self._is_code_line(line):
                current.append(line)
                in_code = True
                continue

            if in_code and self._looks_like_code_continuation(line):
                current.append(line)
                continue

            if current:
                block = "\n".join(current).strip()
                if self._looks_like_code_block(block):
                    candidates.append(block)
                current = []
            in_code = False

        if current:
            block = "\n".join(current).strip()
            if self._looks_like_code_block(block):
                candidates.append(block)

        return candidates

    def _is_code_line(self, line: str) -> bool:
        stripped = line.strip()
        lowered = stripped.casefold()
        if stripped.startswith("# "):
            return True
        if lowered.startswith(("apiversion:", "kind:", "metadata:", "spec:", "data:", "stringdata:")):
            return True
        if lowered.startswith(("oc ", "kubectl ", "helm ", "cat ", "echo ")):
            return True
        if re.match(r"^[A-Za-z0-9_.-]+\.(ya?ml|json)$", stripped):
            return True
        return False

    def _looks_like_code_continuation(self, line: str) -> bool:
        lowered = line.casefold()
        if lowered.startswith(("-", "name:", "image:", "path:", "storage:", "accessmodes:", "resources:", "requests:", "claimname:", "mountpath:", "containers:", "volumes:", "templ", "selector:", "matchlabels:", "replicas:", "provisioner:", "parameters:", "type:", "allowvolumeexpansion:", "mountoptions:", "persistentvolumeclaim:", "volumemounts:", "volumemode:", "persistentvolumereclaimpolicy:")):
            return True
        if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*:\s*", line):
            return True
        return line.startswith(("  ", "\t"))

    def _looks_like_code_block(self, block: str) -> bool:
        lowered = block.casefold()
        indicators = (
            "apiversion:",
            "kind:",
            "metadata:",
            "spec:",
            "oc apply -f",
            "oc get ",
            "kubectl ",
        )
        return any(indicator in lowered for indicator in indicators)

    def build_answer_citation_payload(
        self,
        answer: str,
        context_items: list[dict],
        grounded_pages: list[dict],
        preferred_preview_source: str | None,
    ) -> list[dict]:
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

        fallback_pages = [
            item
            for item in grounded_pages
            if item["source_path"] == preferred_preview_source
        ] or grounded_pages
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
        return citations

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
        return preferred_preview_source, fallback_pages

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
        if self.extract_answer_citations(answer):
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

from __future__ import annotations

import re

from apps.api.schemas.chat import CopilotChatSourceItem
from apps.api.rag.query.query_features import tokenize_query


class CitationGroundingValidator:
    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    _GENERIC_ALIGNMENT_TOKENS = {
        "and",
        "the",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "uses",
        "use",
        "define",
        "defines",
        "show",
        "list",
        "guide",
    }

    def validate(
        self,
        answer: str,
        sources: list[CopilotChatSourceItem],
        *,
        enforce_alignment: bool = True,
    ) -> str:
        if not answer or not sources:
            return answer

        def replace(match: re.Match[str]) -> str:
            try:
                index = int(match.group(1))
            except Exception:
                return ""
            if index < 1 or index > len(sources):
                return ""
            return f"[{index}]"

        cleaned = self._CITATION_PATTERN.sub(replace, answer)
        cleaned = re.sub(r"(?:\[(\d+)\]){2,}", lambda m: m.group(0)[: len(f'[{m.group(1)}]')], cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned).strip()
        if not enforce_alignment:
            return cleaned

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        if not paragraphs:
            return cleaned

        validated: list[str] = []
        for paragraph in paragraphs:
            citations = [int(match) for match in self._CITATION_PATTERN.findall(paragraph)]
            if not citations:
                validated.append(paragraph)
                continue
            body = self._CITATION_PATTERN.sub("", paragraph).strip()
            valid_citations = [
                index
                for index in citations
                if self._supports_paragraph(body, sources[index - 1])
            ]
            body = re.sub(r"\s+$", "", body)
            if valid_citations:
                marker = "".join(f"[{index}]" for index in dict.fromkeys(valid_citations))
                validated.append(f"{body}{marker}".strip())
            else:
                if body:
                    validated.append(body)
        return "\n\n".join(validated).strip()

    @staticmethod
    def _supports_paragraph(paragraph: str, source: CopilotChatSourceItem) -> bool:
        paragraph_tokens = CitationGroundingValidator._alignment_tokens(paragraph)
        if not paragraph_tokens:
            return False
        source_context = " ".join(
            [
                str(source.metadata.get("section_title") or source.label or ""),
                str(source.metadata.get("preview_text") or ""),
                str(source.metadata.get("synthesis_text") or ""),
            ]
        )
        source_tokens = CitationGroundingValidator._alignment_tokens(source_context)
        if not source_tokens:
            return False
        overlap = len(paragraph_tokens.intersection(source_tokens))
        overlap_ratio = overlap / max(len(paragraph_tokens), 1)
        return overlap >= 1 or overlap_ratio >= 0.2

    @classmethod
    def _alignment_tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in tokenize_query(text)
            if token not in cls._GENERIC_ALIGNMENT_TOKENS
        }


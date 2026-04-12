"""문장별 인라인 인용 태그 자동 주입.

LLM 프롬프트에서 "문장마다 출처 태그 달기"를 지시해도 실제로는 지키지 않는
경우가 많아, 답변 후처리 단계에서 각 문장에 가장 근접한 retrieval chunk의
[file.pdf p.N] 태그를 강제 주입한다.

원칙:
  - 이미 인라인 태그가 있는 문장은 건드리지 않는다 (LLM이 달았다면 존중)
  - 코드 블록, 빈 줄, 헤딩 등 문장이 아닌 영역은 건드리지 않는다
  - 문장-청크 매칭: substring 포함 → 키워드 overlap → rerank fallback 순
"""
from __future__ import annotations

import re
from pathlib import Path

from app.rag.utils import keyword_overlap_score, tokenize


# 한국어 + 영어 문장 종결 패턴
_SENTENCE_END_RE = re.compile(
    r"(?<=다\.)(?=\s|$)|(?<=요\.)(?=\s|$)|(?<=음\.)(?=\s|$)|"
    r"(?<=니다\.)(?=\s|$)|(?<=임\.)(?=\s|$)|(?<=[.!?])(?=\s|$)"
)

# 이미 인라인 인용이 달린 문장 판별 (answer_citation.py 패턴과 호환)
_EXISTING_CITATION_RE = re.compile(
    r"\[[^\[\]\n]+?\.pdf[^\[\]\n]*\]|\[source:[^\]]+\]",
    re.IGNORECASE,
)
_SOURCE_TAG_RE = re.compile(r"\s*\[source:([^\]]+)\]")

# 본문 문장으로 취급하지 않을 라인 prefix
_SKIP_LINE_PREFIXES = (
    "Sources:", "출처:", "Source:", "```",
    "#", "##", "###", ">",
)

# 너무 짧은 문장은 주입 대상에서 제외 (한국어 기준 8자)
_MIN_SENTENCE_CHAR = 8

# 키워드 overlap 최소 임계치
_MIN_OVERLAP_SCORE = 0.05

# substring 매칭용: 영어 연속 단어 최소 길이 (너무 짧으면 false positive)
_MIN_SUBSTR_LEN = 12


class InlineCitationMixin:
    """문장별 인라인 인용 태그를 주입하는 후처리 mixin."""

    def inject_inline_citations(
        self,
        answer: str,
        selected_context_items: list[dict],
    ) -> str:
        if not answer or not selected_context_items:
            return answer

        chunk_profiles = self._build_chunk_profiles(selected_context_items)
        if not chunk_profiles:
            return answer

        out_lines: list[str] = []
        in_code_block = False
        for raw_line in answer.split("\n"):
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                out_lines.append(raw_line)
                continue
            if in_code_block or not stripped:
                out_lines.append(raw_line)
                continue
            if any(stripped.startswith(prefix) for prefix in _SKIP_LINE_PREFIXES):
                out_lines.append(raw_line)
                continue
            out_lines.append(self._inject_into_line(raw_line, chunk_profiles))

        return "\n".join(out_lines)

    @staticmethod
    def collapse_single_citation_answer(answer: str) -> str:
        source_tags = _SOURCE_TAG_RE.findall(answer or "")
        unique_tags = []
        for tag in source_tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
        if len(unique_tags) != 1:
            return answer

        stripped = _SOURCE_TAG_RE.sub("", answer or "").rstrip()
        if stripped.endswith("```"):
            return f"{stripped}\n\n[source:{unique_tags[0]}]"
        return f"{stripped} [source:{unique_tags[0]}]"

    def _inject_into_line(self, line: str, chunk_profiles: list[dict]) -> str:
        list_match = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$", line)
        if list_match:
            prefix, body = list_match.group(1), list_match.group(2)
        else:
            prefix, body = "", line

        if not body.strip():
            return line

        if _EXISTING_CITATION_RE.search(body):
            return line

        if len(body.strip()) < _MIN_SENTENCE_CHAR:
            return line

        sentence_parts = self._split_sentences(body)
        if not sentence_parts:
            sentence_parts = [body]

        matches: list[dict | None] = [self._match_sentence_to_chunk(part, chunk_profiles) for part in sentence_parts]
        matched_refs = {
            (match["file_name"], match["page_number"])
            for match in matches
            if match is not None
        }
        if not matched_refs:
            return line

        if len(matched_refs) == 1:
            best = next(match for match in matches if match is not None)
            tag = f" [source:{best['file_name']}:p{best['page_number']}:L1-999]"
            return prefix + body.rstrip() + tag

        rebuilt_parts: list[str] = []
        for part, match in zip(sentence_parts, matches):
            stripped = part.rstrip()
            if not stripped or match is None:
                rebuilt_parts.append(part)
                continue
            tag = f" [source:{match['file_name']}:p{match['page_number']}:L1-999]"
            trailing_ws_len = len(part) - len(part.rstrip())
            trailing_ws = part[-trailing_ws_len:] if trailing_ws_len > 0 else ""
            rebuilt_parts.append(stripped + tag + trailing_ws)

        return prefix + "".join(rebuilt_parts).rstrip()

    @staticmethod
    def _substring_score(sentence: str, chunk_text: str) -> float:
        """문장 내 영어 구문이 청크 텍스트에 포함되어 있는 비율."""
        phrases = re.findall(r"[A-Za-z][A-Za-z0-9_./-]+(?:\s+[A-Za-z][A-Za-z0-9_./-]+)*", sentence)
        if not phrases:
            return 0.0
        matched_chars = 0
        total_chars = 0
        for phrase in phrases:
            phrase_clean = phrase.strip()
            if len(phrase_clean) < _MIN_SUBSTR_LEN:
                # 짧은 구문은 개별 단어로 분해해서 매칭
                words = phrase_clean.lower().split()
                for word in words:
                    if len(word) >= 3:
                        total_chars += len(word)
                        if word in chunk_text.lower():
                            matched_chars += len(word)
            else:
                total_chars += len(phrase_clean)
                if phrase_clean.lower() in chunk_text.lower():
                    matched_chars += len(phrase_clean)
                else:
                    # 긴 구문이 완전 매칭 안 되면 단어 단위로 fallback
                    words = phrase_clean.lower().split()
                    for word in words:
                        if len(word) >= 3:
                            if word in chunk_text.lower():
                                matched_chars += len(word)
        return matched_chars / max(total_chars, 1)

    def _match_sentence_to_chunk(self, sentence: str, chunk_profiles: list[dict]) -> dict | None:
        """문장에 가장 잘 맞는 청크를 찾는다.

        매칭 전략 (우선순위):
        1. substring 포함도 — 문장 내 영어 표현이 청크에 포함되어 있는 비율
        2. 키워드 overlap — 토큰 기반 겹침
        3. rerank fallback — 위 두 가지 다 안 되면 rerank 최고 청크
        """
        sentence_tokens = tokenize(sentence)

        best: dict | None = None
        best_combined = 0.0

        for profile in chunk_profiles:
            # 1. substring 매칭 (영어 구문 기반)
            substr_score = self._substring_score(sentence, profile["text"])
            # 2. 키워드 overlap
            overlap = keyword_overlap_score(sentence_tokens, profile["tokens"])
            # 합산 점수: substring에 더 높은 가중치
            combined = substr_score * 0.6 + overlap * 0.3 + profile["rerank_score"] * 0.1

            if combined > best_combined:
                best = profile
                best_combined = combined

        return best

    @staticmethod
    def _split_sentences(body: str) -> list[str]:
        """문장 종결자 뒤에서 split. 각 segment는 종결자 포함."""
        if not body.strip():
            return [body]

        positions: list[int] = []
        for match in _SENTENCE_END_RE.finditer(body):
            positions.append(match.start())

        if not positions:
            return [body]

        segments: list[str] = []
        start = 0
        for pos in positions:
            segments.append(body[start:pos])
            start = pos
        tail = body[start:]
        if tail:
            segments.append(tail)

        result: list[str] = []
        for seg in segments:
            if not seg:
                continue
            result.append(seg)
        return result

    @staticmethod
    def _build_chunk_profiles(selected_context_items: list[dict]) -> list[dict]:
        profiles: list[dict] = []
        for item in selected_context_items:
            chunk = item.get("chunk") or {}
            text = str(chunk.get("text", "") or "")
            if not text:
                continue
            source_path = str(chunk.get("source_path") or "")
            if not source_path:
                continue
            metadata = chunk.get("metadata") or {}
            page_start = chunk.get("page_number") or metadata.get("page_start")
            if not page_start:
                continue
            try:
                page_number = int(page_start)
            except (TypeError, ValueError):
                continue
            profiles.append({
                "file_name": Path(source_path).name,
                "page_number": page_number,
                "text": text,
                "tokens": tokenize(text),
                "rerank_score": float(item.get("rerank_score", 0.0)),
            })
        return profiles

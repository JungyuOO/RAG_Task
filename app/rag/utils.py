from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")

# 한국어 조사/어미 — 빈도 높은 것만 선별. 긴 것부터 매칭해야 "에서"가 "에"보다 먼저 제거됨.
_KO_SUFFIXES = sorted([
    # 조사
    "에서는", "으로는", "에서도", "으로도", "에서의",
    "에서", "으로", "에게", "까지", "부터", "처럼", "만큼",
    "에는", "에도", "와는", "과는", "이란", "란은",
    "이는", "에의",
    "의", "를", "을", "이", "가", "은", "는", "로", "와", "과",
    "에", "도", "만", "란", "든", "며",
    # 흔한 활용 어미
    "해줘", "해줄", "해서", "해주", "하는", "에대해", "대해",
], key=len, reverse=True)

_KO_CHAR_RANGE = re.compile(r"[가-힣]")


def normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def strip_korean_suffix(token: str) -> str:
    """한국어 토큰에서 조사/어미를 제거한다. 결과가 2자 미만이면 원본 반환."""
    if not _KO_CHAR_RANGE.search(token):
        return token
    for suffix in _KO_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            stripped = token[: -len(suffix)]
            if len(stripped) >= 2:
                return stripped
    return token


def tokenize(text: str) -> list[str]:
    raw_tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [strip_korean_suffix(token) for token in raw_tokens]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def keyword_overlap_score(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    query_counter = Counter(query_tokens)
    candidate_counter = Counter(candidate_tokens)
    overlap = sum(min(query_counter[token], candidate_counter[token]) for token in query_counter)
    return overlap / max(len(query_tokens), 1)

from __future__ import annotations

import hashlib
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
_KO_CHAR_RANGE = re.compile(r"[가-힣]")
_KO_SUFFIXES = sorted(
    [
        "으로부터",
        "에서부터",
        "에게서",
        "까지는",
        "까지도",
        "으로는",
        "에서는",
        "과의",
        "으로",
        "로서",
        "로써",
        "로는",
        "로도",
        "에게",
        "에서",
        "처럼",
        "만큼",
        "보다",
        "까지",
        "부터",
        "에는",
        "에도",
        "에서",
        "에게",
        "으로",
        "와의",
        "과는",
        "과도",
        "에서",
        "에게",
        "으로",
        "로는",
        "로도",
        "이다",
        "이고",
        "이며",
        "이랑",
        "랑",
        "으로",
        "에서",
        "에게",
        "까지",
        "부터",
        "처럼",
        "만큼",
        "에게",
        "에는",
        "으로",
        "에서",
        "하고",
        "이라",
        "라고",
        "이라는",
        "라는",
        "이나",
        "나",
        "이",
        "가",
        "은",
        "는",
        "을",
        "를",
        "와",
        "과",
        "의",
        "도",
        "만",
        "로",
        "에",
        "게",
        "서",
        "요",
        "좀",
        "중",
    ],
    key=len,
    reverse=True,
)
_TECH_TOKEN_WHITELIST = {"pv", "pvc", "rbac", "scc", "api", "cli", "yaml", "json", "oc"}


def normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def strip_korean_suffix(token: str) -> str:
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


def normalize_query_keywords(text: str, keywords: list[str] | None = None) -> list[str]:
    candidates: list[str] = []
    if keywords:
        for keyword in keywords:
            candidates.extend(TOKEN_PATTERN.findall(normalize_text(str(keyword))))
    if text:
        candidates.extend(TOKEN_PATTERN.findall(normalize_text(text)))

    normalized_keywords: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        lowered = strip_korean_suffix(token.lower())
        if not lowered:
            continue
        if len(lowered) < 2 and lowered not in _TECH_TOKEN_WHITELIST:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized_keywords.append(lowered)
    return normalized_keywords


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def keyword_overlap_score(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    query_counter = Counter(query_tokens)
    candidate_counter = Counter(candidate_tokens)
    overlap = sum(min(query_counter[token], candidate_counter[token]) for token in query_counter)
    return overlap / max(len(query_tokens), 1)

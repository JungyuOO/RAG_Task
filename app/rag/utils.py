from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


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

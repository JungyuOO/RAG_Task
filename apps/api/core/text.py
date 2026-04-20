from __future__ import annotations

import hashlib
import re


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_./+-]+")
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
        "께서",
        "과는",
        "과도",
        "라고",
        "라는",
        "이라",
        "이고",
        "이며",
        "이랑",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "과",
        "와",
        "로",
        "도",
        "만",
        "에",
        "의",
    ],
    key=len,
    reverse=True,
)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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

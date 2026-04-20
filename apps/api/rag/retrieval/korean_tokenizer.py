from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_-]+")
_HANGUL_PATTERN = re.compile(r"[가-힣]")


def tokenize_with_char_ngrams(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall(str(text or "").casefold()):
        if len(raw_token) < 2:
            continue
        tokens.append(raw_token)
        if _HANGUL_PATTERN.search(raw_token):
            tokens.extend(_korean_ngrams(raw_token))
    return tokens


def _korean_ngrams(token: str) -> list[str]:
    out: list[str] = []
    for n in (2, 3):
        if len(token) < n:
            continue
        for index in range(len(token) - n + 1):
            chunk = token[index : index + n]
            if chunk == token:
                continue
            out.append(chunk)
    return out

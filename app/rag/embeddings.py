from __future__ import annotations

import hashlib
import math

from app.rag.utils import l2_normalize, tokenize


class HashingEmbedder:
    """A framework-free dense vectorizer based on hashing token statistics."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            index = token_hash % self.dim
            sign = -1.0 if (token_hash >> 1) & 1 else 1.0
            vector[index] += sign * (1.0 + math.log(len(token) + 1))

        return l2_normalize(vector)

from __future__ import annotations

import hashlib
import math

from app.rag.utils import l2_normalize, tokenize


class HashingEmbedder:
    """외부 모델 없이 토큰 해싱으로 고정 차원 벡터를 생성하는 임베더.

    단일 토큰 해싱에 더해 바이그램(bigram) 해싱과 위치 가중치를 적용하여
    토큰 순서와 인접 토큰 조합이 벡터에 반영되도록 한다.
    결정론적이며 외부 임베딩 서비스를 사용하지 않는다.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        """텍스트를 dim차원 벡터로 인코딩한다.

        1) 유니그램 해싱 — 각 토큰의 SHA-256 해시로 벡터 인덱스와 부호를 결정
        2) 바이그램 해싱 — 인접 토큰 쌍의 결합 해시로 토큰 순서 정보 반영
        3) 위치 감쇠 — 문서 앞쪽 토큰에 약간 더 높은 가중치 부여
        """
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector

        total_tokens = len(tokens)
        for position, token in enumerate(tokens):
            position_weight = 1.0 - 0.3 * (position / max(total_tokens, 1))
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            index = token_hash % self.dim
            sign = -1.0 if (token_hash >> 1) & 1 else 1.0
            vector[index] += sign * (1.0 + math.log(len(token) + 1)) * position_weight

        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            bigram_hash = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            index = bigram_hash % self.dim
            sign = -1.0 if (bigram_hash >> 1) & 1 else 1.0
            vector[index] += sign * 0.5

        return l2_normalize(vector)

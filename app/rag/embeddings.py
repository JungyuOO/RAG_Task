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
            # 위치 감쇠: 앞쪽 토큰일수록 약간 높은 가중치 (1.0 ~ 0.7).
            # 0.3 감쇠 폭은 문서 앞부분의 제목/핵심어를 강조하되
            # 뒤쪽 토큰도 0.7 이상의 기여를 유지하는 경험적 균형점.
            position_weight = 1.0 - 0.3 * (position / max(total_tokens, 1))
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            index = token_hash % self.dim
            # 해시의 비트 1로 부호 결정 — 벡터 차원에 +/- 분산을 부여하여
            # 서로 다른 토큰이 같은 인덱스에 매핑될 때 상쇄 효과 생성
            sign = -1.0 if (token_hash >> 1) & 1 else 1.0
            # 토큰 길이의 로그 스케일링: 긴 토큰(전문 용어)이 짧은 토큰(조사)보다
            # 더 큰 기여를 하도록 가중. log로 포화시켜 극단적 길이 차이를 완화.
            vector[index] += sign * (1.0 + math.log(len(token) + 1)) * position_weight

        # 바이그램: 인접 토큰 쌍을 결합 해싱하여 토큰 순서 정보를 벡터에 반영.
        # 가중치 0.5는 유니그램 대비 보조적 역할 — 순서 정보를 추가하되
        # 단일 토큰의 의미 기여를 압도하지 않는 수준.
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            bigram_hash = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            index = bigram_hash % self.dim
            sign = -1.0 if (bigram_hash >> 1) & 1 else 1.0
            vector[index] += sign * 0.5

        return l2_normalize(vector)

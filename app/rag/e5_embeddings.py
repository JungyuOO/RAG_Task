from __future__ import annotations

import math


class E5Embedder:
    """intfloat/multilingual-e5-small 모델을 사용하는 임베더.

    E5 모델 규약에 따라 검색 쿼리에는 "query: " 접두사를,
    인덱싱 청크에는 "passage: " 접두사를 붙인다.
    모델은 첫 encode 호출 시 지연 로드(lazy load)된다.
    출력 벡터는 384차원이며 L2 정규화된다.
    """

    MODEL_NAME = "intfloat/multilingual-e5-small"

    def __init__(self) -> None:
        self.dim = 384
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def encode(self, text: str) -> list[float]:
        """쿼리 텍스트를 384차원 벡터로 인코딩한다. "query: " 접두사를 사용한다."""
        prefixed = "query: " + text
        return self._encode_raw(prefixed)

    def encode_passage(self, text: str) -> list[float]:
        """청크(passage) 텍스트를 384차원 벡터로 인코딩한다. "passage: " 접두사를 사용한다."""
        prefixed = "passage: " + text
        return self._encode_raw(prefixed)

    def _encode_raw(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=False)
        vector = embedding.tolist()
        return _l2_normalize(vector)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]

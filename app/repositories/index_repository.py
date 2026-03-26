from __future__ import annotations

from app.rag.index import VectorIndex
from app.rag.types import Chunk


class IndexRepository:
    """벡터 인덱스 데이터 접근 계층 — VectorIndex와 서비스 계층 사이의 추상 경계.

    인덱스 저장소 교체(예: 파일→원격 DB) 시 이 인터페이스만 재구현하면
    서비스 계층 코드 변경 없이 확장할 수 있다.

    ``load()`` 결과를 인메모리 캐싱하여 매 쿼리마다 DB 전체 스캔을 피한다.
    ``save()``, ``upsert_document()``, ``delete_document()`` 호출 시 캐시를 무효화한다.
    """

    def __init__(self, backend: VectorIndex) -> None:
        self.backend = backend
        self._cache: list[dict] | None = None

    def save(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.backend.save(chunks, vectors)
        self._cache = None

    def load(self) -> list[dict]:
        if self._cache is not None:
            return self._cache
        self._cache = self.backend.load()
        return self._cache

    def invalidate_cache(self) -> None:
        """인메모리 캐시를 수동으로 무효화한다."""
        self._cache = None

    def upsert_document(self, source_path: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.backend.upsert_document(source_path, chunks, vectors)
        self._cache = None

    def delete_document(self, source_path: str) -> None:
        self.backend.delete_document(source_path)
        self._cache = None

    def list_documents(self) -> list[dict]:
        return self.backend.list_documents()


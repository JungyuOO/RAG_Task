from __future__ import annotations

from app.rag.index import VectorIndex
from app.rag.types import Chunk


class IndexRepository:
    """벡터 인덱스 데이터 접근 계층 — VectorIndex와 서비스 계층 사이의 추상 경계.

    인덱스 저장소 교체(예: 파일→원격 DB) 시 이 인터페이스만 재구현하면
    서비스 계층 코드 변경 없이 확장할 수 있다.
    """

    def __init__(self, backend: VectorIndex) -> None:
        self.backend = backend

    def save(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.backend.save(chunks, vectors)

    def load(self) -> list[dict]:
        return self.backend.load()

    def delete_document(self, source_path: str) -> None:
        self.backend.delete_document(source_path)

    def list_documents(self) -> list[dict]:
        return self.backend.list_documents()


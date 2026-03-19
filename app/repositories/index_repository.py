from __future__ import annotations

from app.rag.index import VectorIndex
from app.rag.types import Chunk


class IndexRepository:
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


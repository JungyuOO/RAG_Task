from __future__ import annotations

from contextlib import closing
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path

from app.rag.types import Chunk


class VectorIndex:
    """청크와 벡터를 SQLite에 저장하고 전체 로드하는 벡터 인덱스.

    외부 벡터 DB 없이 SQLite만으로 구현하며, 검색 시 전체 벡터를 메모리에
    로드한 뒤 HybridRetriever가 점수를 계산하는 구조이다.
    """

    def __init__(self, index_dir: Path) -> None:
        self.db_path = index_dir / "rag_store.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """SQLite 연결을 생성한다. WAL 모드와 busy_timeout으로 동시성을 확보한다."""
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _connection(self):
        with closing(self._connect()) as connection:
            with connection:
                yield connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source_path TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    extension TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tokens_json TEXT NOT NULL,
                    page_number INTEGER,
                    metadata_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    FOREIGN KEY (source_path) REFERENCES documents(source_path)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
                """
            )

    def save(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")

            documents: dict[str, tuple[str, str, str]] = {}
            for chunk in chunks:
                source = Path(chunk.source_path)
                documents[chunk.source_path] = (chunk.source_path, source.name, source.suffix.lower())

            connection.executemany(
                "INSERT INTO documents (source_path, file_name, extension) VALUES (?, ?, ?)",
                list(documents.values()),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, source_path, text, tokens_json,
                    page_number, metadata_json, vector_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.source_path,
                        chunk.text,
                        json.dumps(chunk.tokens, ensure_ascii=False),
                        chunk.page_number,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        json.dumps(vector, ensure_ascii=False),
                    )
                    for chunk, vector in zip(chunks, vectors)
                ],
            )

    def delete_document(self, source_path: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
            connection.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))

    def load(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, doc_id, source_path, text, tokens_json,
                       page_number, metadata_json, vector_json
                FROM chunks
                """
            ).fetchall()

        return [
            {
                "chunk": {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "source_path": row["source_path"],
                    "text": row["text"],
                    "tokens": json.loads(row["tokens_json"]),
                    "page_number": row["page_number"],
                    "metadata": json.loads(row["metadata_json"]),
                },
                "vector": json.loads(row["vector_json"]),
            }
            for row in rows
        ]

    def list_documents(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    d.source_path,
                    d.file_name,
                    d.extension,
                    COUNT(c.chunk_id) AS indexed_chunks,
                    COUNT(DISTINCT c.page_number) AS indexed_pages
                FROM documents d
                LEFT JOIN chunks c ON c.source_path = d.source_path
                GROUP BY d.source_path, d.file_name, d.extension
                ORDER BY d.file_name
                """
            ).fetchall()
            loader_rows = connection.execute("SELECT source_path, metadata_json FROM chunks").fetchall()

        loaders_by_path: dict[str, set[str]] = {}
        for row in loader_rows:
            loader = json.loads(row["metadata_json"]).get("loader")
            if loader:
                loaders_by_path.setdefault(row["source_path"], set()).add(loader)

        return [
            {
                "file_name": row["file_name"],
                "source_path": row["source_path"],
                "extension": row["extension"],
                "indexed_pages": row["indexed_pages"] or 0,
                "indexed_chunks": row["indexed_chunks"] or 0,
                "loaders": sorted(loaders_by_path.get(row["source_path"], set())),
            }
            for row in rows
        ]

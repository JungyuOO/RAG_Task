from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.rag.types import Chunk


class VectorIndex:
    """청크와 벡터를 PostgreSQL에 저장하고 전체 로드하는 벡터 인덱스.

    외부 벡터 DB(pgvector 등) 없이 PostgreSQL을 스토리지로만 사용하며,
    검색 시 전체 벡터를 메모리에 로드한 뒤 HybridRetriever가 점수를 계산하는 구조이다.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn)
        self._initialize()

    @contextmanager
    def _connection(self):
        connection = self._pool.getconn()
        connection.autocommit = False
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self._pool.putconn(connection)

    def close(self) -> None:
        """커넥션 풀을 닫는다."""
        self._pool.closeall()

    def _initialize(self) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        source_path TEXT PRIMARY KEY,
                        file_name TEXT NOT NULL,
                        extension TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
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
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path)
                    """
                )

    def save(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks")
                cursor.execute("DELETE FROM documents")

                documents: dict[str, tuple[str, str, str]] = {}
                for chunk in chunks:
                    source = Path(chunk.source_path)
                    documents[chunk.source_path] = (chunk.source_path, source.name, source.suffix.lower())

                for doc_values in documents.values():
                    cursor.execute(
                        "INSERT INTO documents (source_path, file_name, extension) VALUES (%s, %s, %s)",
                        doc_values,
                    )

                for chunk, vector in zip(chunks, vectors):
                    cursor.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, doc_id, source_path, text, tokens_json,
                            page_number, metadata_json, vector_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.source_path,
                            chunk.text,
                            json.dumps(chunk.tokens, ensure_ascii=False),
                            chunk.page_number,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            json.dumps(vector, ensure_ascii=False),
                        ),
                    )

    def upsert_document(self, source_path: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """단일 문서의 청크를 교체한다. 기존 데이터를 삭제하고 새로 삽입."""
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks WHERE source_path = %s", (source_path,))
                cursor.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))
                if not chunks:
                    return
                from pathlib import Path
                source = Path(chunks[0].source_path)
                cursor.execute(
                    "INSERT INTO documents (source_path, file_name, extension) VALUES (%s, %s, %s)",
                    (source_path, source.name, source.suffix.lower()),
                )
                for chunk, vector in zip(chunks, vectors):
                    cursor.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, doc_id, source_path, text, tokens_json,
                            page_number, metadata_json, vector_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.source_path,
                            chunk.text,
                            json.dumps(chunk.tokens, ensure_ascii=False),
                            chunk.page_number,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            json.dumps(vector, ensure_ascii=False),
                        ),
                    )

    def delete_document(self, source_path: str) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks WHERE source_path = %s", (source_path,))
                cursor.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))

    def load(self) -> list[dict]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id, doc_id, source_path, text, tokens_json,
                           page_number, metadata_json, vector_json
                    FROM chunks
                    """
                )
                rows = cursor.fetchall()

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
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
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
                )
                rows = cursor.fetchall()
                cursor.execute("SELECT source_path, metadata_json FROM chunks")
                loader_rows = cursor.fetchall()

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

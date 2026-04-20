from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without pg bindings
    psycopg2 = None  # type: ignore[assignment]

logger = logging.getLogger("rag.retrieval")


class PgvectorIndex:
    def __init__(self, dsn: str) -> None:
        if psycopg2 is None:
            raise RuntimeError("pgvector access requires psycopg2 to be installed.")
        self.dsn = dsn
        self._pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn)
        self._initialize()

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(format(float(value), ".12g") for value in vector) + "]"

    @staticmethod
    def _build_filter_clause(
        *,
        source_paths: list[str] | None = None,
    ) -> tuple[str, list[object]]:
        conditions: list[str] = []
        params: list[object] = []
        if source_paths:
            conditions.append("source_path = ANY(%s)")
            params.append(source_paths)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return where_clause, params

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

    def _initialize(self) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
                        chunk_order INTEGER,
                        page_number INTEGER,
                        metadata_json TEXT NOT NULL,
                        embedding vector,
                        vector_json TEXT NOT NULL,
                        FOREIGN KEY (source_path) REFERENCES documents(source_path)
                    )
                    """
                )
                cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector")
                cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_order INTEGER")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path)")
                try:
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
                    )
                except Exception as exc:
                    logger.warning("skip hnsw index creation: %s", exc)

    def search_by_embedding(
        self,
        query_vector: list[float],
        *,
        limit: int,
        source_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        where_clause, params = self._build_filter_clause(source_paths=source_paths)
        where_clause = f"{where_clause} AND embedding IS NOT NULL" if where_clause else "WHERE embedding IS NOT NULL"
        vector_literal = self._vector_literal(query_vector)
        sql_params: list[object] = [vector_literal, *params, vector_literal, max(int(limit), 1)]
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        chunk_id, doc_id, source_path, text, tokens_json,
                        chunk_order, page_number, metadata_json, vector_json,
                        embedding <=> %s::vector AS distance
                    FROM chunks
                    {where_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    sql_params,
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
                    "chunk_order": row["chunk_order"],
                    "page_number": row["page_number"],
                    "metadata": json.loads(row["metadata_json"]),
                },
                "vector": json.loads(row["vector_json"]),
                "distance": float(row["distance"]),
            }
            for row in rows
        ]

    def upsert_document(self, source_path: str, chunks: list[Any], vectors: list[list[float]]) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks WHERE source_path = %s", (source_path,))
                cursor.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))
                if not chunks:
                    return
                source = Path(getattr(chunks[0], "source_path"))
                cursor.execute(
                    "INSERT INTO documents (source_path, file_name, extension) VALUES (%s, %s, %s)",
                    (source_path, source.name, source.suffix.lower()),
                )
                for chunk, vector in zip(chunks, vectors):
                    metadata = dict(getattr(chunk, "metadata", {}) or {})
                    if not metadata.get("section_title"):
                        metadata["section_title"] = getattr(chunk, "section_title", "") or ""
                    if not metadata.get("section_path"):
                        metadata["section_path"] = list(getattr(chunk, "section_path", []) or [])
                    if not metadata.get("html_anchor"):
                        metadata["html_anchor"] = getattr(chunk, "html_anchor", "") or ""
                    cursor.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, doc_id, source_path, text, tokens_json, chunk_order,
                            page_number, metadata_json, embedding, vector_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                        """,
                        (
                            getattr(chunk, "chunk_id"),
                            getattr(chunk, "doc_id"),
                            getattr(chunk, "source_path"),
                            getattr(chunk, "text"),
                            json.dumps(list(getattr(chunk, "tokens", [])), ensure_ascii=False),
                            getattr(chunk, "chunk_order", None),
                            getattr(chunk, "page_number") or metadata.get("page_start"),
                            json.dumps(metadata, ensure_ascii=False),
                            self._vector_literal(vector),
                            json.dumps(vector, ensure_ascii=False),
                        ),
                    )

    def get_indexed_source_paths(self) -> set[str]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT source_path FROM documents")
                rows = cursor.fetchall()
        return {row[0] for row in rows}

    def get_index_stats(self) -> dict[str, int]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM documents")
                document_count = int(cursor.fetchone()[0])
                cursor.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = int(cursor.fetchone()[0])
        return {
            "documents": document_count,
            "chunks": chunk_count,
        }

    def clear_index(self) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks")
                cursor.execute("DELETE FROM documents")

    def get_document_chunk_counts(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT source_path, COUNT(*) AS chunk_count
                    FROM chunks
                    GROUP BY source_path
                    ORDER BY source_path ASC
                    """
                )
                rows = cursor.fetchall()
        return [
            {
                "source_path": str(row["source_path"] or ""),
                "chunk_count": int(row["chunk_count"] or 0),
            }
            for row in rows
        ]

    def list_chunks(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT source_path, chunk_id, text, chunk_order, page_number, metadata_json
                    FROM chunks
                    ORDER BY source_path ASC, chunk_order ASC NULLS LAST, page_number ASC NULLS LAST, chunk_id ASC
                    """
                )
                rows = cursor.fetchall()
        return [
            {
                "source_path": str(row["source_path"] or ""),
                "chunk_id": str(row["chunk_id"] or ""),
                "text": str(row["text"] or ""),
                "chunk_order": row["chunk_order"],
                "page_number": row["page_number"],
                "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            }
            for row in rows
        ]


class PgvectorIndexRepository:
    def __init__(self, backend: PgvectorIndex, rag_source_dir: Path | None = None) -> None:
        self.backend = backend
        self.rag_source_dir = rag_source_dir

    def _normalize_source_path(self, stored: str) -> str:
        if self.rag_source_dir is None:
            return stored
        src_dir = str(self.rag_source_dir)
        if stored.startswith(src_dir):
            return stored
        normalized = stored.replace("\\", "/")
        for anchor in ("pdfs/", "corpus/pdfs/", "source/"):
            if anchor in normalized:
                rel = normalized.split(anchor, 1)[-1]
                return str(self.rag_source_dir / rel).replace("\\", "/")
        return stored

    def search_dense_candidates(
        self,
        query_vector: list[float],
        *,
        limit: int,
        source_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        raw = self.backend.search_by_embedding(
            query_vector,
            limit=limit,
            source_paths=source_paths,
        )
        if self.rag_source_dir is not None:
            for item in raw:
                item["chunk"]["source_path"] = self._normalize_source_path(item["chunk"]["source_path"])
        return raw

    def upsert_document(self, source_path: str, chunks: list[Any], vectors: list[list[float]]) -> None:
        self.backend.upsert_document(source_path, chunks, vectors)

    def get_indexed_source_paths(self) -> set[str]:
        return {self._normalize_source_path(path) for path in self.backend.get_indexed_source_paths()}

    def get_index_stats(self) -> dict[str, int]:
        return self.backend.get_index_stats()

    def get_document_chunk_counts(self) -> list[dict[str, Any]]:
        counts = self.backend.get_document_chunk_counts()
        aggregated: dict[str, int] = {}
        for item in counts:
            normalized = self._normalize_source_path(str(item.get("source_path") or ""))
            aggregated[normalized] = aggregated.get(normalized, 0) + int(item.get("chunk_count") or 0)
        return [
            {"source_path": source_path, "chunk_count": chunk_count}
            for source_path, chunk_count in sorted(aggregated.items(), key=lambda item: item[0])
        ]

    def list_chunks(self) -> list[dict[str, Any]]:
        items = self.backend.list_chunks()
        for item in items:
            item["source_path"] = self._normalize_source_path(str(item.get("source_path") or ""))
        return items

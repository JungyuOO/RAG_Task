from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.rag.types import Chunk


logger = logging.getLogger("rag.index")


class VectorIndex:
    """청크와 벡터를 PostgreSQL에 저장하고 전체 로드하는 벡터 인덱스.

    외부 벡터 DB(pgvector 등) 없이 PostgreSQL을 스토리지로만 사용하며,
    검색 시 전체 벡터를 메모리에 로드한 뒤 HybridRetriever가 점수를 계산하는 구조이다.
    """

    def __init__(self, dsn: str) -> None:
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
        target_versions: list[str] | None = None,
        doc_type: str | None = None,
        document_group_preference: str | None = None,
    ) -> tuple[str, list[object]]:
        conditions: list[str] = []
        params: list[object] = []
        apply_version_filter = bool(target_versions)

        if document_group_preference in {"official_ocp", "mixed"}:
            apply_version_filter = False
        if doc_type == "official":
            apply_version_filter = False

        if source_paths:
            conditions.append("source_path = ANY(%s)")
            params.append(source_paths)

        if apply_version_filter:
            conditions.append("(metadata_json::jsonb ->> 'version_tag') = ANY(%s)")
            params.append(target_versions)

        if doc_type and doc_type != "auto":
            if doc_type == "operation_manual":
                conditions.append("(metadata_json::jsonb ->> 'doc_type') = %s")
                params.append("operation_manual")
            elif doc_type == "official":
                conditions.append("COALESCE(metadata_json::jsonb ->> 'doc_type', '') <> %s")
                params.append("operation_manual")

        if document_group_preference and document_group_preference not in {"auto", "mixed"}:
            if document_group_preference == "customer_generated":
                conditions.append(
                    "((metadata_json::jsonb ->> 'document_group') = %s OR (metadata_json::jsonb ->> 'doc_type') = %s)"
                )
                params.extend(["customer_generated", "operation_manual"])
            elif document_group_preference == "official_ocp":
                conditions.append(
                    "((metadata_json::jsonb ->> 'document_group') = %s OR COALESCE(metadata_json::jsonb ->> 'doc_type', '') <> %s)"
                )
                params.extend(["official_ocp", "operation_manual"])

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

    def close(self) -> None:
        """커넥션 풀을 닫는다."""
        self._pool.closeall()

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
                        page_number INTEGER,
                        metadata_json TEXT NOT NULL,
                        embedding vector,
                        vector_json TEXT NOT NULL,
                        FOREIGN KEY (source_path) REFERENCES documents(source_path)
                    )
                    """
                )
                cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector")
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path)
                    """
                )
                cursor.execute(
                    """
                    UPDATE chunks
                    SET embedding = vector_json::vector
                    WHERE embedding IS NULL
                      AND vector_json IS NOT NULL
                      AND vector_json <> ''
                      AND vector_json ~ '^\\s*\\['
                    """
                )
                if cursor.rowcount and cursor.rowcount > 0:
                    logger.info("[VectorIndex] backfilled embedding column rows=%d", cursor.rowcount)

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
                            page_number, metadata_json, embedding, vector_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.source_path,
                            chunk.text,
                            json.dumps(chunk.tokens, ensure_ascii=False),
                            chunk.page_number or chunk.metadata.get("page_start"),
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            self._vector_literal(vector),
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
                            page_number, metadata_json, embedding, vector_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.source_path,
                            chunk.text,
                            json.dumps(chunk.tokens, ensure_ascii=False),
                            chunk.page_number or chunk.metadata.get("page_start"),
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            self._vector_literal(vector),
                            json.dumps(vector, ensure_ascii=False),
                        ),
                    )

    def get_indexed_source_paths(self) -> set[str]:
        """DB에 이미 인덱싱된 모든 source_path를 반환한다."""
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT source_path FROM documents")
                rows = cursor.fetchall()
        return {row[0] for row in rows}

    def delete_document(self, source_path: str) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks WHERE source_path = %s", (source_path,))
                cursor.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))

    def load(
        self,
        *,
        source_paths: list[str] | None = None,
        target_versions: list[str] | None = None,
        doc_type: str | None = None,
        document_group_preference: str | None = None,
    ) -> list[dict]:
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                where_clause, params = self._build_filter_clause(
                    source_paths=source_paths,
                    target_versions=target_versions,
                    doc_type=doc_type,
                    document_group_preference=document_group_preference,
                )
                cursor.execute(
                    f"""
                    SELECT chunk_id, doc_id, source_path, text, tokens_json,
                           page_number, metadata_json, vector_json
                    FROM chunks
                    {where_clause}
                    """,
                    params,
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

    def list_chunks(
        self,
        *,
        source_paths: list[str],
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not source_paths:
            return [], 0

        where_clause, params = self._build_filter_clause(source_paths=source_paths)
        count_params = list(params)
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) AS total FROM chunks {where_clause}",
                    count_params,
                )
                total_row = cursor.fetchone() or {"total": 0}
                cursor.execute(
                    f"""
                    SELECT chunk_id, doc_id, source_path, text, tokens_json,
                           page_number, metadata_json
                    FROM chunks
                    {where_clause}
                    ORDER BY COALESCE(page_number, 0), chunk_id
                    OFFSET %s
                    LIMIT %s
                    """,
                    [*params, max(int(offset), 0), max(int(limit), 1)],
                )
                rows = cursor.fetchall()

        items = [
            {
                "chunk": {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "source_path": row["source_path"],
                    "text": row["text"],
                    "tokens": json.loads(row["tokens_json"]),
                    "page_number": row["page_number"],
                    "metadata": json.loads(row["metadata_json"]),
                }
            }
            for row in rows
        ]
        return items, int(total_row["total"] or 0)

    def get_chunk(self, *, source_paths: list[str], chunk_id: str) -> dict[str, Any] | None:
        if not source_paths or not chunk_id:
            return None

        where_clause, params = self._build_filter_clause(source_paths=source_paths)
        if where_clause:
            where_clause = where_clause + " AND chunk_id = %s"
        else:
            where_clause = "WHERE chunk_id = %s"
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, doc_id, source_path, text, tokens_json,
                           page_number, metadata_json
                    FROM chunks
                    {where_clause}
                    LIMIT 1
                    """,
                    [*params, chunk_id],
                )
                row = cursor.fetchone()

        if not row:
            return None
        return {
            "chunk": {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "source_path": row["source_path"],
                "text": row["text"],
                "tokens": json.loads(row["tokens_json"]),
                "page_number": row["page_number"],
                "metadata": json.loads(row["metadata_json"]),
            }
        }

    def search_by_embedding(
        self,
        query_vector: list[float],
        *,
        limit: int,
        source_paths: list[str] | None = None,
        target_versions: list[str] | None = None,
        doc_type: str | None = None,
        document_group_preference: str | None = None,
    ) -> list[dict[str, Any]]:
        where_clause, params = self._build_filter_clause(
            source_paths=source_paths,
            target_versions=target_versions,
            doc_type=doc_type,
            document_group_preference=document_group_preference,
        )
        if where_clause:
            where_clause = where_clause + " AND embedding IS NOT NULL"
        else:
            where_clause = "WHERE embedding IS NOT NULL"
        vector_literal = self._vector_literal(query_vector)
        sql_params: list[object] = [vector_literal, *params, vector_literal, max(int(limit), 1)]
        with self._connection() as connection:
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        chunk_id, doc_id, source_path, text, tokens_json,
                        page_number, metadata_json, vector_json,
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
                    "page_number": row["page_number"],
                    "metadata": json.loads(row["metadata_json"]),
                },
                "vector": json.loads(row["vector_json"]),
                "distance": float(row["distance"]),
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
        doc_type_by_path: dict[str, str] = {}
        document_group_by_path: dict[str, str] = {}
        metadata_by_path: dict[str, dict[str, str]] = {}
        for row in loader_rows:
            metadata = json.loads(row["metadata_json"])
            loader = metadata.get("loader")
            if loader:
                loaders_by_path.setdefault(row["source_path"], set()).add(loader)
            if row["source_path"] not in doc_type_by_path:
                doc_type_by_path[row["source_path"]] = str(metadata.get("doc_type") or "official")
            if row["source_path"] not in document_group_by_path:
                document_group_by_path[row["source_path"]] = str(
                    metadata.get("document_group")
                    or ("customer_generated" if metadata.get("doc_type") == "operation_manual" else "official_ocp")
                )
            if row["source_path"] not in metadata_by_path:
                metadata_by_path[row["source_path"]] = {
                    "source_url": str(metadata.get("source_url") or ""),
                    "viewer_path": str(metadata.get("viewer_path") or ""),
                    "locale": str(metadata.get("locale") or ""),
                    "version_tag": str(metadata.get("version_tag") or metadata.get("version") or ""),
                }

        return [
            {
                "file_name": row["file_name"],
                "source_path": row["source_path"],
                "extension": row["extension"],
                "indexed_pages": row["indexed_pages"] or 0,
                "indexed_chunks": row["indexed_chunks"] or 0,
                "loaders": sorted(loaders_by_path.get(row["source_path"], set())),
                "doc_type": doc_type_by_path.get(row["source_path"], "official"),
                "document_group": document_group_by_path.get(row["source_path"], "official_ocp"),
                "source_url": metadata_by_path.get(row["source_path"], {}).get("source_url", ""),
                "viewer_path": metadata_by_path.get(row["source_path"], {}).get("viewer_path", ""),
                "locale": metadata_by_path.get(row["source_path"], {}).get("locale", ""),
                "version_tag": metadata_by_path.get(row["source_path"], {}).get("version_tag", ""),
            }
            for row in rows
        ]

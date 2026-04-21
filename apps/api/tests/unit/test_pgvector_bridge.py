from __future__ import annotations

import unittest
from types import SimpleNamespace

from apps.api.schemas.ingestion_parser import ChunkRecord
from apps.api.rag.indexing.index_writer_bridge import LegacyPgvectorIndexWriter
from apps.api.rag.retrieval.pgvector_bridge import PgvectorRetrievalBridge


class _FakeEmbedder:
    def encode(self, text: str, *, is_query: bool = False) -> list[float]:
        del is_query
        return [0.1, 0.2, float(len(text))]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] for index, _text in enumerate(texts)]


class _FakeIndexRepository:
    def __init__(self) -> None:
        self.last_upsert: tuple[str, list[object], list[list[float]]] | None = None

    def search_dense_candidates(self, _query_vector: list[float], *, limit: int, source_paths: list[str] | None = None):
        del source_paths
        return [
            {
                "chunk": {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "source_path": "C:/repo/data/doc1.md",
                    "text": "route yaml example",
                    "page_number": 3,
                    "metadata": {
                        "section_title": "Route",
                        "section_path": ["OpenShift", "Networking"],
                    },
                },
                "distance": 0.12,
            }
        ][:limit]

    def upsert_document(self, source_path: str, chunks: list[object], vectors: list[list[float]]) -> None:
        self.last_upsert = (source_path, chunks, vectors)


class _FakeRuntime:
    def __init__(self) -> None:
        self.deps = SimpleNamespace(
            embedder=_FakeEmbedder(),
            index_repository=_FakeIndexRepository(),
        )

    def get(self):
        return self.deps


class PgvectorBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_pgvector_bridge_uses_runtime_without_container(self) -> None:
        bridge = PgvectorRetrievalBridge(runtime=_FakeRuntime())  # type: ignore[arg-type]

        result = await bridge.answer(message="route yaml")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.lane, "doc_pgvector")
        self.assertEqual(result.mode, "pgvector_dense")
        self.assertEqual(result.sources[0].metadata["retrieval_backend"], "pgvector_bridge")


class LegacyPgvectorIndexWriterTests(unittest.TestCase):
    def test_index_writer_uses_runtime_without_container(self) -> None:
        runtime = _FakeRuntime()
        writer = LegacyPgvectorIndexWriter(runtime=runtime)  # type: ignore[arg-type]
        chunks = [
            ChunkRecord(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_path="C:/repo/data/doc1.md",
                text="route yaml example",
                retrieval_text="Route yaml example",
                display_text="Route yaml example",
                chunk_order=0,
                token_count=3,
                metadata={"section_title": "Route"},
            )
        ]

        wrote = writer.upsert_chunks("C:/repo/data/doc1.md", chunks)

        self.assertTrue(wrote)
        upsert = runtime.deps.index_repository.last_upsert
        self.assertIsNotNone(upsert)
        assert upsert is not None
        _source_path, legacy_chunks, vectors = upsert
        self.assertEqual(vectors, [[1.0]])
        self.assertEqual(legacy_chunks[0].tokens, ["route", "yaml", "example"])


if __name__ == "__main__":
    unittest.main()




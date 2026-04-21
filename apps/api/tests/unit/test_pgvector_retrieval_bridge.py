from __future__ import annotations

import unittest

from apps.api.rag.retrieval.pgvector_bridge import PgvectorRetrievalBridge


class _FakeEmbedder:
    def encode(self, text: str, *, is_query: bool = False):
        del is_query
        return [0.1, 0.2]


class _FakeIndexRepository:
    def search_dense_candidates(self, query_vector, *, limit, source_paths=None):
        del query_vector, limit, source_paths
        return [
            {
                "distance": 0.12,
                "chunk": {
                    "chunk_id": "chunk-1",
                    "source_path": "data/corpus/pdfs/manual.md",
                    "text": "Routes expose services externally.",
                    "page_number": 1,
                    "metadata": {
                        "section_title": "Route Overview",
                        "section_path": ["Route Overview"],
                        "html_anchor": "route-overview",
                    },
                },
            }
        ]


class _FakeFilteredIndexRepository:
    def search_dense_candidates(self, query_vector, *, limit, source_paths=None):
        del query_vector, limit, source_paths
        return [
            {
                "distance": 0.12,
                "chunk": {
                    "chunk_id": "chunk-1",
                    "source_path": "C:/legacy/path/manual.md",
                    "text": "Routes expose services externally.",
                    "page_number": 1,
                    "metadata": {
                        "section_title": "Route Overview",
                        "section_path": ["Route Overview"],
                        "html_anchor": "route-overview",
                    },
                },
            },
            {
                "distance": 0.15,
                "chunk": {
                    "chunk_id": "chunk-2",
                    "source_path": "C:/legacy/path/other.md",
                    "text": "Unrelated content.",
                    "page_number": 1,
                    "metadata": {
                        "section_title": "Other",
                        "section_path": ["Other"],
                        "html_anchor": "other",
                    },
                },
            },
        ]


class _FakeRuntime:
    def get(self):
        return type(
            "Deps",
            (),
            {
                "embedder": _FakeEmbedder(),
                "index_repository": _FakeIndexRepository(),
            },
        )()


class _FakeFilteredRuntime:
    def get(self):
        return type(
            "Deps",
            (),
            {
                "embedder": _FakeEmbedder(),
                "index_repository": _FakeFilteredIndexRepository(),
            },
        )()


class PgvectorRetrievalBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_pgvector_bridge_returns_doc_response(self) -> None:
        bridge = PgvectorRetrievalBridge(runtime=_FakeRuntime())  # type: ignore[arg-type]
        response = await bridge.answer(message="route 설명")

        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.lane, "doc_pgvector")
        self.assertEqual(response.mode, "pgvector_dense")
        self.assertTrue(response.preview_ready)
        self.assertGreater(response.sources[0].score, 0)
        self.assertEqual(response.sources[0].provenance, ["doc_pgvector"])
        self.assertTrue(response.sources[0].relative_source_path)
        self.assertIn("Route Overview", response.answer)
        self.assertEqual(response.sources[0].metadata["preview_text"], "Routes expose services externally.")

    async def test_pgvector_bridge_post_filters_by_basename(self) -> None:
        bridge = PgvectorRetrievalBridge(runtime=_FakeFilteredRuntime())  # type: ignore[arg-type]
        response = await bridge.answer(
            message="route 설명",
            allowed_source_paths=["/app/data/corpus/pdfs/official/en/manual.md"],
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(len(response.sources), 1)
        self.assertTrue(str(response.sources[0].source_path).endswith("manual.md"))


if __name__ == "__main__":
    unittest.main()



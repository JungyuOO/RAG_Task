from __future__ import annotations

import unittest
from pathlib import Path

from app.storage.vector_store import IndexRepository


class _BackendStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def search_by_embedding(self, query_vector, **kwargs):
        self.calls.append(("search_by_embedding", {"query_vector": query_vector, **kwargs}))
        return [
            {
                "chunk": {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "source_path": r"C:\workspace\pdfs\ocp-4.21\guide.pdf",
                    "text": "sample",
                    "tokens": ["sample"],
                    "page_number": 1,
                    "metadata": {},
                },
                "vector": [0.1, 0.2, 0.3],
                "distance": 0.0,
            }
        ]

    def list_chunks(self, **kwargs):
        self.calls.append(("list_chunks", kwargs))
        return (
            [
                {
                    "chunk": {
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "source_path": "/app/data/corpus/pdfs/ocp-4.21/guide.pdf",
                        "text": "sample",
                        "tokens": ["sample"],
                        "page_number": 7,
                        "metadata": {"section_title": "Install"},
                    }
                }
            ],
            1,
        )

    def get_chunk(self, **kwargs):
        self.calls.append(("get_chunk", kwargs))
        return {
            "chunk": {
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "source_path": "/app/data/corpus/pdfs/ocp-4.21/guide.pdf",
                "text": "sample",
                "tokens": ["sample"],
                "page_number": 7,
                "metadata": {"section_title": "Install"},
            }
        }


class VectorStorePgvectorWrapperTests(unittest.TestCase):
    def test_search_dense_candidates_normalizes_source_path(self) -> None:
        backend = _BackendStub()
        repository = IndexRepository(backend, rag_source_dir=Path("/app/data/corpus/pdfs"))

        rows = repository.search_dense_candidates(
            [0.1, 0.2, 0.3],
            limit=10,
            target_versions=["4.21"],
            doc_type="official",
            document_group_preference="official_ocp",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk"]["source_path"], "/app/data/corpus/pdfs/ocp-4.21/guide.pdf")
        self.assertEqual(backend.calls[0][0], "search_by_embedding")
        self.assertEqual(backend.calls[0][1]["limit"], 10)
        self.assertEqual(backend.calls[0][1]["target_versions"], ["4.21"])

    def test_list_chunks_uses_source_path_candidates_and_normalizes_rows(self) -> None:
        backend = _BackendStub()
        repository = IndexRepository(backend, rag_source_dir=Path("/app/data/corpus/pdfs"))

        rows, total = repository.list_chunks(r"C:\workspace\pdfs\ocp-4.21\guide.pdf", page=2, page_size=20)

        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["chunk"]["source_path"], "/app/data/corpus/pdfs/ocp-4.21/guide.pdf")
        self.assertEqual(backend.calls[-1][0], "list_chunks")
        self.assertEqual(backend.calls[-1][1]["offset"], 20)
        self.assertEqual(backend.calls[-1][1]["limit"], 20)
        self.assertIn("/app/data/corpus/pdfs/ocp-4.21/guide.pdf", backend.calls[-1][1]["source_paths"])

    def test_get_chunk_uses_source_path_candidates(self) -> None:
        backend = _BackendStub()
        repository = IndexRepository(backend, rag_source_dir=Path("/app/data/corpus/pdfs"))

        row = repository.get_chunk(r"C:\workspace\pdfs\ocp-4.21\guide.pdf", "chunk-1")

        self.assertIsNotNone(row)
        self.assertEqual(row["chunk"]["source_path"], "/app/data/corpus/pdfs/ocp-4.21/guide.pdf")
        self.assertEqual(backend.calls[-1][0], "get_chunk")
        self.assertEqual(backend.calls[-1][1]["chunk_id"], "chunk-1")
        self.assertIn("/app/data/corpus/pdfs/ocp-4.21/guide.pdf", backend.calls[-1][1]["source_paths"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
import shutil

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_library import router as library_router
from app.dependencies import get_container


class _IndexRepositoryStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def list_chunks(self, source_path: str, *, page: int, page_size: int):
        self.calls.append(("list_chunks", {"source_path": source_path, "page": page, "page_size": page_size}))
        return (
            [
                {
                    "chunk": {
                        "chunk_id": "chunk-1",
                        "text": "normalized chunk",
                        "tokens": ["normalized", "chunk"],
                        "page_number": 7,
                        "metadata": {
                            "display_text": "normalized chunk",
                            "block_types": "paragraph",
                            "section_title": "Install",
                            "html_anchor": "page-7",
                            "primary_block_anchor": "page-7-block-1",
                        },
                    }
                }
            ],
            1,
        )

    def list_all_chunks(self, source_path: str, *, strict: bool = False):
        self.calls.append(("list_all_chunks", {"source_path": source_path, "strict": strict}))
        return [
            {
                "chunk": {
                    "chunk_id": "chunk-1",
                    "text": "normalized chunk",
                    "tokens": ["normalized", "chunk"],
                    "page_number": 7,
                    "metadata": {
                        "display_text": "normalized chunk",
                        "block_types": "paragraph",
                        "section_title": "Install",
                        "html_anchor": "page-7",
                        "primary_block_anchor": "page-7-block-1",
                    },
                }
            }
        ]

    def get_chunk(self, source_path: str, chunk_id: str, *, strict: bool = False):
        self.calls.append(("get_chunk", {"source_path": source_path, "chunk_id": chunk_id, "strict": strict}))
        return {
            "chunk": {
                "chunk_id": chunk_id,
                "text": "normalized chunk",
                "tokens": ["normalized", "chunk"],
                "page_number": 7,
                "source_path": source_path,
                "metadata": {"section_title": "Install"},
            }
        }


class LibraryChunkRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_library_chunk_routes")
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "ocp-4.20").mkdir(parents=True, exist_ok=True)
        (self.root / "ocp-4.20" / "guide.pdf").write_bytes(b"%PDF-1.4\n")
        self.index_repository = _IndexRepositoryStub()

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        app.include_router(library_router)
        container = SimpleNamespace(
            settings=SimpleNamespace(rag_source_dir=self.root, rag_extract_dir=self.root / "extract"),
            pipeline=SimpleNamespace(index_repository=self.index_repository),
            indexing_service=SimpleNamespace(list_library_documents=lambda: {"indexed_documents": []}),
            task_repository=SimpleNamespace(get_task=lambda _task_id: None),
        )
        app.dependency_overrides[get_container] = lambda: container
        return app

    def test_list_chunks_uses_repository_pagination(self) -> None:
        client = TestClient(self._build_app())

        response = client.get("/api/library/guide.pdf/chunks?page=1&page_size=10&source_path=/custom/source/guide.pdf")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["chunks"][0]["chunk_id"], "chunk-1")
        self.assertEqual(payload["source_path"], "/custom/source/guide.pdf")
        self.assertEqual(self.index_repository.calls[0][0], "list_all_chunks")
        self.assertEqual(self.index_repository.calls[0][1]["source_path"], "/custom/source/guide.pdf")
        self.assertTrue(self.index_repository.calls[0][1]["strict"])

    def test_list_chunks_prefers_source_path_query_when_provided(self) -> None:
        client = TestClient(self._build_app())

        response = client.get("/api/library/guide.pdf/chunks?page=1&page_size=10&source_path=/custom/source/guide.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.index_repository.calls[-1][0], "list_all_chunks")
        self.assertEqual(self.index_repository.calls[-1][1]["source_path"], "/custom/source/guide.pdf")
        self.assertTrue(self.index_repository.calls[-1][1]["strict"])

    def test_get_chunk_detail_uses_repository_lookup(self) -> None:
        client = TestClient(self._build_app())

        response = client.get("/api/library/guide.pdf/chunks/chunk-1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["chunk_id"], "chunk-1")
        self.assertEqual(payload["page_number"], 7)
        self.assertEqual(self.index_repository.calls[-1][0], "get_chunk")
        self.assertEqual(self.index_repository.calls[-1][1]["chunk_id"], "chunk-1")


if __name__ == "__main__":
    unittest.main()

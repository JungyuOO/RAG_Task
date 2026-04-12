from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes_library import router as library_router
from app.dependencies import get_container


class LibraryPreviewRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_library_preview")
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "ocp-html-single-4.20-en").mkdir(parents=True, exist_ok=True)
        (self.root / "extract").mkdir(parents=True, exist_ok=True)
        self.md_path = self.root / "ocp-html-single-4.20-en" / "advanced_networking.md"
        self.md_path.write_text("## Page 1\n# Title\n\nSome paragraph.\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        app.include_router(library_router)
        container = SimpleNamespace(
            settings=SimpleNamespace(rag_source_dir=self.root, rag_extract_dir=self.root / "extract", pdf_render_dpi=150),
            pipeline=SimpleNamespace(index_repository=None),
            indexing_service=SimpleNamespace(list_library_documents=lambda: {"indexed_documents": []}),
            task_repository=SimpleNamespace(get_task=lambda _task_id: None),
        )
        app.dependency_overrides[get_container] = lambda: container
        return app

    def test_preview_html_renders_markdown_source_path(self) -> None:
        client = TestClient(self._build_app())

        response = client.get("/api/library/preview-html", params={"file_name": "advanced_networking.md", "source_path": str(self.md_path)})

        self.assertEqual(response.status_code, 200)
        self.assertIn("<h1>Title</h1>", response.text)
        self.assertIn("Some paragraph.", response.text)
        self.assertIn('id="page-1"', response.text)

    def test_preview_html_accepts_windows_style_source_path(self) -> None:
        client = TestClient(self._build_app())
        windows_like = str(self.md_path).replace("/", "\\")

        response = client.get("/api/library/preview-html", params={"file_name": "advanced_networking.md", "source_path": windows_like})

        self.assertEqual(response.status_code, 200)
        self.assertIn("<h1>Title</h1>", response.text)


if __name__ == "__main__":
    unittest.main()

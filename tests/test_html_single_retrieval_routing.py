from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.rag.retrieval_service import RetrievalService


def _item(source_path: str, *, document_group: str = "official_ocp", doc_type: str = "official") -> dict:
    return {
        "chunk": {
            "chunk_id": source_path,
            "source_path": source_path,
            "text": "sample",
            "metadata": {
                "document_group": document_group,
                "doc_type": doc_type,
            },
        },
        "rerank_score": 0.5,
        "final_retrieval_score": 0.5,
    }


class HtmlSingleRetrievalRoutingTests(unittest.TestCase):
    def test_official_queries_prefer_html_single_sources(self) -> None:
        service = RetrievalService(SimpleNamespace())
        items = [
            _item("/app/data/corpus/pdfs/ocp-4.20/legacy.pdf"),
            _item("/app/data/corpus/pdfs/ocp-html-single-4.20-en/advanced_networking.md"),
        ]

        filtered = service.filter_index_items(items, allowed_source_paths=None, doc_type="official", document_group_preference="official_ocp")

        self.assertEqual(len(filtered), 1)
        self.assertTrue(filtered[0]["chunk"]["source_path"].endswith("advanced_networking.md"))

    def test_mixed_queries_keep_customer_docs_but_swap_official_side_to_html_single(self) -> None:
        service = RetrievalService(SimpleNamespace())
        items = [
            _item("/app/data/corpus/pdfs/ocp-4.20/legacy.pdf"),
            _item("/app/data/corpus/pdfs/ocp-html-single-4.20-en/advanced_networking.md"),
            _item("/app/data/corpus/pdfs/generated/customer-guide.pdf", document_group="customer_generated", doc_type="operation_manual"),
        ]

        filtered = service.filter_index_items(items, allowed_source_paths=None, doc_type=None, document_group_preference="mixed")

        source_paths = [item["chunk"]["source_path"] for item in filtered]
        self.assertIn("/app/data/corpus/pdfs/ocp-html-single-4.20-en/advanced_networking.md", source_paths)
        self.assertIn("/app/data/corpus/pdfs/generated/customer-guide.pdf", source_paths)
        self.assertNotIn("/app/data/corpus/pdfs/ocp-4.20/legacy.pdf", source_paths)


if __name__ == "__main__":
    unittest.main()

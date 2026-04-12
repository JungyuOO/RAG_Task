from __future__ import annotations

import unittest

from app.rag.answer import AnswerGenerator


class _RetrievalServiceStub:
    @staticmethod
    def build_grounded_preview_pages(preferred_preview_source, grounded_pages):
        return grounded_pages

    @staticmethod
    def aggregate_source_grounding(grounded_pages):
        return grounded_pages

    @staticmethod
    def build_context_items_payload(context_items):
        return context_items


class AnswerPreviewPayloadTests(unittest.TestCase):
    def test_build_answer_aligned_preview_pages_includes_html_and_block_anchor(self) -> None:
        generator = AnswerGenerator(retrieval_service=_RetrievalServiceStub())
        answer_citations = [
            {
                "source_path": "/docs/sample.pdf",
                "file_name": "sample.pdf",
                "page_number": 3,
                "score": 0.9,
                "chunk_id": "chunk-1",
                "origin": "answer_text",
            }
        ]
        context_items = [
            {
                "chunk": {
                    "chunk_id": "chunk-1",
                    "source_path": "/docs/sample.pdf",
                    "metadata": {
                        "html_anchor": "page-3",
                        "primary_block_anchor": "page-3-block-2",
                    },
                }
            }
        ]
        source, preview_pages = generator.build_answer_aligned_preview_pages(
            answer_citations,
            context_items,
            preferred_preview_source="/docs/sample.pdf",
            grounded_pages=[],
        )
        self.assertEqual(source, "/docs/sample.pdf")
        self.assertEqual(preview_pages[0]["html_anchor"], "page-3")
        self.assertEqual(preview_pages[0]["block_anchor"], "page-3-block-2")
        self.assertEqual(answer_citations[0].get("chunk_id"), "chunk-1")

    def test_public_context_payload_caps_items_to_three(self) -> None:
        generator = AnswerGenerator(retrieval_service=_RetrievalServiceStub())

        payload = generator.public_context_payload(
            {
                "items": [
                    {"chunk_id": "c1"},
                    {"chunk_id": "c2"},
                    {"chunk_id": "c3"},
                    {"chunk_id": "c4"},
                ]
            }
        )

        self.assertEqual(len(payload["items"]), 3)
        self.assertEqual([item["chunk_id"] for item in payload["items"]], ["c1", "c2", "c3"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.rag.answer import AnswerGenerator


class AnswerFormatRetrievalTextTests(unittest.TestCase):
    def test_extractive_text_prefers_retrieval_text(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "chunk": {
                    "source_path": "/docs/sample.pdf",
                    "page_number": 1,
                    "metadata": {
                        "is_toc": False,
                        "retrieval_text": "Pod lifecycle verify DNS verify network before installation.",
                    },
                    "text": "## Page 1\n- loader: `pdf_text`\n- chars: 120\nPod lifecycle verify DNS verify network before installation.",
                }
            }
        ]
        answer = generator.build_extractive_text_answer(context_items)
        self.assertIsNotNone(answer)
        self.assertIn("Pod lifecycle verify DNS verify network before installation.", answer)
        self.assertNotIn("loader", answer.casefold())
        self.assertNotIn("chars", answer.casefold())

    def test_extractive_compare_prefers_retrieval_text(self) -> None:
        generator = AnswerGenerator(retrieval_service=None)
        context_items = [
            {
                "chunk": {
                    "source_path": "/docs/official.pdf",
                    "page_number": 1,
                    "metadata": {
                        "is_toc": False,
                        "document_group": "official_ocp",
                        "retrieval_text": "Official document explains installation requirements and prerequisites.",
                    },
                    "text": "## Page 1\nOfficial raw text",
                }
            },
            {
                "chunk": {
                    "source_path": "/docs/customer.pdf",
                    "page_number": 2,
                    "metadata": {
                        "is_toc": False,
                        "document_group": "customer_generated",
                        "retrieval_text": "Customer guide explains operational preparation and validation steps.",
                    },
                    "text": "## Page 2\nCustomer raw text",
                }
            },
        ]
        answer = generator.build_extractive_compare_answer(context_items)
        self.assertIsNotNone(answer)
        self.assertIn("Official document explains installation requirements", answer)
        self.assertIn("Customer guide explains operational preparation", answer)


if __name__ == "__main__":
    unittest.main()

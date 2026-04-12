from __future__ import annotations

import unittest

from app.rag.chunking import MarkdownBlock
from app.rag.chunking_markdown_support import StructuredMarkdownChunkerSupport
from app.rag.utils import normalize_markdown_display_text, normalize_retrieval_text


class _ChunkerStub(StructuredMarkdownChunkerSupport):
    chunk_size = 900


class RetrievalTextNormalizationTests(unittest.TestCase):
    def test_normalize_retrieval_text_removes_markdown_table_noise(self) -> None:
        text = """
        | Name | Value |
        | ---- | ----- |
        | Pod | demo |

        ## Heading

        - verify DNS
        - verify network
        """
        normalized = normalize_retrieval_text(text)
        self.assertNotIn("|", normalized)
        self.assertNotIn("----", normalized)
        self.assertIn("Name Value Pod demo Heading verify DNS verify network", normalized)

    def test_normalize_retrieval_text_removes_page_and_loader_noise(self) -> None:
        text = """
        # guide.pdf

        ## Page 7
        - loader: `pdf_text`
        - chars: 120
        - source_path: `/docs/guide.pdf`

        > quoted note
        ---
        Actual content starts here.
        """
        normalized = normalize_retrieval_text(text)
        self.assertNotIn("loader", normalized.casefold())
        self.assertNotIn("chars", normalized.casefold())
        self.assertNotIn("page 7", normalized.casefold())
        self.assertNotIn("---", normalized)
        self.assertIn("guide.pdf", normalized)
        self.assertIn("quoted note", normalized)
        self.assertIn("Actual content starts here", normalized)

    def test_chunk_tokens_use_retrieval_text(self) -> None:
        chunker = _ChunkerStub()
        block = MarkdownBlock(
            text="| Name | Value |\n| ---- | ----- |\n| Pod | demo |",
            page_start=1,
            page_end=1,
            kind="table",
        )
        chunk = chunker._build_chunk("doc-1", "/docs/sample.pdf", [block], 0)
        self.assertEqual(chunk.metadata["retrieval_text"], "table Name Value Pod demo")
        self.assertIn("name", chunk.tokens)
        self.assertIn("demo", chunk.tokens)
        self.assertNotIn("----", chunk.tokens)
        self.assertEqual(chunk.text, "| Name | Value |\n| ---- | ----- |\n| Pod | demo |")
        self.assertEqual(chunk.metadata["display_text"], "| Name | Value |\n| ---- | ----- |\n| Pod | demo |")

    def test_block_aware_retrieval_text_preserves_code_and_list_signals(self) -> None:
        chunker = _ChunkerStub()
        blocks = [
            MarkdownBlock(text="# Pod lifecycle", page_start=1, page_end=1, kind="heading"),
            MarkdownBlock(text="- create manifest\n- apply manifest", page_start=1, page_end=1, kind="list"),
            MarkdownBlock(text="```bash\noc get pod\n```", page_start=1, page_end=1, kind="code"),
        ]
        retrieval_text = chunker._build_retrieval_text(blocks)
        self.assertIn("Pod lifecycle", retrieval_text)
        self.assertIn("steps create manifest apply manifest", retrieval_text)
        self.assertIn("code example oc get pod", retrieval_text)

    def test_normalize_markdown_display_text_cleans_noise_but_preserves_structure(self) -> None:
        text = """
        ## Page 3
        - loader: pdf_text

        # 제목

        | Name | Value |
        | ---- | ----- |
        | Pod | demo |

        ```bash
        oc get pod
        ```
        """
        display = normalize_markdown_display_text(text)
        self.assertNotIn("loader", display.casefold())
        self.assertNotIn("page 3", display.casefold())
        self.assertIn("# 제목", display)
        self.assertIn("| Name | Value |", display)
        self.assertIn("```bash", display)
        self.assertIn("oc get pod", display)

    def test_build_display_text_removes_low_signal_lines(self) -> None:
        chunker = _ChunkerStub()
        blocks = [
            MarkdownBlock(text="Last Updated: 2026-03-18", page_start=1, page_end=1, kind="paragraph"),
            MarkdownBlock(text="........ ........ ........", page_start=1, page_end=1, kind="paragraph"),
            MarkdownBlock(text="47", page_start=1, page_end=1, kind="paragraph"),
            MarkdownBlock(text="Actual content starts here.", page_start=1, page_end=1, kind="paragraph"),
        ]

        display = chunker._build_display_text(blocks)

        self.assertNotIn("Last Updated", display)
        self.assertNotIn("47", display)
        self.assertIn("Actual content starts here.", display)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.rag.answer_citation import AnswerCitationMixin


class _AnswerCitationStub(AnswerCitationMixin):
    pass


class AnswerCitationTests(unittest.TestCase):
    def test_extract_answer_citations_parses_source_tag_for_markdown(self) -> None:
        stub = _AnswerCitationStub()

        citations = stub.extract_answer_citations(
            "문서에 나온 예시입니다.\n\n```bash\noc get pods\n``` [source:cli_tools.md:p56:L1-999]"
        )

        self.assertEqual(citations, [("cli_tools.md", 56, 56)])

    def test_append_citation_entry_keeps_anchor_metadata(self) -> None:
        stub = _AnswerCitationStub()
        payload = []
        seen = set()

        stub.append_citation_entry(
            payload,
            seen,
            source_path="/docs/cli_tools.md",
            page_number=56,
            score=0.9,
            chunk_id="chunk-1",
            origin="answer_text",
            html_anchor="page-56",
            block_anchor="page-56-block-2",
        )

        self.assertEqual(payload[0]["html_anchor"], "page-56")
        self.assertEqual(payload[0]["block_anchor"], "page-56-block-2")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from app.rag.answer_inline_citation import InlineCitationMixin


class _InlineCitationStub(InlineCitationMixin):
    pass


class AnswerInlineCitationTests(unittest.TestCase):
    def test_collapse_single_citation_answer_keeps_one_tag_at_end(self) -> None:
        stub = _InlineCitationStub()
        answer = (
            "? ?? ?????. [source:file.pdf:p2:L1-999]\n"
            "? ?? ?????. [source:file.pdf:p2:L1-999]"
        )

        collapsed = stub.collapse_single_citation_answer(answer)

        self.assertEqual(collapsed.count("[source:file.pdf:p2:L1-999]"), 1)
        self.assertTrue(collapsed.endswith("[source:file.pdf:p2:L1-999]"))

    def test_collapse_single_citation_answer_preserves_multiple_sources(self) -> None:
        stub = _InlineCitationStub()
        answer = (
            "?? A [source:file-a.pdf:p2:L1-999]\n"
            "?? B [source:file-b.pdf:p3:L1-999]"
        )

        collapsed = stub.collapse_single_citation_answer(answer)

        self.assertEqual(collapsed, answer)

    def test_collapse_single_citation_answer_moves_tag_below_code_fence(self) -> None:
        stub = _InlineCitationStub()
        answer = "```bash\noc get pods\n``` [source:file.pdf:p2:L1-999]"

        collapsed = stub.collapse_single_citation_answer(answer)

        self.assertIn("```\n\n[source:file.pdf:p2:L1-999]", collapsed)
        self.assertNotIn("``` [source:file.pdf:p2:L1-999]", collapsed)


if __name__ == "__main__":
    unittest.main()

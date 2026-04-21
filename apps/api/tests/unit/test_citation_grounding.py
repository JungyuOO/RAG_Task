from __future__ import annotations

import unittest

from apps.api.schemas.copilot_chat import CopilotChatSourceItem
from apps.api.rag.generation.citation_grounding import CitationGroundingValidator


class CitationGroundingValidatorTests(unittest.TestCase):
    def test_invalid_citations_are_removed(self) -> None:
        validator = CitationGroundingValidator()
        sources = [
            CopilotChatSourceItem(source_type="doc", label="a"),
        ]

        result = validator.validate("설명입니다[1][2]", sources, enforce_alignment=False)

        self.assertEqual(result, "설명입니다[1]")

    def test_single_discriminative_overlap_is_enough(self) -> None:
        validator = CitationGroundingValidator()
        sources = [
            CopilotChatSourceItem(
                source_type="doc",
                label="auth",
                metadata={
                    "section_title": "OAuth token duration",
                    "preview_text": "Configure the internal OAuth server token duration",
                    "synthesis_text": "Configure the internal OAuth server token duration",
                },
            ),
        ]

        result = validator.validate(
            "OAuth 토큰 유효 기간은 내부 OAuth 서버 설정에서 조정합니다[1]",
            sources,
            enforce_alignment=True,
        )

        self.assertIn("[1]", result)


if __name__ == "__main__":
    unittest.main()


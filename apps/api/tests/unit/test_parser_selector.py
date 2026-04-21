from __future__ import annotations

import unittest

from apps.api.schemas.ingestion_parser import SourceDescriptor, SourceType
from apps.api.rag.indexing.parsers import build_default_parser_selector


class ParserSelectorTests(unittest.TestCase):
    def test_parser_selector_returns_matching_parser(self) -> None:
        selector = build_default_parser_selector()
        parser = selector.select(
            SourceDescriptor(
                source_type=SourceType.GENERATED_MANUAL,
                source_path="guide.md",
                file_name="guide.md",
            )
        )
        self.assertEqual(parser.parser_name, "generated_manual")


if __name__ == "__main__":
    unittest.main()




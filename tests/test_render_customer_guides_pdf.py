from __future__ import annotations

import unittest

from scripts.render_customer_guides_pdf import markdown_to_blocks


class RenderCustomerGuidesPdfTests(unittest.TestCase):
    def test_markdown_to_blocks_ignores_front_matter(self) -> None:
        markdown = """---
product: OCP
version: 4.20
doc_type: operation_manual
---

# Guide Title

Actual paragraph
"""
        blocks = markdown_to_blocks(markdown, "demo-guide")
        rendered = "".join(blocks)

        self.assertNotIn("product: OCP", rendered)
        self.assertNotIn("version: 4.20", rendered)
        self.assertIn("<h1>Guide Title</h1>", rendered)
        self.assertIn("<p>Actual paragraph</p>", rendered)

    def test_markdown_to_blocks_groups_list_items(self) -> None:
        markdown = """
- first
- second
1. third
2. fourth
"""
        blocks = markdown_to_blocks(markdown, "demo-guide")
        rendered = "".join(blocks)

        self.assertIn("<ul><li>first</li><li>second</li></ul>", rendered)
        self.assertIn("<ol><li>third</li><li>fourth</li></ol>", rendered)
        self.assertEqual(rendered.count("<ul>"), 1)
        self.assertEqual(rendered.count("<ol>"), 1)

    def test_markdown_to_blocks_renders_markdown_table(self) -> None:
        markdown = """
| Name | Value |
| --- | --- |
| Pod | demo |
| Service | api |
"""
        blocks = markdown_to_blocks(markdown, "demo-guide")
        rendered = "".join(blocks)

        self.assertIn("<table>", rendered)
        self.assertIn("<th>Name</th>", rendered)
        self.assertIn("<td>Pod</td>", rendered)
        self.assertIn("<td>api</td>", rendered)


if __name__ == "__main__":
    unittest.main()

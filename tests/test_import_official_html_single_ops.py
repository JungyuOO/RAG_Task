from __future__ import annotations

import unittest

from scripts.import_official_html_single_ops import OPS_HTML_SINGLE_SLUGS, build_html_single_url


class ImportOfficialHtmlSingleOpsTests(unittest.TestCase):
    def test_build_html_single_url(self) -> None:
        url = build_html_single_url(version="4.20", locale="en", slug="advanced_networking")
        self.assertEqual(
            url,
            "https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html-single/advanced_networking/index",
        )

    def test_ops_slug_list_contains_expected_operational_topics(self) -> None:
        self.assertIn("advanced_networking", OPS_HTML_SINGLE_SLUGS)
        self.assertIn("storage", OPS_HTML_SINGLE_SLUGS)
        self.assertIn("nodes", OPS_HTML_SINGLE_SLUGS)
        self.assertNotIn("overview", OPS_HTML_SINGLE_SLUGS)


if __name__ == "__main__":
    unittest.main()

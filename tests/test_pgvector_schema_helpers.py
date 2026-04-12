from __future__ import annotations

import unittest

from app.rag.index import VectorIndex


class PgvectorSchemaHelperTests(unittest.TestCase):
    def test_vector_literal_formats_pgvector_input(self) -> None:
        literal = VectorIndex._vector_literal([1.0, 2.5, -3.25])
        self.assertEqual(literal, "[1,2.5,-3.25]")

    def test_build_filter_clause_for_customer_generated(self) -> None:
        where_clause, params = VectorIndex._build_filter_clause(
            source_paths=["/docs/a.pdf"],
            target_versions=["4.21"],
            doc_type="operation_manual",
            document_group_preference="customer_generated",
        )
        self.assertIn("source_path = ANY(%s)", where_clause)
        self.assertIn("(metadata_json::jsonb ->> 'version_tag') = ANY(%s)", where_clause)
        self.assertIn("(metadata_json::jsonb ->> 'doc_type') = %s", where_clause)
        self.assertIn("(metadata_json::jsonb ->> 'document_group') = %s", where_clause)
        self.assertEqual(params[0], ["/docs/a.pdf"])
        self.assertEqual(params[1], ["4.21"])
        self.assertEqual(params[2], "operation_manual")
        self.assertEqual(params[3], "customer_generated")
        self.assertEqual(params[4], "operation_manual")

    def test_build_filter_clause_empty_when_no_filters(self) -> None:
        where_clause, params = VectorIndex._build_filter_clause()
        self.assertEqual(where_clause, "")
        self.assertEqual(params, [])

    def test_build_filter_clause_skips_version_for_official_html_single_queries(self) -> None:
        where_clause, params = VectorIndex._build_filter_clause(
            target_versions=["4.20"],
            document_group_preference="official_ocp",
        )
        self.assertNotIn("version_tag", where_clause)
        self.assertIn("(metadata_json::jsonb ->> 'document_group') = %s", where_clause)
        self.assertEqual(params, ["official_ocp", "operation_manual"])


if __name__ == "__main__":
    unittest.main()

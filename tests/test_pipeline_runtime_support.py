from __future__ import annotations

import unittest

from app.rag.pipeline_runtime_support import PipelineRuntimeMixin


class PipelineRuntimeSupportTests(unittest.TestCase):
    def test_build_retrieval_diagnostics_extracts_phase3_fields(self) -> None:
        state = {
            "index_load_strategy": "pgvector_dense_supplemented",
            "source_filter_strategy": "keyword_scoped",
            "index_items_loaded": 120,
            "index_items_filtered": 48,
            "candidate_pool_size": 15,
            "top_score": 0.42,
            "use_retrieved_context": True,
        }
        diagnostics = PipelineRuntimeMixin._build_retrieval_diagnostics(state)
        self.assertEqual(diagnostics["index_load_strategy"], "pgvector_dense_supplemented")
        self.assertEqual(diagnostics["source_filter_strategy"], "keyword_scoped")
        self.assertEqual(diagnostics["index_items_loaded"], 120)
        self.assertEqual(diagnostics["index_items_filtered"], 48)
        self.assertEqual(diagnostics["candidate_pool_size"], 15)
        self.assertEqual(diagnostics["top_score"], 0.42)
        self.assertTrue(diagnostics["use_retrieved_context"])


if __name__ == "__main__":
    unittest.main()

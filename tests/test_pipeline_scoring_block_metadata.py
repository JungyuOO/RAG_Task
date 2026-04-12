from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.rag.pipeline_scoring import PipelineRetrievalMixin


class _ScoringStub(PipelineRetrievalMixin):
    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            retrieval_final_ce_weight=0.65,
            retrieval_final_metadata_weight=0.25,
            retrieval_final_anchor_weight=0.10,
            retrieval_explain_focus_boost=0.08,
        )

    def _resolve_requested_resource_kinds(self, query_interpretation: dict | None) -> set[str]:
        query_interpretation = query_interpretation or {}
        return {str(value).casefold() for value in query_interpretation.get("resources", []) if value}

    def _heading_overlap_score(self, _user_message: str, _metadata: dict) -> float:
        return 0.0

    def _compute_focus_multiplier(self, _target_resource: str, _all_requested_resources: set[str], _lowered_text: str, _section_focus_text: str, current_multiplier: float) -> float:
        return current_multiplier


class PipelineScoringBlockMetadataTests(unittest.TestCase):
    def test_table_metadata_boosts_table_queries(self) -> None:
        scoring = _ScoringStub()
        item = {
            "chunk": {
                "text": "table data",
                "source_path": "/docs/sample.pdf",
                "metadata": {
                    "block_types": "table",
                    "table_headers": ["name", "value"],
                    "table_row_count": 2,
                    "table_column_count": 2,
                },
            },
            "dense_score": 0.1,
            "sparse_score": 0.1,
            "rerank_score": 0.2,
            "ce_score": 0.2,
        }
        query_interpretation = {
            "normalized_keywords": ["name", "value"],
            "resources": [],
            "actions": [],
            "format_constraints": ["table"],
            "response_shape": "table",
            "intent": "table",
            "document_group_preference": "auto",
        }
        scored = scoring._metadata_aware_score("name value table", query_interpretation, item)
        self.assertGreater(scored["format_match_score"], 1.0)
        self.assertGreater(scored["shape_match_score"], 0.75)

    def test_code_block_attributes_boost_code_queries(self) -> None:
        scoring = _ScoringStub()
        item = {
            "chunk": {
                "text": "code example",
                "source_path": "/docs/sample.pdf",
                "metadata": {
                    "block_types": "code",
                    "block_code_languages": ["bash"],
                    "block_code_resource_kinds": ["pod"],
                    "has_cli_block": True,
                },
            },
            "dense_score": 0.1,
            "sparse_score": 0.1,
            "rerank_score": 0.2,
            "ce_score": 0.2,
        }
        query_interpretation = {
            "normalized_keywords": ["pod"],
            "resources": ["pod"],
            "actions": [],
            "format_constraints": ["cli"],
            "response_shape": "code",
            "intent": "cli_example",
            "document_group_preference": "auto",
        }
        scored = scoring._metadata_aware_score("pod cli example", query_interpretation, item)
        self.assertGreater(scored["resource_match_score"], 0.9)
        self.assertGreater(scored["format_match_score"], 0.5)
        self.assertGreater(scored["shape_match_score"], 0.9)

    def test_list_item_count_boosts_procedure_queries(self) -> None:
        scoring = _ScoringStub()
        item = {
            "chunk": {
                "text": "procedure list",
                "source_path": "/docs/sample.pdf",
                "metadata": {
                    "block_types": "list",
                    "list_item_count": 4,
                    "is_procedure": True,
                },
            },
            "dense_score": 0.1,
            "sparse_score": 0.1,
            "rerank_score": 0.2,
            "ce_score": 0.2,
        }
        query_interpretation = {
            "normalized_keywords": ["step"],
            "resources": [],
            "actions": [],
            "format_constraints": [],
            "response_shape": "procedure",
            "intent": "procedure_followup",
            "document_group_preference": "auto",
        }
        scored = scoring._metadata_aware_score("step by step", query_interpretation, item)
        self.assertGreater(scored["action_match_score"], 0.5)
        self.assertGreater(scored["shape_match_score"], 0.3)


if __name__ == "__main__":
    unittest.main()

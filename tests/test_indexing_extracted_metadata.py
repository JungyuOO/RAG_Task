from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from app.rag.indexing import IndexingService
from app.rag.types import Chunk


def _build_service() -> IndexingService:
    settings = SimpleNamespace(
        rag_source_dir=Path("/docs"),
        rag_extract_dir=Path("/extract"),
        embedding_batch_size=16,
        embedding_parallel_workers=1,
        embedding_batch_char_limit=0,
        embedding_backend="tei",
    )
    return IndexingService(
        settings=settings,
        ingestor=None,
        structured_chunker=None,
        embedder=None,
        index_repository=None,
        embedding_cache_repository=None,
    )


class IndexingExtractedMetadataTests(unittest.TestCase):
    def test_apply_extracted_structure_metadata_enriches_chunk(self) -> None:
        service = _build_service()
        chunk = Chunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            source_path="/docs/guide.pdf",
            text="chunk text",
            tokens=["chunk", "text"],
            page_number=1,
            metadata={"page_start": 1, "page_end": 1, "block_types": "paragraph"},
        )
        extracted_metadata = {
            "pages": [
                {
                    "page_number": 1,
                    "html_anchor": "page-1",
                    "section_title": "Install prerequisites",
                    "section_path": "Install prerequisites",
                    "blocks": [
                        {
                            "block_id": "page-1-block-1",
                            "block_type": "heading",
                            "html_anchor": "page-1-block-1",
                            "section_title": "Install prerequisites",
                            "section_path": "Install prerequisites",
                        },
                        {
                            "block_id": "page-1-block-2",
                            "block_type": "list",
                            "html_anchor": "page-1-block-2",
                        },
                    ],
                }
            ]
        }

        service._apply_extracted_structure_metadata([chunk], extracted_metadata)

        self.assertEqual(chunk.metadata["html_anchor"], "page-1")
        self.assertEqual(chunk.metadata["primary_block_anchor"], "page-1-block-1")
        self.assertEqual(chunk.metadata["section_title"], "Install prerequisites")
        self.assertEqual(chunk.metadata["section_path"], "Install prerequisites")
        self.assertIn("heading", str(chunk.metadata["block_types"]))
        self.assertIn("list", str(chunk.metadata["block_types"]))
        self.assertEqual(chunk.metadata["block_ids"], ["page-1-block-1", "page-1-block-2"])
        self.assertNotIn("block_code_languages", chunk.metadata)
        self.assertNotIn("list_item_count", chunk.metadata)

    def test_apply_extracted_structure_metadata_aggregates_block_attributes(self) -> None:
        service = _build_service()
        chunk = Chunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            source_path="/docs/guide.pdf",
            text="chunk text",
            tokens=["chunk", "text"],
            page_number=1,
            metadata={"page_start": 1, "page_end": 1, "block_types": "paragraph"},
        )
        extracted_metadata = {
            "pages": [
                {
                    "page_number": 1,
                    "html_anchor": "page-1",
                    "section_title": "Install prerequisites",
                    "section_path": "Install prerequisites",
                    "blocks": [
                        {
                            "block_id": "page-1-block-1",
                            "block_type": "table",
                            "html_anchor": "page-1-block-1",
                            "attributes": {"headers": ["Name", "Value"], "row_count": 2, "column_count": 2},
                        },
                        {
                            "block_id": "page-1-block-2",
                            "block_type": "code",
                            "html_anchor": "page-1-block-2",
                            "attributes": {"language": "yaml", "resource_kind": "Pod", "has_cli": True},
                        },
                        {
                            "block_id": "page-1-block-3",
                            "block_type": "list",
                            "html_anchor": "page-1-block-3",
                            "attributes": {"item_count": 3},
                        },
                    ],
                }
            ]
        }

        service._apply_extracted_structure_metadata([chunk], extracted_metadata)

        self.assertEqual(chunk.metadata["table_headers"], ["Name", "Value"])
        self.assertEqual(chunk.metadata["table_row_count"], 2)
        self.assertEqual(chunk.metadata["table_column_count"], 2)
        self.assertEqual(chunk.metadata["block_code_languages"], ["yaml"])
        self.assertEqual(chunk.metadata["block_code_resource_kinds"], ["Pod"])
        self.assertTrue(chunk.metadata["has_cli_block"])
        self.assertEqual(chunk.metadata["list_item_count"], 3)

    def test_filter_empty_chunks_drops_chunks_with_no_display_or_retrieval_text(self) -> None:
        service = _build_service()
        empty_chunk = Chunk(
            chunk_id="chunk-empty",
            doc_id="doc-1",
            source_path="/docs/guide.pdf",
            text="",
            tokens=[],
            page_number=1,
            metadata={"retrieval_text": "   "},
        )
        valid_chunk = Chunk(
            chunk_id="chunk-valid",
            doc_id="doc-1",
            source_path="/docs/guide.pdf",
            text="Visible text",
            tokens=["visible", "text"],
            page_number=1,
            metadata={"retrieval_text": "visible text"},
        )

        filtered = service._filter_empty_chunks([empty_chunk, valid_chunk])

        self.assertEqual([chunk.chunk_id for chunk in filtered], ["chunk-valid"])

    def test_filter_low_signal_chunks_drops_toc_like_chunk(self) -> None:
        service = _build_service()
        toc_chunk = Chunk(
            chunk_id="chunk-toc",
            doc_id="doc-1",
            source_path="/docs/guide.pdf",
            text=(
                "1.5. VERIFYING NETWORK CONNECTIVITY FOR AN ENDPOINT\n"
                "1.4.1. Connection log fields\n"
                "11\n"
                ". . . . . . . . . . . . . . .\n"
                "CHAPTER 2. CHANGING THE MTU FOR THE CLUSTER NETWORK\n"
                "16"
            ),
            tokens=[],
            page_number=5,
            metadata={"retrieval_text": ""},
        )
        valid_chunk = Chunk(
            chunk_id="chunk-valid",
            doc_id="doc-1",
            source_path="/docs/guide.pdf",
            text="Configure the ingress controller load balancer for external access.",
            tokens=["configure", "ingress"],
            page_number=8,
            metadata={"retrieval_text": "Configure the ingress controller load balancer for external access."},
        )

        filtered = service._filter_low_signal_chunks([toc_chunk, valid_chunk])

        self.assertEqual([chunk.chunk_id for chunk in filtered], ["chunk-valid"])

    def test_index_single_file_reads_raw_markdown_for_md_sources(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = root / "advanced_networking.md"
            raw_markdown = "## Page 1\n# Title\n\n## Page 2\nMeaningful body\n"
            source_path.write_text(raw_markdown, encoding="utf-8")

            class _Ingestor:
                def ingest_paths(self, paths, progress_callback=None):  # noqa: ANN001
                    del progress_callback
                    self.paths = paths
                    return [
                        SimpleNamespace(
                            source_path=str(source_path),
                            text="normalized single line text",
                            metadata={"loader": "text"},
                        )
                    ], []

            class _Chunker:
                def __init__(self):
                    self.markdown_text = None

                def split(self, documents, markdown_text=None):  # noqa: ANN001
                    self.markdown_text = markdown_text
                    return []

            service = IndexingService(
                settings=SimpleNamespace(
                    rag_source_dir=root,
                    rag_extract_dir=root / "extract",
                    embedding_batch_size=16,
                    embedding_parallel_workers=1,
                    embedding_batch_char_limit=0,
                    embedding_backend="tei",
                ),
                ingestor=_Ingestor(),
                structured_chunker=_Chunker(),
                embedder=None,
                index_repository=None,
                embedding_cache_repository=None,
            )

            result = service.index_single_file(source_path)

            self.assertTrue(result["skipped"])
            self.assertEqual(service.structured_chunker.markdown_text, raw_markdown)


if __name__ == "__main__":
    unittest.main()

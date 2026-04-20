from __future__ import annotations

import inspect
import logging
from pathlib import Path

from apps.api.persistence.artifacts import NormalizedArtifactRepository
from apps.api.api.schemas.indexing import NewIndexRequest, NewIndexResponse, SourceDescriptor
from apps.api.rag.indexing.chunking import BlockPreservingChunker
from apps.api.rag.indexing.enrich import MetadataEnricher
from apps.api.rag.indexing.index_writer_bridge import LegacyPgvectorIndexWriter
from apps.api.rag.indexing.normalize import BlockNormalizer
from apps.api.rag.indexing.parsers import ParserSelector, build_default_parser_selector

logger = logging.getLogger("rag.new_index")


class NewIndexingService:
    def __init__(
        self,
        *,
        parser_selector: ParserSelector | None = None,
        block_normalizer: BlockNormalizer | None = None,
        metadata_enricher: MetadataEnricher | None = None,
        block_chunker: BlockPreservingChunker | None = None,
        artifact_repository: NormalizedArtifactRepository | None = None,
        index_writer: LegacyPgvectorIndexWriter | None = None,
    ) -> None:
        self.parser_selector = parser_selector or build_default_parser_selector()
        self.block_normalizer = block_normalizer or BlockNormalizer()
        self.metadata_enricher = metadata_enricher or MetadataEnricher()
        self.block_chunker = block_chunker or BlockPreservingChunker()
        self.artifact_repository = artifact_repository or NormalizedArtifactRepository()
        self.index_writer = index_writer or LegacyPgvectorIndexWriter()

    def index_source(self, request: NewIndexRequest, *, stage_callback=None, status_callback=None, current_file: str | None = None) -> NewIndexResponse:
        source = SourceDescriptor(**request.model_dump())
        source_path = Path(source.source_path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {source.source_path}")

        self._emit(stage_callback, f"[index] parser start source={source.source_path}")
        self._emit_status(status_callback, step="PARSER", message="parser started", current_file=current_file or source.source_path)
        parser = self.parser_selector.select(source)
        bundle = parser.parse(source)
        self._emit(stage_callback, f"[index] parser done blocks={len(bundle.blocks)} source={source.source_path}")
        self._emit_status(status_callback, step="PARSER_DONE", message=f"parser done blocks={len(bundle.blocks)}", current_file=current_file or source.source_path)

        self._emit(stage_callback, f"[index] normalize start source={source.source_path}")
        self._emit_status(status_callback, step="NORMALIZING", message="normalize started", current_file=current_file or source.source_path)
        normalized = self.block_normalizer.normalize_bundle(bundle)
        self._emit(stage_callback, f"[index] normalize done blocks={len(normalized)} source={source.source_path}")

        self._emit(stage_callback, f"[index] enrich start source={source.source_path}")
        self._emit_status(status_callback, step="ENRICHING", message="metadata enrich started", current_file=current_file or source.source_path)
        enriched = self.metadata_enricher.enrich_blocks(bundle, normalized)
        self._emit(stage_callback, f"[index] enrich done blocks={len(enriched)} source={source.source_path}")

        self._emit(stage_callback, f"[index] chunk start source={source.source_path}")
        self._emit_status(status_callback, step="CHUNKING", message="chunking started", current_file=current_file or source.source_path)
        chunks = self.block_chunker.chunk_document(bundle, enriched)
        self._emit(stage_callback, f"[index] chunk done chunks={len(chunks)} source={source.source_path}")
        self._emit_status(status_callback, step="CHUNKING", message=f"chunking done chunks={len(chunks)}", current_file=current_file or source.source_path)

        self._emit(stage_callback, f"[index] artifact save start source={source.source_path}")
        parsed_artifact_path = self.artifact_repository.save_parsed(source.source_path, bundle.model_dump())
        normalized_artifact_path = self.artifact_repository.save_normalized(
            source.source_path,
            [item.model_dump() for item in enriched],
        )
        chunk_artifact_path = self.artifact_repository.save_chunks(
            source.source_path,
            [item.model_dump() for item in chunks],
        )
        self._emit(stage_callback, f"[index] artifact save done source={source.source_path}")

        self._emit(stage_callback, f"[index] embed/upsert start chunks={len(chunks)} source={source.source_path}")
        self._emit_status(status_callback, step="EMBEDDING", message=f"embedding chunks={len(chunks)}", current_file=current_file or source.source_path)
        writer_kwargs = {}
        writer_signature = inspect.signature(self.index_writer.upsert_chunks)
        if "stage_callback" in writer_signature.parameters:
            writer_kwargs["stage_callback"] = stage_callback
        if "status_callback" in writer_signature.parameters:
            writer_kwargs["status_callback"] = status_callback
        indexed = self.index_writer.upsert_chunks(source.source_path, chunks, **writer_kwargs)
        self._emit(stage_callback, f"[index] embed/upsert done indexed={indexed} source={source.source_path}")
        self._emit_status(status_callback, step="UPSERTING", message=f"upsert done indexed={indexed}", current_file=current_file or source.source_path)

        return NewIndexResponse(
            source_path=source.source_path,
            source_type=source.source_type,
            parsed_blocks=len(bundle.blocks),
            normalized_blocks=len(enriched),
            chunks=len(chunks),
            indexed=indexed,
            parsed_artifact_path=parsed_artifact_path,
            normalized_artifact_path=normalized_artifact_path,
            chunk_artifact_path=chunk_artifact_path,
        )

    @staticmethod
    def _emit(stage_callback, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        logger.info(text)
        if stage_callback:
            stage_callback(text)

    @staticmethod
    def _emit_status(status_callback, *, step: str, message: str, current_file: str) -> None:
        if status_callback:
            status_callback(step=step, message=message, current_file=current_file)

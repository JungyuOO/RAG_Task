from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from app.config import get_settings
from app.rag.bge_embeddings import BGEOllamaEmbedder
from app.rag.cache import JsonFileCache
from app.rag.chunking import TextChunker
from app.rag.chunking_markdown import StructuredMarkdownChunker
from app.rag.index import VectorIndex
from app.rag.indexing import IndexingService
from app.rag.ingestion_pdf import DocumentIngestor
from app.storage import CacheRepository, IndexRepository


def _build_indexing_service(settings) -> IndexingService:
    """RagPipeline 전체(reranker 포함)를 생성하지 않고 IndexingService만 조립한다."""
    embedder = BGEOllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        timeout=settings.ollama_timeout,
    )
    settings.vector_dim = embedder.dim
    index = VectorIndex(settings.db_dsn)
    embedding_cache = JsonFileCache(
        settings.rag_cache_dir / "embeddings",
        max_entries=settings.cache_max_entries,
        ttl_hours=settings.cache_ttl_hours,
    )
    return IndexingService(
        settings=settings,
        ingestor=DocumentIngestor(settings),
        chunker=TextChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap),
        structured_chunker=StructuredMarkdownChunker(
            chunk_size=settings.structured_chunk_size,
            overlap=settings.structured_chunk_overlap,
        ),
        embedder=embedder,
        index_repository=IndexRepository(index),
        embedding_cache_repository=CacheRepository(embedding_cache),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 인덱스 구축")
    parser.add_argument(
        "--version", "-v",
        metavar="VERSION",
        help="특정 버전만 인덱싱 (예: 4.16). 생략 시 전체 인덱싱.",
    )
    parser.add_argument(
        "--skip-indexed",
        action="store_true",
        help="이미 DB에 인덱싱된 파일은 건너뜀. 기존 데이터 유지하며 이어서 실행할 때 사용.",
    )
    parser.add_argument(
        "--start-from",
        metavar="FILENAME",
        help="지정한 파일명부터 시작 (예: OpenShift_Container_Platform-4.16-Networking-en-US.pdf). 그 앞 파일은 건너뜀.",
    )
    args = parser.parse_args()

    settings = get_settings()
    indexing_service = _build_indexing_service(settings)

    SUPPORTED_EXTS = {".pdf", ".md"}

    if args.version:
        version_dir = settings.rag_source_dir / f"ocp-{args.version}"
        if not version_dir.exists():
            print(f"[오류] 버전 폴더를 찾을 수 없습니다: {version_dir}")
            return
        source_files = sorted(
            path for path in version_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        )
        print(f"[버전 {args.version}] 파일 {len(source_files)}개 발견 (PDF+MD).")
    else:
        source_files = sorted(
            path for path in settings.rag_source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
        )
        print(f"[전체] 파일 {len(source_files)}개 발견 (PDF+MD).")

    if args.start_from:
        target = args.start_from.lower()
        before = len(source_files)
        # 파일명(확장자 포함)이 target과 일치하는 지점부터 자름
        start_idx = next(
            (i for i, p in enumerate(source_files) if p.name.lower() == target),
            None,
        )
        if start_idx is None:
            print(f"[오류] --start-from 파일을 목록에서 찾을 수 없습니다: {args.start_from}")
            print("발견된 파일 목록:")
            for p in source_files:
                print(f"  {p.name}")
            return
        source_files = source_files[start_idx:]
        print(f"[start-from] '{args.start_from}'부터 시작. {before - start_idx}개 건너뜀, 남은 파일: {len(source_files)}개.")

    if args.skip_indexed:
        # 기존 데이터를 보존하면서 미인덱싱 파일만 추가 — index_single_file() 사용
        indexed_paths = indexing_service.index_repository.get_indexed_source_paths()
        before = len(source_files)
        source_files = [p for p in source_files if str(p) not in indexed_paths]
        skipped = before - len(source_files)
        print(f"[skip-indexed] 이미 인덱싱된 {skipped}개 파일 건너뜀. 남은 파일: {len(source_files)}개.")

    if not source_files:
        print("인덱싱할 파일이 없습니다.")
        return

    total = len(source_files)
    for i, path in enumerate(source_files, 1):
        print(f"[{i}/{total}] {path.name} ...")
        if path.suffix.lower() == ".md":
            result = indexing_service.index_markdown_file(path)
        else:
            result = indexing_service.index_single_file(path)
        print(f"  → chunks={result.get('indexed_chunks')}, pages={result.get('indexed_pages')}, skipped={result.get('skipped')}")
    print("완료.")


if __name__ == "__main__":
    main()

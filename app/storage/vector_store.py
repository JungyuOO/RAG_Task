from __future__ import annotations

from pathlib import Path

from app.rag.index import VectorIndex
from app.rag.types import Chunk


class IndexRepository:
    """벡터 인덱스 데이터 접근 계층 — VectorIndex와 서비스 계층 사이의 추상 경계.

    인덱스 저장소 교체(예: 파일→원격 DB) 시 이 인터페이스만 재구현하면
    서비스 계층 코드 변경 없이 확장할 수 있다.

    ``load()`` 결과를 인메모리 캐싱하여 매 쿼리마다 DB 전체 스캔을 피한다.
    ``save()``, ``upsert_document()``, ``delete_document()`` 호출 시 캐시를 무효화한다.
    """

    def __init__(self, backend: VectorIndex, rag_source_dir: Path | None = None) -> None:
        self.backend = backend
        self.rag_source_dir = rag_source_dir
        self._cache: list[dict] | None = None

    def _normalize_source_path(self, stored: str) -> str:
        """로컬↔Docker 환경 간 절대경로 불일치를 상대경로 기반으로 보정한다.

        로컬 Windows에서 인덱싱한 경로(C:\\Users\\...\\pdfs\\ocp-4.15\\file.pdf)가
        Docker Linux 환경(/app/data/corpus/pdfs/ocp-4.15/file.pdf)에서도
        올바르게 조회될 수 있도록 rag_source_dir 기준 상대경로를 추출해 재구성한다.
        """
        if self.rag_source_dir is None:
            return stored

        src_dir = str(self.rag_source_dir)
        # 같은 환경이면 그대로 반환
        if stored.startswith(src_dir):
            return stored

        # 다른 환경의 절대경로 → 공통 앵커(pdfs/, corpus/pdfs/)로 상대경로 추출
        normalized = stored.replace("\\", "/")
        for anchor in ("pdfs/", "corpus/pdfs/", "source/"):
            if anchor in normalized:
                rel = normalized.split(anchor, 1)[-1]
                return str(self.rag_source_dir / rel).replace("\\", "/")
        return stored

    def _source_path_candidates(self, source_path: str) -> list[str]:
        candidates = {
            str(source_path),
            str(source_path).replace("\\", "/"),
        }
        if self.rag_source_dir is None:
            return sorted(candidates)

        normalized = str(source_path).replace("\\", "/")
        source_root = str(self.rag_source_dir).replace("\\", "/")
        relative_path = ""
        for anchor in (source_root + "/", "pdfs/", "corpus/pdfs/"):
            if anchor in normalized:
                relative_path = normalized.split(anchor, 1)[-1]
                break

        if relative_path:
            local_path = self.rag_source_dir / Path(relative_path)
            candidates.add(str(local_path))
            candidates.add(str(local_path).replace("\\", "/"))
            candidates.add(relative_path)
            candidates.add(relative_path.replace("/", "\\"))
            candidates.add(f"data/corpus/pdfs/{relative_path}".replace("\\", "/"))
            candidates.add(f"data\\corpus\\pdfs\\{relative_path.replace('/', '\\')}")
            candidates.add(f"/app/data/corpus/pdfs/{relative_path}".replace("\\", "/"))
            candidates.add(f"/data/corpus/pdfs/{relative_path}".replace("\\", "/"))

        return sorted(candidate for candidate in candidates if candidate)

    def save(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.backend.save(chunks, vectors)
        self._cache = None

    def warm_cache(self) -> int:
        return len(self.load())

    def clear_cache(self) -> None:
        self._cache = None

    def load(
        self,
        *,
        source_paths: list[str] | None = None,
        target_versions: list[str] | None = None,
        doc_type: str | None = None,
        document_group_preference: str | None = None,
    ) -> list[dict]:
        use_full_cache = not source_paths and not target_versions and not doc_type and not document_group_preference
        if use_full_cache and self._cache is not None:
            return self._cache
        raw = self.backend.load(
            source_paths=source_paths,
            target_versions=target_versions,
            doc_type=doc_type,
            document_group_preference=document_group_preference,
        )
        if self.rag_source_dir is not None:
            for item in raw:
                item["chunk"]["source_path"] = self._normalize_source_path(
                    item["chunk"]["source_path"]
                )
        if use_full_cache:
            self._cache = raw
            return self._cache
        return raw

    def search_dense_candidates(
        self,
        query_vector: list[float],
        *,
        limit: int,
        source_paths: list[str] | None = None,
        target_versions: list[str] | None = None,
        doc_type: str | None = None,
        document_group_preference: str | None = None,
    ) -> list[dict]:
        raw = self.backend.search_by_embedding(
            query_vector,
            limit=limit,
            source_paths=source_paths,
            target_versions=target_versions,
            doc_type=doc_type,
            document_group_preference=document_group_preference,
        )
        if self.rag_source_dir is not None:
            for item in raw:
                item["chunk"]["source_path"] = self._normalize_source_path(
                    item["chunk"]["source_path"]
                )
        return raw

    def list_chunks(self, source_path: str, *, page: int, page_size: int, strict: bool = False) -> tuple[list[dict], int]:
        rows, total = self.backend.list_chunks(
            source_paths=[source_path] if strict else self._source_path_candidates(source_path),
            offset=max(page - 1, 0) * page_size,
            limit=page_size,
        )
        if self.rag_source_dir is not None:
            for item in rows:
                item["chunk"]["source_path"] = self._normalize_source_path(item["chunk"]["source_path"])
        return rows, total

    def list_all_chunks(self, source_path: str, *, strict: bool = False) -> list[dict]:
        rows, _total = self.backend.list_chunks(
            source_paths=[source_path] if strict else self._source_path_candidates(source_path),
            offset=0,
            limit=100000,
        )
        if self.rag_source_dir is not None:
            for item in rows:
                item["chunk"]["source_path"] = self._normalize_source_path(item["chunk"]["source_path"])
        return rows

    def get_chunk(self, source_path: str, chunk_id: str, *, strict: bool = False) -> dict | None:
        row = self.backend.get_chunk(
            source_paths=[source_path] if strict else self._source_path_candidates(source_path),
            chunk_id=chunk_id,
        )
        if row and self.rag_source_dir is not None:
            row["chunk"]["source_path"] = self._normalize_source_path(row["chunk"]["source_path"])
        return row

    def upsert_document(self, source_path: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.backend.upsert_document(source_path, chunks, vectors)
        self._cache = None

    def delete_document(self, source_path: str) -> None:
        self.backend.delete_document(source_path)
        self._cache = None

    def list_documents(self) -> list[dict]:
        return self.backend.list_documents()
    
    def get_indexed_source_paths(self) -> set[str]:
        return self.backend.get_indexed_source_paths()

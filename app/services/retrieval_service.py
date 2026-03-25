from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from app.config import Settings


class RetrievalService:
    """검색 결과를 페이지 기반 그라운딩, 출처 집계, context 아이템 선택으로 가공하는 서비스.

    HybridRetriever가 반환한 청크 목록을 페이지 단위로 점수를 집계하고,
    근거가 강한 페이지/출처를 우선 선택하여 LLM에 전달할 context를 구성한다.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_context_items_payload(self, context_items: list[dict]) -> list[dict]:
        return [
            {
                "chunk_id": item["chunk"]["chunk_id"],
                "source_path": item["chunk"]["source_path"],
                "page_number": item["chunk"]["page_number"] or item["chunk"]["metadata"].get("page_start"),
                "page_start": item["chunk"]["metadata"].get("page_start"),
                "page_end": item["chunk"]["metadata"].get("page_end"),
                "chunking_strategy": item["chunk"]["metadata"].get("chunking_strategy"),
                "block_types": item["chunk"]["metadata"].get("block_types", ""),
                "block_count": item["chunk"]["metadata"].get("block_count"),
                "score": round(item["rerank_score"], 4),
                "rerank_score": round(item["rerank_score"], 4),
                "base_score": round(item.get("score", 0.0), 4),
                "dense_score": round(item.get("dense_score", 0.0), 4),
                "sparse_score": round(item.get("sparse_score", 0.0), 4),
                "title_score": round(item.get("title_score", 0.0), 4),
                "title_match_bonus": round(item.get("title_match_bonus", 0.0), 4),
                "compact_match_bonus": round(item.get("compact_match_bonus", 0.0), 4),
                "selection_score": round(item.get("selection_score", item["rerank_score"]), 4),
                "selection_best_page_rank": item.get("selection_best_page_rank"),
                "selection_page_overlap": item.get("selection_page_overlap", 0),
                "text_preview": item["chunk"]["text"][:240],
            }
            for item in context_items
        ]

    def aggregate_page_grounding(self, context_items: list[dict]) -> list[dict]:
        page_scores: dict[tuple[str, int], dict] = {}
        for item in context_items:
            chunk = item["chunk"]
            score = float(item["rerank_score"])
            page_start = int(chunk["metadata"].get("page_start") or chunk["page_number"] or 1)
            page_end = int(chunk["metadata"].get("page_end") or page_start)
            if page_end < page_start:
                page_end = page_start
            span = max(page_end - page_start + 1, 1)
            apportioned_score = score / span
            for page_number in range(page_start, page_end + 1):
                key = (chunk["source_path"], page_number)
                bucket = page_scores.setdefault(
                    key,
                    {
                        "source_path": chunk["source_path"],
                        "page_number": page_number,
                        "score": 0.0,
                        "match_count": 0,
                        "best_chunk_score": 0.0,
                    },
                )
                bucket["score"] += apportioned_score
                bucket["match_count"] += 1
                bucket["best_chunk_score"] = max(bucket["best_chunk_score"], score)

        grounded_pages = list(page_scores.values())
        grounded_pages.sort(
            key=lambda item: (
                item["score"],
                item["best_chunk_score"],
                item["match_count"],
                -item["page_number"],
            ),
            reverse=True,
        )
        return [
            {
                "source_path": item["source_path"],
                "page_number": item["page_number"],
                "score": round(item["score"], 4),
                "match_count": item["match_count"],
                "best_chunk_score": round(item["best_chunk_score"], 4),
            }
            for item in grounded_pages
        ]

    def aggregate_source_grounding(self, grounded_pages: list[dict]) -> list[dict]:
        source_scores: dict[str, dict] = {}
        for item in grounded_pages:
            source_path = item["source_path"]
            bucket = source_scores.setdefault(
                source_path,
                {
                    "source_path": source_path,
                    "file_name": Path(source_path).name,
                    "score": 0.0,
                    "page_hits": 0,
                    "best_page_score": 0.0,
                },
            )
            bucket["score"] += float(item["score"])
            bucket["page_hits"] += 1
            bucket["best_page_score"] = max(bucket["best_page_score"], float(item["best_chunk_score"]))

        aggregated_sources = list(source_scores.values())
        aggregated_sources.sort(
            key=lambda item: (
                item["best_page_score"],
                item["score"],
                item["page_hits"],
            ),
            reverse=True,
        )
        return [
            {
                "source_path": item["source_path"],
                "file_name": item["file_name"],
                "score": round(item["score"], 4),
                "page_hits": item["page_hits"],
                "best_page_score": round(item["best_page_score"], 4),
            }
            for item in aggregated_sources
        ]

    def select_grounded_preview_source(self, grounded_pages: list[dict]) -> str | None:
        if not grounded_pages:
            return None

        source_scores: dict[str, float] = defaultdict(float)
        source_best_page: dict[str, float] = defaultdict(float)
        source_match_counts: dict[str, int] = defaultdict(int)
        for item in grounded_pages:
            source_path = item["source_path"]
            source_scores[source_path] += float(item["score"])
            source_best_page[source_path] = max(source_best_page[source_path], float(item["score"]))
            source_match_counts[source_path] += int(item["match_count"])

        return max(
            source_scores.keys(),
            key=lambda source: (
                source_best_page[source],
                source_scores[source],
                source_match_counts[source],
            ),
        )

    def build_grounded_preview_pages(
        self,
        preferred_preview_source: str | None,
        grounded_pages: list[dict],
        limit: int = 3,
    ) -> list[dict]:
        if not preferred_preview_source:
            return []

        preview_pages = [
            {
                "source_path": item["source_path"],
                "page_number": item["page_number"],
                "score": item["score"],
            }
            for item in grounded_pages
            if item["source_path"] == preferred_preview_source
        ]
        return preview_pages[:limit]

    def chunk_page_numbers(self, item: dict) -> list[int]:
        chunk = item["chunk"]
        page_start = int(chunk["metadata"].get("page_start") or chunk["page_number"] or 1)
        page_end = int(chunk["metadata"].get("page_end") or page_start)
        if page_end < page_start:
            page_end = page_start
        return list(range(page_start, page_end + 1))

    def order_context_items_by_grounded_pages(
        self,
        context_items: list[dict],
        grounded_pages: list[dict],
    ) -> list[dict]:
        ordered_context_items = context_items.copy()
        page_rank = {
            (item["source_path"], item["page_number"]): index
            for index, item in enumerate(grounded_pages)
        }
        ordered_context_items.sort(
            key=lambda item: (
                min(
                    [
                        page_rank.get((item["chunk"]["source_path"], page_number), len(grounded_pages))
                        for page_number in self.chunk_page_numbers(item)
                    ]
                    or [len(grounded_pages)]
                ),
                -float(item["rerank_score"]),
            )
        )
        return ordered_context_items

    def select_context_items_by_grounded_pages(
        self,
        ordered_context_items: list[dict],
        grounded_pages: list[dict],
    ) -> list[dict]:
        if not ordered_context_items or not grounded_pages:
            return ordered_context_items

        selected_page_limit = max(int(self.settings.grounded_page_top_n), 1)
        selected_chunk_limit = max(int(self.settings.grounded_chunk_top_n), 1)

        # 소스 다양성 보장: 각 소스에서 best-ranked 페이지 1개를 먼저 확보하고,
        # 남은 슬롯을 점수 순으로 채운다. RBAC처럼 한 문서가 상위 페이지를 독점해도
        # SCC 등 다른 소스의 페이지(→ 청크)가 후보에서 완전히 배제되지 않도록 한다.
        source_best_pages: dict[str, dict] = {}
        for page in grounded_pages:
            source = page["source_path"]
            if source not in source_best_pages:
                source_best_pages[source] = page
        diversity_pages = list(source_best_pages.values())
        diversity_page_keys = {(p["source_path"], p["page_number"]) for p in diversity_pages}
        fill_pages = [p for p in grounded_pages if (p["source_path"], p["page_number"]) not in diversity_page_keys]
        selected_pages = (diversity_pages + fill_pages)[:selected_page_limit]

        preferred_preview_source = self.select_grounded_preview_source(grounded_pages)
        selected_page_ranks = {
            (item["source_path"], int(item["page_number"])): index
            for index, item in enumerate(selected_pages)
        }

        prioritized_items: list[dict] = []
        for item in ordered_context_items:
            page_numbers = self.chunk_page_numbers(item)
            matched_ranks = [
                selected_page_ranks[(item["chunk"]["source_path"], page_number)]
                for page_number in page_numbers
                if (item["chunk"]["source_path"], page_number) in selected_page_ranks
            ]
            if not matched_ranks:
                continue

            best_rank = min(matched_ranks)
            page_overlap = len(set(matched_ranks))
            span_penalty = max(len(page_numbers) - 1, 0) * 0.04
            source_bonus = 0.03 if item["chunk"]["source_path"] == preferred_preview_source else 0.0
            page_signal = 0.18 / (best_rank + 1)
            overlap_bonus = 0.03 * page_overlap
            selection_score = float(item["rerank_score"]) + page_signal + overlap_bonus + source_bonus - span_penalty
            prioritized_items.append(
                {
                    **item,
                    "selection_score": selection_score,
                    "selection_best_page_rank": best_rank,
                    "selection_page_overlap": page_overlap,
                }
            )

        if not prioritized_items:
            return ordered_context_items[:selected_chunk_limit]

        prioritized_items.sort(
            key=lambda item: (
                item["selection_best_page_rank"],
                -item["selection_score"],
                -float(item["rerank_score"]),
            )
        )

        # 소스 다양성 보장: 각 소스에서 best-ranked 청크 1개씩 먼저 확보한 뒤 나머지 슬롯을 점수 순으로 채운다.
        # RBAC처럼 한 문서가 상위 점수를 독점해도 SCC 등 다른 소스의 청크가 누락되지 않도록 한다.
        source_best: dict[str, dict] = {}
        for item in prioritized_items:
            source = item["chunk"]["source_path"]
            if source not in source_best:
                source_best[source] = item

        diversity_items = list(source_best.values())
        diversity_ids = {id(item) for item in diversity_items}
        fill_items = [item for item in prioritized_items if id(item) not in diversity_ids]

        merged = diversity_items + fill_items
        return merged[:selected_chunk_limit]

    def filter_index_items(
        self,
        index_items: list[dict],
        allowed_source_paths: set[str] | None,
    ) -> list[dict]:
        if not allowed_source_paths:
            return index_items

        normalized_allowed = set()
        for path in allowed_source_paths:
            normalized_allowed.add(str(Path(path)))
            normalized_allowed.add(str(Path(path).resolve()))
        return [
            item
            for item in index_items
            if str(Path(item["chunk"]["source_path"])) in normalized_allowed
            or str(Path(item["chunk"]["source_path"]).resolve()) in normalized_allowed
        ]

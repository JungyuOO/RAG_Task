from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

from app.config import Settings

logger = logging.getLogger("rag.retrieval_service")

class RetrievalService:
    """검색 결과를 페이지 기반 그라운딩, 출처 집계, context 아이템 선택으로 가공하는 서비스.

    HybridRetriever가 반환한 청크 목록을 페이지 단위로 점수를 집계하고,
    근거가 강한 페이지/출처를 우선 선택하여 LLM에 전달할 context를 구성한다.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def primary_score(item: dict) -> float:
        return float(item.get("final_retrieval_score", item.get("rerank_score", 0.0)))

    @staticmethod
    def document_group(item: dict) -> str:
        metadata = item["chunk"].get("metadata", {}) or {}
        explicit_group = str(metadata.get("document_group") or "").strip()
        if explicit_group:
            return explicit_group
        if str(metadata.get("doc_type") or "") == "operation_manual":
            return "customer_generated"
        return "official_ocp"

    def build_context_items_payload(self, context_items: list[dict]) -> list[dict]:
        return [
            {
                "chunk_id": item["chunk"]["chunk_id"],
                "source_path": item["chunk"]["source_path"],
                "page_number": item["chunk"]["page_number"] or item["chunk"]["metadata"].get("page_start"),
                "page_start": item["chunk"]["metadata"].get("page_start"),
                "page_end": item["chunk"]["metadata"].get("page_end"),
                "block_types": item["chunk"]["metadata"].get("block_types", ""),
                "block_count": item["chunk"]["metadata"].get("block_count"),
                "section_title": item["chunk"]["metadata"].get("section_title", ""),
                "section_path": item["chunk"]["metadata"].get("section_path", ""),
                "nearest_heading": item["chunk"]["metadata"].get("nearest_heading", ""),
                "parent_headings": item["chunk"]["metadata"].get("parent_headings", []),
                "code_language": item["chunk"]["metadata"].get("code_language", ""),
                "code_subtype": item["chunk"]["metadata"].get("code_subtype", ""),
                "code_signals": item["chunk"]["metadata"].get("code_signals", []),
                "score": round(self.primary_score(item), 4),
                "final_retrieval_score": round(self.primary_score(item), 4),
                "rerank_score": round(item["rerank_score"], 4),
                "ce_score": round(item.get("ce_score", item.get("rerank_score", 0.0)), 4),
                "retrieval_score": round(item.get("retrieval_score", item.get("score", 0.0)), 4),
                "base_score": round(item.get("score", 0.0), 4),
                "dense_score": round(item.get("dense_score", 0.0), 4),
                "sparse_score": round(item.get("sparse_score", 0.0), 4),
                "title_score": round(item.get("title_score", 0.0), 4),
                "title_match_bonus": round(item.get("title_match_bonus", 0.0), 4),
                "compact_match_bonus": round(item.get("compact_match_bonus", 0.0), 4),
                "selection_score": round(item.get("selection_score", self.primary_score(item)), 4),
                "selection_best_page_rank": item.get("selection_best_page_rank"),
                "selection_page_overlap": item.get("selection_page_overlap", 0),
                "text_preview": item["chunk"]["text"][:240],
            }
            for item in context_items
        ]

    def aggregate_page_grounding(self, context_items: list[dict]) -> list[dict]:
        t_total = time.perf_counter()
        page_scores: dict[tuple[str, int], dict] = {}

        for item in context_items:
            chunk = item["chunk"]
            score = self.primary_score(item)
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

        logger.info(
            "[Timing][RetrievalService.aggregate_page_grounding] total=%.3fs context_items=%d grounded_pages=%d",
            time.perf_counter() - t_total,
            len(context_items),
            len(grounded_pages),
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
                -self.primary_score(item),
            )
        )
        return ordered_context_items

    def select_context_items_by_grounded_pages(
        self,
        ordered_context_items: list[dict],
        grounded_pages: list[dict],
    ) -> list[dict]:
        t_total = time.perf_counter()

        if not ordered_context_items or not grounded_pages:
            logger.info(
                "[Timing][RetrievalService.select_context_items_by_grounded_pages] total=%.3fs short_circuit ordered=%d grounded=%d",
                time.perf_counter() - t_total,
                len(ordered_context_items),
                len(grounded_pages),
            )
            return ordered_context_items

        selected_page_limit = max(int(self.settings.grounded_page_top_n), 1)
        selected_chunk_limit = max(int(self.settings.grounded_chunk_top_n), 1)
        preferred_preview_source = self.select_grounded_preview_source(grounded_pages)

        logger.info(
            "[Timing][RetrievalService.select_context_items_by_grounded_pages] setup=%.3fs selected_page_limit=%d selected_chunk_limit=%d preferred_preview_source=%s",
            time.perf_counter() - t_total,
            selected_page_limit,
            selected_chunk_limit,
            preferred_preview_source,
        )

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
        preferred_page = next(
            (page for page in diversity_pages if page["source_path"] == preferred_preview_source),
            None,
        )
        continuity_pages: list[dict] = []
        if preferred_page is not None:
            continuity_pages = [
                page
                for page in grounded_pages
                if page["source_path"] == preferred_preview_source
                and (page["source_path"], page["page_number"]) not in diversity_page_keys
                and abs(int(page["page_number"]) - int(preferred_page["page_number"])) <= 1
            ]
            continuity_pages.sort(
                key=lambda page: (
                    abs(int(page["page_number"]) - int(preferred_page["page_number"])),
                    -float(page["score"]),
                    -float(page["best_chunk_score"]),
                )
            )
        continuity_page_keys = {(p["source_path"], p["page_number"]) for p in continuity_pages}
        fill_pages = [
            p for p in grounded_pages
            if (p["source_path"], p["page_number"]) not in diversity_page_keys
            and (p["source_path"], p["page_number"]) not in continuity_page_keys
        ]
        selected_pages: list[dict] = []
        if preferred_page is not None:
            selected_pages.append(preferred_page)
            selected_pages.extend(continuity_pages[:1])
        selected_pages.extend(
            page
            for page in diversity_pages
            if preferred_page is None
            or (page["source_path"], page["page_number"]) != (preferred_page["source_path"], preferred_page["page_number"])
        )
        selected_pages.extend(fill_pages)
        selected_pages = selected_pages[:selected_page_limit]

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
            selection_score = self.primary_score(item) + page_signal + overlap_bonus + source_bonus - span_penalty
            prioritized_items.append(
                {
                    **item,
                    "selection_score": selection_score,
                    "selection_best_page_rank": best_rank,
                    "selection_page_overlap": page_overlap,
                }
            )

        if not prioritized_items:
            logger.info(
                "[Timing][RetrievalService.select_context_items_by_grounded_pages] total=%.3fs fallback_ordered_items=%d",
                time.perf_counter() - t_total,
                min(len(ordered_context_items), selected_chunk_limit),
            )
            return ordered_context_items[:selected_chunk_limit]

        prioritized_items.sort(
            key=lambda item: (
                item["selection_best_page_rank"],
                -item["selection_score"],
                -self.primary_score(item),
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
        selected_items = merged[:selected_chunk_limit]

        logger.info(
            "[Timing][RetrievalService.select_context_items_by_grounded_pages] total=%.3fs selected_chunks=%d grounded_pages=%d",
            time.perf_counter() - t_total,
            len(selected_items),
            len(grounded_pages),
        )
        return selected_items

    def filter_index_items(
        self,
        index_items: list[dict],
        allowed_source_paths: set[str] | None,
        uploaded_source_paths: set[str] | None = None,
        doc_type: str | None = None,
        document_group_preference: str | None = None,
    ) -> list[dict]:
        items = index_items

        if uploaded_source_paths:
            normalized_uploaded = set()
            for path in uploaded_source_paths:
                normalized_uploaded.add(str(Path(path)))
                normalized_uploaded.add(str(Path(path).resolve()))

            def _is_uploaded(item: dict) -> bool:
                source_path = str(item["chunk"].get("source_path") or "")
                return (
                    str(Path(source_path)) in normalized_uploaded
                    or str(Path(source_path).resolve()) in normalized_uploaded
                )

            if document_group_preference == "mixed":
                items = [
                    item for item in items
                    if _is_uploaded(item) or self.document_group(item) == "official_ocp"
                ]
            else:
                items = [item for item in items if _is_uploaded(item)]
        elif allowed_source_paths:
            normalized_allowed = set()
            for path in allowed_source_paths:
                normalized_allowed.add(str(Path(path)))
                normalized_allowed.add(str(Path(path).resolve()))
            items = [
                item
                for item in items
                if str(Path(item["chunk"]["source_path"])) in normalized_allowed
                or str(Path(item["chunk"]["source_path"]).resolve()) in normalized_allowed
            ]

        if doc_type and doc_type != "auto":
            if doc_type == "operation_manual":
                items = [
                    item for item in items
                    if (item["chunk"].get("metadata") or {}).get("doc_type") == "operation_manual"
                ]
            elif doc_type == "official":
                items = [
                    item for item in items
                    if (item["chunk"].get("metadata") or {}).get("doc_type") != "operation_manual"
                ]

        if document_group_preference and document_group_preference not in {"auto", "mixed"}:
            if document_group_preference == "customer_generated":
                items = [
                    item for item in items
                    if (
                        (
                            (item["chunk"].get("metadata") or {}).get("document_group") == "customer_generated"
                            or (item["chunk"].get("metadata") or {}).get("doc_type") == "operation_manual"
                        )
                        and str(item["chunk"].get("source_path") or "").replace("\\", "/").lower().endswith(".pdf")
                    )
                ]
            elif document_group_preference == "official_ocp":
                items = [
                    item for item in items
                    if (item["chunk"].get("metadata") or {}).get("document_group") == "official_ocp"
                    or (item["chunk"].get("metadata") or {}).get("doc_type") != "operation_manual"
                ]
        elif document_group_preference == "mixed":
            filtered: list[dict] = []
            for item in items:
                metadata = item["chunk"].get("metadata") or {}
                group = str(metadata.get("document_group") or "")
                source_path = str(item["chunk"].get("source_path") or "").replace("\\", "/").lower()
                if group == "customer_generated" and not source_path.endswith(".pdf"):
                    continue
                filtered.append(item)
            items = filtered

        return items

    def rebalance_context_items_by_document_group(
        self,
        items: list[dict],
        document_group_preference: str | None,
        *,
        limit: int | None = None,
    ) -> list[dict]:
        if not items:
            return items
        if document_group_preference != "mixed":
            return items[:limit] if limit is not None else items

        preferred_limit = limit if limit is not None else len(items)
        grouped: dict[str, list[dict]] = {"official_ocp": [], "customer_generated": []}
        for item in items:
            group = self.document_group(item)
            if group in grouped:
                grouped[group].append(item)

        if not grouped["official_ocp"] or not grouped["customer_generated"]:
            return items[:preferred_limit]

        selected: list[dict] = [grouped["official_ocp"][0], grouped["customer_generated"][0]]
        selected_ids = {item["chunk"]["chunk_id"] for item in selected}
        for item in items:
            if len(selected) >= preferred_limit:
                break
            chunk_id = item["chunk"]["chunk_id"]
            if chunk_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(chunk_id)
        return selected[:preferred_limit]

from __future__ import annotations

import re
from collections.abc import Iterable

from apps.api.api.schemas.chat import CopilotChatResponse, CopilotChatSourceItem
from apps.api.rag.query.query_features import answer_source_budget


def merge_hybrid_sources(
    message: str,
    dense_response: CopilotChatResponse | None,
    sparse_response: CopilotChatResponse | None,
    *,
    source_key_builder,
    source_merger,
    top_k: int = 8,
    rrf_k: int = 60,
    mmr_lambda: float = 0.85,
) -> list[CopilotChatSourceItem]:
    if dense_response is None or sparse_response is None:
        return []
    if not dense_response.sources or not sparse_response.sources:
        return []

    merged_scores: dict[str, float] = {}
    merged_sources: dict[str, CopilotChatSourceItem] = {}
    merged_provenance: dict[str, set[str]] = {}

    for response in (dense_response, sparse_response):
        for rank, source in enumerate(response.sources, start=1):
            key = source_key_builder(source)
            merged_scores[key] = merged_scores.get(key, 0.0) + _rrf_score(rank, k=rrf_k)
            merged_provenance.setdefault(key, set()).update(source.provenance or [response.lane])
            if key not in merged_sources:
                merged_sources[key] = source
            else:
                merged_sources[key] = source_merger(merged_sources[key], source)

    ranked_keys = [key for key, _score in sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)]
    candidate_keys = ranked_keys[:top_k]
    if not candidate_keys:
        return []

    selected_keys = _mmr_select(
        candidate_keys,
        merged_scores,
        {key: merged_sources[key] for key in candidate_keys},
        limit=answer_source_budget(message) + 1,
        lambda_mult=mmr_lambda,
    )

    selected_sources: list[CopilotChatSourceItem] = []
    for key in selected_keys:
        source = merged_sources[key]
        selected_sources.append(
            source.model_copy(
                update={
                    "score": round(merged_scores[key], 3),
                    "provenance": sorted(merged_provenance.get(key) or []),
                    "metadata": {
                        **source.metadata,
                        "hybrid_origin_lanes": sorted(merged_provenance.get(key) or []),
                        "retrieval_backend": "doc_hybrid",
                    },
                }
            )
        )
    return selected_sources


def _rrf_score(rank: int, *, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _mmr_select(
    keys: list[str],
    scores: dict[str, float],
    sources: dict[str, CopilotChatSourceItem],
    *,
    limit: int,
    lambda_mult: float,
) -> list[str]:
    if not keys:
        return []

    selected = [keys[0]]
    remaining = keys[1:]

    while remaining and len(selected) < limit:
        best_key = None
        best_score = float("-inf")
        for key in remaining:
            diversity_penalty = max(
                _source_similarity(sources[key], sources[selected_key])
                for selected_key in selected
            )
            mmr_score = lambda_mult * scores[key] - (1.0 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_key = key
        if best_key is None:
            break
        selected.append(best_key)
        remaining.remove(best_key)
    return selected


def _source_similarity(left: CopilotChatSourceItem, right: CopilotChatSourceItem) -> float:
    left_tokens = _source_tokens(left)
    right_tokens = _source_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return overlap / max(union, 1)


def _source_tokens(source: CopilotChatSourceItem) -> set[str]:
    text = " ".join(
        [
            str(source.metadata.get("section_title") or source.label or ""),
            str(source.metadata.get("preview_text") or ""),
            str(source.metadata.get("synthesis_text") or ""),
        ]
    )
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9가-힣_-]+", text.casefold())
        if len(token) >= 2
    }

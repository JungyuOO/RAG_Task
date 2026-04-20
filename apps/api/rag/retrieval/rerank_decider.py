from __future__ import annotations


def needs_rerank(scores: list[float], *, threshold: float = 0.15) -> bool:
    if len(scores) < 3:
        return False
    top = scores[0]
    if top <= 0:
        return False
    relative_gap = (top - scores[1]) / top
    return relative_gap < threshold

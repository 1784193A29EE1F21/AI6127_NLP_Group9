"""Reciprocal Rank Fusion and cross-encoder re-ranking."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from . import config


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[int]], k: int = 60,
                           weights: Sequence[float] | None = None) -> list[tuple[int, float]]:
    """RRF (Cormack et al., 2009): score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[int, float] = defaultdict(float)
    for w, lst in zip(weights, ranked_lists):
        for rank, doc in enumerate(lst, start=1):
            fused[int(doc)] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class CrossEncoderReranker:
    """Re-scores (query, passage) pairs jointly with a cross-encoder."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 384):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> np.ndarray:
        if not passages:
            return np.array([], dtype=np.float32)
        pairs = [(query, p) for p in passages]
        return np.asarray(self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False),
                          dtype=np.float32)

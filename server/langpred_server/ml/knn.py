"""
Tier-1 predictor: k-Nearest-Neighbours on prefix features.

Given a partial trajectory, find the *k* most similar completed trajectories
(by their prefix at the same length) and report the empirical p50/p90/p99 of
their final outcomes (cost, steps, wall time).

No training step; the "model" is just the dataset of finished trajectories.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..trajectories import Trajectory
from .featurize import FeatureVector, featurize, l2_distance


@dataclass
class Neighbour:
    distance: float
    final_usd: float
    final_steps: int
    final_seconds: float
    status: str
    trace_id: str


@dataclass
class PrefixPrediction:
    n_samples: int
    confidence: float
    p50_usd: float
    p90_usd: float
    p99_usd: float
    p50_steps: float
    p90_steps: float
    p99_steps: float
    p50_seconds: float
    p90_seconds: float
    p99_seconds: float
    offrails_score: float
    explanation: str = ""


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    pos = q * (len(vs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vs[lo]
    frac = pos - lo
    return vs[lo] * (1 - frac) + vs[hi] * frac


class KNNPredictor:
    """Per-(project, trace_name) prefix-kNN predictor. Stateless except for
    the dataset it's handed; rebuild cheap, no I/O.
    """

    def __init__(
        self,
        completed: Iterable[Trajectory],
        k: int = 20,
        max_prefix: int = 200,
    ) -> None:
        self.k = k
        self.max_prefix = max_prefix
        # Cache: for each completed traj, store its FeatureVectors at every prefix
        # length up to max_prefix, plus its final outcomes.
        self._traces: list[Trajectory] = []
        self._prefixes: list[list[FeatureVector]] = []
        for t in completed:
            if not t.is_complete or t.step_count == 0:
                continue
            self._traces.append(t)
            cap = min(t.step_count, self.max_prefix)
            self._prefixes.append([featurize(t, k) for k in range(1, cap + 1)])

    @property
    def n(self) -> int:
        return len(self._traces)

    def predict(self, prefix: Trajectory) -> PrefixPrediction:
        prefix_len = max(1, prefix.step_count)
        query = featurize(prefix, prefix_len)

        # Score each completed trajectory: distance from the closest prefix of
        # matching depth (or its deepest available if shorter).
        scored: list[Neighbour] = []
        for traj, fvs in zip(self._traces, self._prefixes):
            if not fvs:
                continue
            # Use the prefix at index min(prefix_len-1, last) for fair comparison.
            idx = min(prefix_len - 1, len(fvs) - 1)
            dist = l2_distance(query, fvs[idx])
            n = Neighbour(
                distance=dist,
                final_usd=traj.total_usd,
                final_steps=traj.step_count,
                final_seconds=traj.wall_seconds,
                status=traj.status,
                trace_id=traj.trace_id,
            )
            scored.append(n)
        if not scored:
            return _zero_prediction()

        neighbours = heapq.nsmallest(self.k, scored, key=lambda x: x.distance)
        confidence = _confidence(neighbours)
        return PrefixPrediction(
            n_samples=len(neighbours),
            confidence=confidence,
            p50_usd=_quantile([n.final_usd for n in neighbours], 0.5),
            p90_usd=_quantile([n.final_usd for n in neighbours], 0.9),
            p99_usd=_quantile([n.final_usd for n in neighbours], 0.99),
            p50_steps=_quantile([n.final_steps for n in neighbours], 0.5),
            p90_steps=_quantile([n.final_steps for n in neighbours], 0.9),
            p99_steps=_quantile([n.final_steps for n in neighbours], 0.99),
            p50_seconds=_quantile([n.final_seconds for n in neighbours], 0.5),
            p90_seconds=_quantile([n.final_seconds for n in neighbours], 0.9),
            p99_seconds=_quantile([n.final_seconds for n in neighbours], 0.99),
            offrails_score=sum(1 for n in neighbours if n.status in ("error", "cancelled"))
            / len(neighbours),
            explanation=(
                f"kNN k={len(neighbours)} median final $"
                f"{_quantile([n.final_usd for n in neighbours], 0.5):.4f} "
                f"from {len(self._traces)} completed traces"
            ),
        )


def _confidence(neighbours: Sequence[Neighbour]) -> float:
    """Heuristic: tight clusters → high confidence; sparse → low."""
    if len(neighbours) < 3:
        return 0.2
    dists = [n.distance for n in neighbours]
    mean = sum(dists) / len(dists)
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    std = math.sqrt(var)
    if mean <= 0:
        return 0.9
    cv = std / mean
    # cv ~0 → confident; cv > 1 → unconfident.
    return max(0.1, min(0.95, 1.0 - cv / 2.0))


def _zero_prediction() -> PrefixPrediction:
    return PrefixPrediction(
        n_samples=0,
        confidence=0.0,
        p50_usd=0.0,
        p90_usd=0.0,
        p99_usd=0.0,
        p50_steps=0.0,
        p90_steps=0.0,
        p99_steps=0.0,
        p50_seconds=0.0,
        p90_seconds=0.0,
        p99_seconds=0.0,
        offrails_score=0.0,
        explanation="no neighbours",
    )

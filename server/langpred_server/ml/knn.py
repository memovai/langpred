"""
Tier-1 predictor: k-Nearest-Neighbours on prefix features.

The k=20 nearest finished trajectories supply not just final cost / ETA /
step count quantiles but *also* the per-neighbour next-step kind, next tool,
per-tool histograms and per-model cost breakdowns. One lookup → everything
the omnibus :class:`AgentPrediction` needs.
"""
from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..trajectories import Step, Trajectory
from .featurize import FeatureVector, featurize, l2_distance


@dataclass
class Neighbour:
    distance: float
    trajectory: Trajectory
    prefix_len: int  # the prefix length at which we matched

    # convenience pass-throughs ---------------------------------------------
    @property
    def final_usd(self) -> float:
        return self.trajectory.total_usd

    @property
    def final_steps(self) -> int:
        return self.trajectory.step_count

    @property
    def final_seconds(self) -> float:
        return self.trajectory.wall_seconds

    @property
    def status(self) -> str:
        return self.trajectory.status


@dataclass
class PrefixPrediction:
    """Output of one kNN query.

    Carries the empirical quantiles of the headline outcomes plus the
    aggregated histograms used by the omnibus predictor.
    """

    n_samples: int
    confidence: float
    # Total outcomes (kept for legacy per-kind endpoint compatibility).
    p50_usd: float
    p90_usd: float
    p99_usd: float
    p50_steps: float
    p90_steps: float
    p99_steps: float
    p50_seconds: float
    p90_seconds: float
    p99_seconds: float
    p50_tokens: float = 0.0
    p90_tokens: float = 0.0
    p50_prompt_tokens: float = 0.0
    p90_prompt_tokens: float = 0.0
    p50_completion_tokens: float = 0.0
    p90_completion_tokens: float = 0.0
    p50_llm_calls: float = 0.0
    p50_tool_calls: float = 0.0
    p50_compute_seconds: float = 0.0
    p50_io_seconds: float = 0.0
    offrails_score: float = 0.0
    # Next-step distributions, conditional on prefix length.
    next_kind_distribution: dict[str, float] = field(default_factory=dict)
    next_tool_distribution: dict[str, float] = field(default_factory=dict)
    likely_next_model: str | None = None
    expected_next_step_usd: float = 0.0
    expected_next_step_seconds: float = 0.0
    p_finish_within_one_step: float = 0.0
    # Resource histograms over neighbours' full trajectories.
    tool_call_counts_p50: dict[str, float] = field(default_factory=dict)
    tool_call_counts_p90: dict[str, float] = field(default_factory=dict)
    usd_by_model_p50: dict[str, float] = field(default_factory=dict)
    usd_by_model_p90: dict[str, float] = field(default_factory=dict)
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
    """Per-(project, trace_name) prefix-kNN predictor."""

    def __init__(
        self,
        completed: Iterable[Trajectory],
        k: int = 20,
        max_prefix: int = 200,
    ) -> None:
        self.k = k
        self.max_prefix = max_prefix
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

    # --------------------------------------------------------------- predict

    def predict(self, prefix: Trajectory) -> PrefixPrediction:
        prefix_len = max(1, prefix.step_count)
        query = featurize(prefix, prefix_len)

        scored: list[Neighbour] = []
        for traj, fvs in zip(self._traces, self._prefixes):
            if not fvs:
                continue
            idx = min(prefix_len - 1, len(fvs) - 1)
            dist = l2_distance(query, fvs[idx])
            scored.append(Neighbour(distance=dist, trajectory=traj, prefix_len=idx + 1))
        if not scored:
            return _zero_prediction()

        neighbours = heapq.nsmallest(self.k, scored, key=lambda x: x.distance)
        return self._aggregate(neighbours, prefix_len)

    # ------------------------------------------------------------ aggregate

    def _aggregate(self, neighbours: list[Neighbour], prefix_len: int) -> PrefixPrediction:
        n = len(neighbours)
        if n == 0:
            return _zero_prediction()

        confidence = _confidence(neighbours)

        # Headline totals.
        usd = [nb.final_usd for nb in neighbours]
        steps = [float(nb.final_steps) for nb in neighbours]
        seconds = [nb.final_seconds for nb in neighbours]
        tokens = [float(nb.trajectory.total_tokens) for nb in neighbours]
        prompt_tokens = [float(nb.trajectory.prompt_tokens) for nb in neighbours]
        completion_tokens = [float(nb.trajectory.completion_tokens) for nb in neighbours]
        llm_calls = [float(nb.trajectory.llm_call_count) for nb in neighbours]
        tool_calls = [float(nb.trajectory.tool_call_count) for nb in neighbours]
        compute_seconds = [nb.trajectory.compute_seconds for nb in neighbours]
        io_seconds = [nb.trajectory.io_seconds for nb in neighbours]

        # Next-step distributions: look at neighbour[prefix_len] (the step
        # *after* the matched prefix). Trajectories that already ended
        # contribute a synthetic "end" outcome.
        next_kind_counts: Counter[str] = Counter()
        next_tool_counts: Counter[str] = Counter()
        next_step_usd: list[float] = []
        next_step_seconds: list[float] = []
        next_models: Counter[str] = Counter()
        finished_within_one: int = 0

        for nb in neighbours:
            step = nb.trajectory.step_at(prefix_len)
            if step is None:
                next_kind_counts["end"] += 1
                finished_within_one += 1
                continue
            kind_label = "generation" if step.kind == "generation" else "tool_call"
            next_kind_counts[kind_label] += 1
            if kind_label == "tool_call":
                next_tool_counts[step.tool_name or step.name or "unknown"] += 1
            else:
                if step.model:
                    next_models[step.model] += 1
            next_step_usd.append(step.usd)
            next_step_seconds.append(step.latency_ms / 1000.0)
            # Is this the *last* step of the neighbour? Then this prefix ends
            # immediately after one more step.
            if prefix_len + 1 >= nb.trajectory.step_count:
                finished_within_one += 1

        next_kind_dist = {k: c / n for k, c in next_kind_counts.items()}
        next_tool_dist = {k: c / n for k, c in next_tool_counts.items()}
        likely_next_model = next_models.most_common(1)[0][0] if next_models else None
        p_finish_within_one_step = finished_within_one / n

        # Per-tool histograms over each neighbour's remaining suffix.
        tool_call_counts_p50: dict[str, float] = {}
        tool_call_counts_p90: dict[str, float] = {}
        all_tools: set[str] = set()
        per_neighbour_hist: list[dict[str, int]] = []
        for nb in neighbours:
            h: dict[str, int] = {}
            for step in nb.trajectory.steps[prefix_len:]:
                if step.is_tool:
                    name = step.tool_name or step.name or "unknown"
                    h[name] = h.get(name, 0) + 1
            per_neighbour_hist.append(h)
            all_tools.update(h)
        for tool in all_tools:
            counts = [float(h.get(tool, 0)) for h in per_neighbour_hist]
            tool_call_counts_p50[tool] = _quantile(counts, 0.5)
            tool_call_counts_p90[tool] = _quantile(counts, 0.9)

        # Per-model cost split over each neighbour's remaining suffix.
        usd_by_model_p50: dict[str, float] = {}
        usd_by_model_p90: dict[str, float] = {}
        all_models: set[str] = set()
        per_neighbour_model_cost: list[dict[str, float]] = []
        for nb in neighbours:
            m: dict[str, float] = {}
            for step in nb.trajectory.steps[prefix_len:]:
                if step.kind == "generation" and step.model:
                    m[step.model] = m.get(step.model, 0.0) + step.usd
            per_neighbour_model_cost.append(m)
            all_models.update(m)
        for model in all_models:
            costs = [m.get(model, 0.0) for m in per_neighbour_model_cost]
            usd_by_model_p50[model] = _quantile(costs, 0.5)
            usd_by_model_p90[model] = _quantile(costs, 0.9)

        offrails = sum(1 for nb in neighbours if nb.status in ("error", "cancelled")) / n

        return PrefixPrediction(
            n_samples=n,
            confidence=confidence,
            p50_usd=_quantile(usd, 0.5),
            p90_usd=_quantile(usd, 0.9),
            p99_usd=_quantile(usd, 0.99),
            p50_steps=_quantile(steps, 0.5),
            p90_steps=_quantile(steps, 0.9),
            p99_steps=_quantile(steps, 0.99),
            p50_seconds=_quantile(seconds, 0.5),
            p90_seconds=_quantile(seconds, 0.9),
            p99_seconds=_quantile(seconds, 0.99),
            p50_tokens=_quantile(tokens, 0.5),
            p90_tokens=_quantile(tokens, 0.9),
            p50_prompt_tokens=_quantile(prompt_tokens, 0.5),
            p90_prompt_tokens=_quantile(prompt_tokens, 0.9),
            p50_completion_tokens=_quantile(completion_tokens, 0.5),
            p90_completion_tokens=_quantile(completion_tokens, 0.9),
            p50_llm_calls=_quantile(llm_calls, 0.5),
            p50_tool_calls=_quantile(tool_calls, 0.5),
            p50_compute_seconds=_quantile(compute_seconds, 0.5),
            p50_io_seconds=_quantile(io_seconds, 0.5),
            offrails_score=offrails,
            next_kind_distribution=next_kind_dist,
            next_tool_distribution=next_tool_dist,
            likely_next_model=likely_next_model,
            expected_next_step_usd=_quantile(next_step_usd, 0.5) if next_step_usd else 0.0,
            expected_next_step_seconds=_quantile(next_step_seconds, 0.5)
            if next_step_seconds
            else 0.0,
            p_finish_within_one_step=p_finish_within_one_step,
            tool_call_counts_p50=tool_call_counts_p50,
            tool_call_counts_p90=tool_call_counts_p90,
            usd_by_model_p50=usd_by_model_p50,
            usd_by_model_p90=usd_by_model_p90,
            explanation=(
                f"kNN k={n}: median final ${_quantile(usd, 0.5):.4f}, "
                f"next-kind {next_kind_dist}"
            ),
        )


def _confidence(neighbours: Sequence[Neighbour]) -> float:
    """Heuristic: tight clusters → high confidence; sparse → low."""
    if len(neighbours) < 3:
        return 0.2
    dists = [nb.distance for nb in neighbours]
    mean = sum(dists) / len(dists)
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    std = math.sqrt(var)
    if mean <= 0:
        return 0.9
    cv = std / mean
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
        explanation="no neighbours",
    )

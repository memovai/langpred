"""
Prefix featurization — turn a (possibly partial) Trajectory into a feature
vector the predictors can consume.

Pure-Python, no numpy dependency, so the server can run without scikit-learn.
The kNN predictor consumes :class:`FeatureVector` directly; the optional GBM
predictor turns the same dataclass into a numpy array on demand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from ..trajectories import Step, Trajectory


# Ordered list of numeric features. The exact order is part of the on-disk
# contract for any persisted predictor — append, don't reorder.
FEATURE_NAMES: list[str] = [
    "step_count",
    "elapsed_seconds",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "total_usd",
    "unique_observation_kinds",
    "unique_tool_names",
    "unique_models",
    "loop_repeat_streak",
    "last_step_latency_ms",
    "mean_step_latency_ms",
    "prompt_growth_slope",
    "fraction_generations",
    "fraction_tools",
    "had_error",
]


@dataclass
class FeatureVector:
    trace_name: str | None  # used as a categorical bucket by the predictor
    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values))


def _utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def featurize(traj: Trajectory, prefix_len: int | None = None) -> FeatureVector:
    """Featurise either the full trajectory or its prefix of length ``prefix_len``."""
    steps: Sequence[Step] = traj.steps if prefix_len is None else traj.steps[:prefix_len]

    step_count = len(steps)
    total_prompt = sum(s.prompt_tokens for s in steps)
    total_completion = sum(s.completion_tokens for s in steps)
    total_tokens = sum(s.total_tokens for s in steps)
    total_usd = sum(s.usd for s in steps)

    kinds = {s.kind for s in steps}
    tool_names = {s.tool_name or s.name for s in steps if s.kind != "generation"}
    tool_names.discard(None)
    models = {s.model for s in steps if s.kind == "generation" and s.model}

    # Loop / repeat streak: how many of the trailing steps share descriptor
    streak = 0
    if steps:
        last_desc = steps[-1].descriptor()
        for s in reversed(steps):
            if s.descriptor() == last_desc:
                streak += 1
            else:
                break

    last_latency = steps[-1].latency_ms if steps else 0.0
    mean_latency = (sum(s.latency_ms for s in steps) / step_count) if step_count else 0.0

    # Prompt growth slope (linear regression slope over prompt token sizes by step idx).
    if step_count >= 3:
        xs = list(range(step_count))
        ys = [float(s.prompt_tokens) for s in steps]
        x_mean = sum(xs) / step_count
        y_mean = sum(ys) / step_count
        num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(step_count))
        den = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = num / den
    else:
        slope = 0.0

    # Elapsed wall time.
    start = _utc(traj.start_ts)
    end_candidate = _utc(steps[-1].end_ts) if steps and steps[-1].end_ts else None
    if start and end_candidate:
        elapsed = max(0.0, (end_candidate - start).total_seconds())
    elif start:
        elapsed = max(0.0, (datetime.now(timezone.utc) - start).total_seconds())
    else:
        elapsed = 0.0

    fraction_generations = (
        sum(1 for s in steps if s.kind == "generation") / step_count
        if step_count
        else 0.0
    )
    fraction_tools = (
        sum(1 for s in steps if s.is_tool) / step_count if step_count else 0.0
    )

    had_error = any((s.level or "").upper() == "ERROR" for s in steps)

    values = [
        float(step_count),
        float(elapsed),
        float(total_prompt),
        float(total_completion),
        float(total_tokens),
        float(total_usd),
        float(len(kinds)),
        float(len(tool_names)),
        float(len(models)),
        float(streak),
        float(last_latency),
        float(mean_latency),
        float(slope),
        float(fraction_generations),
        float(fraction_tools),
        float(int(had_error)),
    ]
    return FeatureVector(trace_name=traj.name, values=values)


def l2_distance(a: FeatureVector, b: FeatureVector, weights: Sequence[float] | None = None) -> float:
    """Weighted L2 between two prefix feature vectors. Same trace_name → small bonus."""
    if len(a.values) != len(b.values):
        return float("inf")
    w = weights or [1.0] * len(a.values)
    s = 0.0
    for i in range(len(a.values)):
        s += w[i] * (a.values[i] - b.values[i]) ** 2
    d = math.sqrt(s)
    if a.trace_name and b.trace_name and a.trace_name == b.trace_name:
        d *= 0.5  # halve distance when shapes match
    return d

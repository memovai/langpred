"""
Tier-2 predictor: gradient-boosted quantile regressors.

Optional — only imported if scikit-learn is installed. When unavailable, the
predictor service falls back to kNN+heuristic. We train three small models
per (project, trace_name): final cost, final steps, final wall-time. Each
model has q∈{0.5, 0.9, 0.99} heads.

Training is O(milliseconds) on small datasets so we just retrain from scratch
on the schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..trajectories import Trajectory
from .featurize import FeatureVector, featurize
from .knn import PrefixPrediction

try:  # optional dependency
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    _SK_AVAILABLE = True
except Exception:  # pragma: no cover
    _SK_AVAILABLE = False


@dataclass
class _Triple:
    p50: object
    p90: object
    p99: object


class GBMPredictor:
    """Lightweight quantile-regression head over prefix features.

    A single instance handles all three targets (cost, steps, seconds) and all
    three quantiles. Falls back gracefully when scikit-learn is missing.
    """

    available: bool = _SK_AVAILABLE

    def __init__(self, completed: Iterable[Trajectory], max_prefix: int = 200) -> None:
        self.max_prefix = max_prefix
        self.n = 0
        self._models: dict[str, _Triple] | None = None
        if not _SK_AVAILABLE:
            return
        X, y_cost, y_steps, y_secs = [], [], [], []
        for traj in completed:
            if not traj.is_complete or traj.step_count == 0:
                continue
            cap = min(traj.step_count, max_prefix)
            for k in range(1, cap + 1):
                fv = featurize(traj, k)
                X.append(fv.values)
                y_cost.append(traj.total_usd)
                y_steps.append(traj.step_count)
                y_secs.append(traj.wall_seconds)
        if not X:
            return
        Xa = np.asarray(X, dtype=float)
        self.n = len(X)
        self._models = {
            "cost": self._fit_triple(Xa, np.asarray(y_cost, dtype=float)),
            "steps": self._fit_triple(Xa, np.asarray(y_steps, dtype=float)),
            "seconds": self._fit_triple(Xa, np.asarray(y_secs, dtype=float)),
        }

    @staticmethod
    def _fit_triple(X: "np.ndarray", y: "np.ndarray") -> _Triple:
        def _fit(q: float):
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=80,
                max_depth=4,
                learning_rate=0.07,
            )
            m.fit(X, y)
            return m

        return _Triple(p50=_fit(0.5), p90=_fit(0.9), p99=_fit(0.99))

    def predict(self, prefix: Trajectory) -> PrefixPrediction | None:
        if not _SK_AVAILABLE or self._models is None or self.n == 0:
            return None
        fv = featurize(prefix, prefix.step_count or 1)
        x = np.asarray([fv.values], dtype=float)

        def _q(name: str) -> tuple[float, float, float]:
            t = self._models[name]
            return float(t.p50.predict(x)[0]), float(t.p90.predict(x)[0]), float(t.p99.predict(x)[0])

        c50, c90, c99 = _q("cost")
        s50, s90, s99 = _q("steps")
        t50, t90, t99 = _q("seconds")
        return PrefixPrediction(
            n_samples=self.n,
            confidence=0.6,  # GBM is overconfident on small data; conservative default
            p50_usd=max(0.0, c50),
            p90_usd=max(0.0, c90),
            p99_usd=max(0.0, c99),
            p50_steps=max(0.0, s50),
            p90_steps=max(0.0, s90),
            p99_steps=max(0.0, s99),
            p50_seconds=max(0.0, t50),
            p90_seconds=max(0.0, t90),
            p99_seconds=max(0.0, t99),
            offrails_score=0.0,  # GBM here doesn't model status; kNN owns that
            explanation=f"GBM quantile regression on {self.n} prefix samples",
        )

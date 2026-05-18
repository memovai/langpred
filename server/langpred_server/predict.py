"""
Prediction service — combines tier-0 heuristic, kNN, and (optional) GBM.

The :class:`PredictionService` owns the in-memory predictor cache. It rebuilds
predictors lazily; concurrent reads are safe because we *replace* the
predictor object atomically, never mutate it in place.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

from .schemas import Prediction
from .settings import SETTINGS
from .trajectories import Trajectory, all_trajectories, get_trajectory
from .ml.featurize import featurize
from .ml.gbm import GBMPredictor
from .ml.knn import KNNPredictor, PrefixPrediction


PredKind = Literal["eta", "cost", "offrails", "steps"]


@dataclass
class _ProjectModels:
    knn: KNNPredictor
    gbm: GBMPredictor | None
    n_complete: int


class PredictionService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, _ProjectModels] | None = None  # keyed by trace.name
        # Single global bucket also stored under key="" for cross-shape fallback.

    # --------------------------------------------------------------- training

    def rebuild(self) -> None:
        """(Re)build per-trace-name predictors from current store."""
        trajs = [t for t in all_trajectories() if t.is_complete]
        groups: dict[str, list[Trajectory]] = {"": []}
        for t in trajs:
            key = t.name or ""
            groups.setdefault(key, []).append(t)
            groups[""].append(t)
        new_models: dict[str, _ProjectModels] = {}
        for key, group in groups.items():
            knn = KNNPredictor(group)
            gbm: GBMPredictor | None = None
            if len(group) >= SETTINGS.gbm_promote_threshold and GBMPredictor.available:
                gbm = GBMPredictor(group)
            new_models[key] = _ProjectModels(knn=knn, gbm=gbm, n_complete=len(group))
        with self._lock:
            self._models = new_models

    def _ensure_built(self) -> None:
        if self._models is None:
            self.rebuild()

    # --------------------------------------------------------------- predict

    def predict(self, trace_id: str, kind: PredKind) -> Prediction:
        self._ensure_built()
        traj = get_trajectory(trace_id)
        if traj is None:
            return _empty(trace_id, kind, reason="trace not found")
        models = self._pick_models(traj)

        # Tier 0 heuristic floor: simple "median per step" from the current cohort.
        h = self._heuristic(traj, models)

        # Tier 1: kNN.
        knn_pred = models.knn.predict(traj) if models.knn.n else None
        # Tier 2: GBM (if promoted).
        gbm_pred = models.gbm.predict(traj) if models.gbm else None

        chosen, tier = self._select(gbm_pred, knn_pred, h)
        return _to_prediction(trace_id, kind, chosen, tier)

    # ------------------------------------------------------------- internals

    def _pick_models(self, traj: Trajectory) -> _ProjectModels:
        assert self._models is not None
        key = traj.name or ""
        # Prefer same-shape model when it has enough data, else global.
        same = self._models.get(key)
        if same and same.n_complete >= 5:
            return same
        return self._models.get("", _ProjectModels(knn=KNNPredictor([]), gbm=None, n_complete=0))

    def _heuristic(self, traj: Trajectory, models: _ProjectModels) -> PrefixPrediction:
        """Tier-0 floor — derive median/p90 per-step rates from same-shape cohort."""
        cohort: list[Trajectory] = []
        if models is not None and models.knn is not None:
            cohort = models.knn._traces  # type: ignore[attr-defined]
        if not cohort:
            # No history at all — extrapolate from the current trace's own pace.
            spent = traj.total_usd
            steps = max(1, traj.step_count)
            per = spent / steps if steps else 0.0
            return PrefixPrediction(
                n_samples=0,
                confidence=0.15,
                p50_usd=spent + per * max(1, steps) * 1.0,
                p90_usd=spent + per * max(1, steps) * 3.0,
                p99_usd=spent + per * max(1, steps) * 8.0,
                p50_steps=steps * 2.0,
                p90_steps=steps * 4.0,
                p99_steps=steps * 8.0,
                p50_seconds=max(1.0, traj.wall_seconds * 2.0),
                p90_seconds=max(1.0, traj.wall_seconds * 4.0),
                p99_seconds=max(1.0, traj.wall_seconds * 8.0),
                offrails_score=0.0,
                explanation="cold-start heuristic (no cohort)",
            )

        # Use per-step rates of the cohort.
        per_step_costs = []
        per_step_secs = []
        step_totals = []
        for c in cohort:
            if c.step_count > 0:
                per_step_costs.append(c.total_usd / c.step_count)
                per_step_secs.append(c.wall_seconds / max(1, c.step_count))
                step_totals.append(c.step_count)
        if not per_step_costs:
            return _zero()

        def q(vs, p):
            vs = sorted(vs)
            idx = int(p * (len(vs) - 1))
            return vs[idx]

        spent_steps = traj.step_count
        median_total_steps = q(step_totals, 0.5)
        p90_total_steps = q(step_totals, 0.9)
        p99_total_steps = q(step_totals, 0.99)
        steps_left_p50 = max(0.0, median_total_steps - spent_steps)
        steps_left_p90 = max(0.0, p90_total_steps - spent_steps)
        steps_left_p99 = max(0.0, p99_total_steps - spent_steps)

        cost_p50 = traj.total_usd + steps_left_p50 * q(per_step_costs, 0.5)
        cost_p90 = traj.total_usd + steps_left_p90 * q(per_step_costs, 0.9)
        cost_p99 = traj.total_usd + steps_left_p99 * q(per_step_costs, 0.99)

        time_p50 = traj.wall_seconds + steps_left_p50 * q(per_step_secs, 0.5)
        time_p90 = traj.wall_seconds + steps_left_p90 * q(per_step_secs, 0.9)
        time_p99 = traj.wall_seconds + steps_left_p99 * q(per_step_secs, 0.99)

        return PrefixPrediction(
            n_samples=len(cohort),
            confidence=0.35,
            p50_usd=cost_p50,
            p90_usd=cost_p90,
            p99_usd=cost_p99,
            p50_steps=median_total_steps,
            p90_steps=p90_total_steps,
            p99_steps=p99_total_steps,
            p50_seconds=time_p50,
            p90_seconds=time_p90,
            p99_seconds=time_p99,
            offrails_score=0.0,
            explanation=f"heuristic on cohort of {len(cohort)} traces",
        )

    @staticmethod
    def _select(
        gbm: PrefixPrediction | None,
        knn: PrefixPrediction | None,
        heur: PrefixPrediction,
    ) -> tuple[PrefixPrediction, str]:
        if gbm is not None and knn is not None and knn.n_samples >= 30:
            # Blend gbm + knn 50/50 — they tend to disagree in useful ways.
            blended = PrefixPrediction(
                n_samples=knn.n_samples,
                confidence=max(knn.confidence, gbm.confidence),
                p50_usd=(gbm.p50_usd + knn.p50_usd) / 2,
                p90_usd=(gbm.p90_usd + knn.p90_usd) / 2,
                p99_usd=(gbm.p99_usd + knn.p99_usd) / 2,
                p50_steps=(gbm.p50_steps + knn.p50_steps) / 2,
                p90_steps=(gbm.p90_steps + knn.p90_steps) / 2,
                p99_steps=(gbm.p99_steps + knn.p99_steps) / 2,
                p50_seconds=(gbm.p50_seconds + knn.p50_seconds) / 2,
                p90_seconds=(gbm.p90_seconds + knn.p90_seconds) / 2,
                p99_seconds=(gbm.p99_seconds + knn.p99_seconds) / 2,
                offrails_score=knn.offrails_score,
                explanation="blend(gbm, knn)",
            )
            return blended, "gbm"
        if knn is not None and knn.n_samples >= 5:
            return knn, "knn"
        return heur, "heuristic"


def _to_prediction(trace_id: str, kind: PredKind, p: PrefixPrediction, tier: str) -> Prediction:
    if kind == "cost":
        return Prediction(
            trace_id=trace_id,
            kind=kind,
            p50=p.p50_usd,
            p90=p.p90_usd,
            p99=p.p99_usd,
            confidence=p.confidence,
            tier=tier,  # type: ignore[arg-type]
            n_samples=p.n_samples,
            explanation=p.explanation,
        )
    if kind == "eta":
        return Prediction(
            trace_id=trace_id,
            kind=kind,
            p50=p.p50_seconds,
            p90=p.p90_seconds,
            p99=p.p99_seconds,
            confidence=p.confidence,
            tier=tier,  # type: ignore[arg-type]
            n_samples=p.n_samples,
            explanation=p.explanation,
        )
    if kind == "steps":
        return Prediction(
            trace_id=trace_id,
            kind=kind,
            p50=p.p50_steps,
            p90=p.p90_steps,
            p99=p.p99_steps,
            confidence=p.confidence,
            tier=tier,  # type: ignore[arg-type]
            n_samples=p.n_samples,
            explanation=p.explanation,
        )
    # offrails
    return Prediction(
        trace_id=trace_id,
        kind="offrails",
        p50=p.offrails_score,
        p90=p.offrails_score,
        p99=p.offrails_score,
        confidence=p.confidence,
        tier=tier,  # type: ignore[arg-type]
        n_samples=p.n_samples,
        explanation=p.explanation,
    )


def _empty(trace_id: str, kind: PredKind, reason: str) -> Prediction:
    return Prediction(
        trace_id=trace_id,
        kind=kind,
        p50=0.0,
        p90=0.0,
        p99=0.0,
        confidence=0.0,
        tier="heuristic",
        n_samples=0,
        explanation=reason,
    )


def _zero() -> PrefixPrediction:
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
        explanation="empty",
    )


_SERVICE: PredictionService | None = None


def get_service() -> PredictionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PredictionService()
    return _SERVICE


def reset_service_for_tests() -> PredictionService:
    global _SERVICE
    _SERVICE = PredictionService()
    return _SERVICE

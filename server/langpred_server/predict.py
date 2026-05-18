"""
Prediction service — combines tier-0 heuristic, kNN, and (optional) GBM, then
folds the result into the omnibus :class:`AgentPrediction`.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from .schemas import (
    AgentPrediction,
    CostPrediction,
    NextActionPrediction,
    Prediction,
    ResourcePrediction,
    RiskPrediction,
    TimePrediction,
    _Meta,
    _ToolCount,
    _ToolProb,
    _UsdByModel,
)
from .settings import SETTINGS
from .trajectories import Trajectory, all_trajectories, get_trajectory
from .ml.featurize import featurize
from .ml.gbm import GBMPredictor
from .ml.knn import KNNPredictor, PrefixPrediction
from .ml.pricing import context_window


PredKind = Literal["eta", "cost", "offrails", "steps"]


@dataclass
class _ProjectModels:
    knn: KNNPredictor
    gbm: GBMPredictor | None
    n_complete: int


class PredictionService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, _ProjectModels] | None = None

    # --------------------------------------------------------------- training

    def rebuild(self) -> None:
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

    # ------------------------------------------------- single-kind (legacy)

    def predict(self, trace_id: str, kind: PredKind) -> Prediction:
        """Return the per-kind :class:`Prediction` (backward-compat endpoint)."""
        full = self.predict_all(trace_id)
        if full is None:
            return _empty(trace_id, kind, reason="trace not found")
        if kind == "cost":
            return Prediction(
                trace_id=trace_id,
                kind="cost",
                p50=full.cost.usd_total_p50,
                p90=full.cost.usd_total_p90,
                p99=full.cost.usd_total_p99,
                confidence=full.meta.confidence,
                tier=full.meta.tier,
                n_samples=full.meta.n_samples,
                explanation=full.meta.explanation,
            )
        if kind == "eta":
            return Prediction(
                trace_id=trace_id,
                kind="eta",
                p50=full.time.total_seconds_p50,
                p90=full.time.total_seconds_p90,
                p99=full.time.total_seconds_p99,
                confidence=full.meta.confidence,
                tier=full.meta.tier,
                n_samples=full.meta.n_samples,
                explanation=full.meta.explanation,
            )
        if kind == "steps":
            return Prediction(
                trace_id=trace_id,
                kind="steps",
                p50=full.resources.total_steps_p50,
                p90=full.resources.total_steps_p90,
                p99=full.resources.total_steps_p90,  # p99 not split for steps
                confidence=full.meta.confidence,
                tier=full.meta.tier,
                n_samples=full.meta.n_samples,
                explanation=full.meta.explanation,
            )
        return Prediction(
            trace_id=trace_id,
            kind="offrails",
            p50=full.risk.offrails_risk,
            p90=full.risk.offrails_risk,
            p99=full.risk.offrails_risk,
            confidence=full.meta.confidence,
            tier=full.meta.tier,
            n_samples=full.meta.n_samples,
            explanation=full.meta.explanation,
        )

    # ------------------------------------------------------------- forecast

    def forecast(
        self,
        trace_name: str,
        budget_cap_usd: float | None = None,
    ) -> AgentPrediction:
        """Predict outcomes for a **hypothetical** trace with this name —
        before any steps exist. Powers reject-upfront and route-at-start by
        letting callers ask "what will this kind of run typically cost?"
        without first creating a trace.

        Uses the same-name cohort directly (median/p90 of final outcomes,
        aggregated tool / model histograms). Falls back to the global cohort
        if the named one is empty, and to a zero prediction if there's no
        cohort at all.
        """
        self._ensure_built()
        assert self._models is not None
        models = self._models.get(trace_name) or self._models.get("") or _ProjectModels(
            knn=KNNPredictor([]), gbm=None, n_complete=0
        )
        cohort = models.knn._traces  # type: ignore[attr-defined]

        # Build an "empty" trajectory placeholder so the assembly helper has
        # something to compute remaining_* and elapsed_* against. We set
        # ``start_ts == end_ts`` so elapsed is exactly 0 (no clock-skew slop).
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        placeholder = Trajectory(
            trace_id="forecast-" + trace_name,
            name=trace_name,
            start_ts=now,
            end_ts=now,
            steps=[],
        )

        prefix = self._cohort_prefix_prediction(cohort)
        agent = _to_agent_prediction(placeholder, prefix, "knn" if cohort else "heuristic", budget_cap_usd)
        return agent

    @staticmethod
    def _cohort_prefix_prediction(cohort: list[Trajectory]) -> PrefixPrediction:
        """Aggregate a cohort directly into a PrefixPrediction (no kNN
        ranking — this is the prior, used when there's no query trajectory)."""
        from collections import Counter

        from .ml.knn import _quantile, _zero_prediction

        if not cohort:
            return _zero_prediction()

        usd = [c.total_usd for c in cohort]
        steps = [float(c.step_count) for c in cohort]
        seconds = [c.wall_seconds for c in cohort]
        tokens = [float(c.total_tokens) for c in cohort]
        prompt_t = [float(c.prompt_tokens) for c in cohort]
        completion_t = [float(c.completion_tokens) for c in cohort]
        llm_calls = [float(c.llm_call_count) for c in cohort]
        tool_calls = [float(c.tool_call_count) for c in cohort]
        compute_s = [c.compute_seconds for c in cohort]
        io_s = [c.io_seconds for c in cohort]
        offrails = sum(1 for c in cohort if c.status in ("error", "cancelled")) / len(cohort)

        # First-step distribution and likely first model.
        first_kind: Counter[str] = Counter()
        first_tool: Counter[str] = Counter()
        first_model: Counter[str] = Counter()
        first_step_usd: list[float] = []
        first_step_secs: list[float] = []
        for c in cohort:
            if not c.steps:
                continue
            step = c.steps[0]
            label = "generation" if step.kind == "generation" else "tool_call"
            first_kind[label] += 1
            if label == "tool_call":
                first_tool[step.tool_name or step.name or "unknown"] += 1
            elif step.model:
                first_model[step.model] += 1
            first_step_usd.append(step.usd)
            first_step_secs.append(step.latency_ms / 1000.0)

        total = len(cohort)
        next_kind_dist = {k: c / total for k, c in first_kind.items()}
        next_tool_dist = {k: c / total for k, c in first_tool.items()}
        likely_model = first_model.most_common(1)[0][0] if first_model else None

        # Per-tool and per-model histograms over full traces.
        per_tool: dict[str, list[float]] = {}
        per_model: dict[str, list[float]] = {}
        for c in cohort:
            for t, n in c.tool_histogram().items():
                per_tool.setdefault(t, [0.0] * total)
            for m in c.model_cost_histogram():
                per_model.setdefault(m, [0.0] * total)
        for i, c in enumerate(cohort):
            for t, n in c.tool_histogram().items():
                per_tool[t][i] = float(n)
            for m, cost in c.model_cost_histogram().items():
                per_model[m][i] = float(cost)

        tool_call_counts_p50 = {t: _quantile(vs, 0.5) for t, vs in per_tool.items()}
        tool_call_counts_p90 = {t: _quantile(vs, 0.9) for t, vs in per_tool.items()}
        usd_by_model_p50 = {m: _quantile(vs, 0.5) for m, vs in per_model.items()}
        usd_by_model_p90 = {m: _quantile(vs, 0.9) for m, vs in per_model.items()}

        return PrefixPrediction(
            n_samples=total,
            confidence=min(0.9, 0.3 + total / 200.0),
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
            p50_prompt_tokens=_quantile(prompt_t, 0.5),
            p90_prompt_tokens=_quantile(prompt_t, 0.9),
            p50_completion_tokens=_quantile(completion_t, 0.5),
            p90_completion_tokens=_quantile(completion_t, 0.9),
            p50_llm_calls=_quantile(llm_calls, 0.5),
            p50_tool_calls=_quantile(tool_calls, 0.5),
            p50_compute_seconds=_quantile(compute_s, 0.5),
            p50_io_seconds=_quantile(io_s, 0.5),
            offrails_score=offrails,
            next_kind_distribution=next_kind_dist,
            next_tool_distribution=next_tool_dist,
            likely_next_model=likely_model,
            expected_next_step_usd=_quantile(first_step_usd, 0.5) if first_step_usd else 0.0,
            expected_next_step_seconds=_quantile(first_step_secs, 0.5)
            if first_step_secs
            else 0.0,
            p_finish_within_one_step=0.0,
            tool_call_counts_p50=tool_call_counts_p50,
            tool_call_counts_p90=tool_call_counts_p90,
            usd_by_model_p50=usd_by_model_p50,
            usd_by_model_p90=usd_by_model_p90,
            explanation=f"cohort forecast over {total} finished traces",
        )

    # ------------------------------------------------------------- omnibus

    def predict_all(
        self, trace_id: str, budget_cap_usd: float | None = None
    ) -> AgentPrediction | None:
        """Compute the full :class:`AgentPrediction` for a trace."""
        self._ensure_built()
        traj = get_trajectory(trace_id)
        if traj is None:
            return None
        models = self._pick_models(traj)

        heur = self._heuristic(traj, models)
        knn_pred = models.knn.predict(traj) if models.knn.n else None
        gbm_pred = models.gbm.predict(traj) if models.gbm else None

        chosen, tier = self._select(gbm_pred, knn_pred, heur)
        return _to_agent_prediction(traj, chosen, tier, budget_cap_usd)

    # ------------------------------------------------------------- internals

    def _pick_models(self, traj: Trajectory) -> _ProjectModels:
        assert self._models is not None
        key = traj.name or ""
        same = self._models.get(key)
        if same and same.n_complete >= 5:
            return same
        return self._models.get("", _ProjectModels(knn=KNNPredictor([]), gbm=None, n_complete=0))

    def _heuristic(self, traj: Trajectory, models: _ProjectModels) -> PrefixPrediction:
        cohort: list[Trajectory] = []
        if models is not None and models.knn is not None:
            cohort = models.knn._traces  # type: ignore[attr-defined]
        if not cohort:
            spent = traj.total_usd
            steps = max(1, traj.step_count)
            per = spent / steps if steps else 0.0
            return PrefixPrediction(
                n_samples=0,
                confidence=0.15,
                p50_usd=spent + per * steps,
                p90_usd=spent + per * steps * 3.0,
                p99_usd=spent + per * steps * 8.0,
                p50_steps=steps * 2.0,
                p90_steps=steps * 4.0,
                p99_steps=steps * 8.0,
                p50_seconds=max(1.0, traj.elapsed_seconds * 2.0),
                p90_seconds=max(1.0, traj.elapsed_seconds * 4.0),
                p99_seconds=max(1.0, traj.elapsed_seconds * 8.0),
                explanation="cold-start heuristic (no cohort)",
            )

        def q(vs, p):
            vs = sorted(vs)
            idx = int(p * (len(vs) - 1))
            return vs[idx]

        per_step_costs = [c.total_usd / c.step_count for c in cohort if c.step_count]
        per_step_secs = [c.wall_seconds / max(1, c.step_count) for c in cohort]
        step_totals = [c.step_count for c in cohort]
        if not per_step_costs:
            return _zero_prefix()

        spent_steps = traj.step_count
        median_total_steps = q(step_totals, 0.5)
        p90_total_steps = q(step_totals, 0.9)
        steps_left_p50 = max(0.0, median_total_steps - spent_steps)
        steps_left_p90 = max(0.0, p90_total_steps - spent_steps)

        return PrefixPrediction(
            n_samples=len(cohort),
            confidence=0.35,
            p50_usd=traj.total_usd + steps_left_p50 * q(per_step_costs, 0.5),
            p90_usd=traj.total_usd + steps_left_p90 * q(per_step_costs, 0.9),
            p99_usd=traj.total_usd + steps_left_p90 * q(per_step_costs, 0.99),
            p50_steps=median_total_steps,
            p90_steps=p90_total_steps,
            p99_steps=q(step_totals, 0.99),
            p50_seconds=traj.elapsed_seconds + steps_left_p50 * q(per_step_secs, 0.5),
            p90_seconds=traj.elapsed_seconds + steps_left_p90 * q(per_step_secs, 0.9),
            p99_seconds=traj.elapsed_seconds + steps_left_p90 * q(per_step_secs, 0.99),
            explanation=f"heuristic on cohort of {len(cohort)} traces",
        )

    @staticmethod
    def _select(
        gbm: PrefixPrediction | None,
        knn: PrefixPrediction | None,
        heur: PrefixPrediction,
    ) -> tuple[PrefixPrediction, str]:
        if gbm is not None and knn is not None and knn.n_samples >= 30:
            # Blend headline numbers but keep all of kNN's distributional/histogram
            # information — GBM can't produce those.
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
                p50_tokens=knn.p50_tokens,
                p90_tokens=knn.p90_tokens,
                p50_prompt_tokens=knn.p50_prompt_tokens,
                p90_prompt_tokens=knn.p90_prompt_tokens,
                p50_completion_tokens=knn.p50_completion_tokens,
                p90_completion_tokens=knn.p90_completion_tokens,
                p50_llm_calls=knn.p50_llm_calls,
                p50_tool_calls=knn.p50_tool_calls,
                p50_compute_seconds=knn.p50_compute_seconds,
                p50_io_seconds=knn.p50_io_seconds,
                offrails_score=knn.offrails_score,
                next_kind_distribution=knn.next_kind_distribution,
                next_tool_distribution=knn.next_tool_distribution,
                likely_next_model=knn.likely_next_model,
                expected_next_step_usd=knn.expected_next_step_usd,
                expected_next_step_seconds=knn.expected_next_step_seconds,
                p_finish_within_one_step=knn.p_finish_within_one_step,
                tool_call_counts_p50=knn.tool_call_counts_p50,
                tool_call_counts_p90=knn.tool_call_counts_p90,
                usd_by_model_p50=knn.usd_by_model_p50,
                usd_by_model_p90=knn.usd_by_model_p90,
                explanation="blend(gbm, knn) — distributions from kNN",
            )
            return blended, "gbm"
        if knn is not None and knn.n_samples >= 5:
            return knn, "knn"
        return heur, "heuristic"


# --------------------------------------------------------------- assembly


def _to_agent_prediction(
    traj: Trajectory,
    p: PrefixPrediction,
    tier: str,
    budget_cap_usd: float | None,
) -> AgentPrediction:
    elapsed = traj.elapsed_seconds
    spent = traj.total_usd

    time_pred = TimePrediction(
        total_seconds_p50=max(elapsed, p.p50_seconds),
        total_seconds_p90=max(elapsed, p.p90_seconds),
        total_seconds_p99=max(elapsed, p.p99_seconds),
        remaining_seconds_p50=max(0.0, p.p50_seconds - elapsed),
        remaining_seconds_p90=max(0.0, p.p90_seconds - elapsed),
        remaining_seconds_p99=max(0.0, p.p99_seconds - elapsed),
        next_step_seconds_p50=p.expected_next_step_seconds,
        next_step_seconds_p90=p.expected_next_step_seconds,  # no p90 split yet
        compute_seconds_p50=p.p50_compute_seconds,
        io_seconds_p50=p.p50_io_seconds,
        elapsed_seconds=elapsed,
    )

    cost_pred = CostPrediction(
        usd_total_p50=max(spent, p.p50_usd),
        usd_total_p90=max(spent, p.p90_usd),
        usd_total_p99=max(spent, p.p99_usd),
        usd_remaining_p50=max(0.0, p.p50_usd - spent),
        usd_remaining_p90=max(0.0, p.p90_usd - spent),
        usd_remaining_p99=max(0.0, p.p99_usd - spent),
        next_step_usd_p50=p.expected_next_step_usd,
        next_step_usd_p90=p.expected_next_step_usd,
        usd_by_model=[
            _UsdByModel(
                model=m,
                usd_p50=p.usd_by_model_p50.get(m, 0.0),
                usd_p90=p.usd_by_model_p90.get(m, 0.0),
            )
            for m in sorted(p.usd_by_model_p50)
        ],
        spent_usd=spent,
    )

    resources_pred = ResourcePrediction(
        total_tokens_p50=p.p50_tokens,
        total_tokens_p90=p.p90_tokens,
        prompt_tokens_p50=p.p50_prompt_tokens,
        prompt_tokens_p90=p.p90_prompt_tokens,
        completion_tokens_p50=p.p50_completion_tokens,
        completion_tokens_p90=p.p90_completion_tokens,
        total_steps_p50=p.p50_steps,
        total_steps_p90=p.p90_steps,
        steps_remaining_p50=max(0.0, p.p50_steps - traj.step_count),
        steps_remaining_p90=max(0.0, p.p90_steps - traj.step_count),
        llm_calls_p50=p.p50_llm_calls,
        tool_calls_p50=p.p50_tool_calls,
        tool_call_counts=[
            _ToolCount(
                tool=t,
                p50=p.tool_call_counts_p50.get(t, 0.0),
                p90=p.tool_call_counts_p90.get(t, 0.0),
            )
            for t in sorted(p.tool_call_counts_p50)
        ],
    )

    top_tools = sorted(
        p.next_tool_distribution.items(), key=lambda kv: kv[1], reverse=True
    )[:5]
    next_pred = NextActionPrediction(
        next_kind_distribution=dict(p.next_kind_distribution),
        top_next_tools=[_ToolProb(tool=t, probability=prob) for t, prob in top_tools],
        likely_next_model=p.likely_next_model,
        p_finish_within_one_step=p.p_finish_within_one_step,
        expected_next_step_usd_p50=p.expected_next_step_usd,
        expected_next_step_seconds_p50=p.expected_next_step_seconds,
    )

    # ---- Risks ---------------------------------------------------------
    # Loop risk: trailing repeat-streak as a fraction of step count.
    loop_risk = 0.0
    if traj.step_count >= 4:
        streak = 0
        last_desc = traj.steps[-1].descriptor()
        for s in reversed(traj.steps):
            if s.descriptor() == last_desc:
                streak += 1
            else:
                break
        loop_risk = min(1.0, max(0.0, (streak - 1) / max(3, traj.step_count - 1)))

    # Context-overflow risk: predicted-total prompt tokens vs model context.
    # Use the most-common model used in the trace so far (or the predicted
    # next model) as the guide.
    overflow_risk = 0.0
    representative_model = (
        p.likely_next_model
        or next(
            (s.model for s in reversed(traj.steps) if s.kind == "generation" and s.model),
            None,
        )
    )
    if representative_model:
        cw = context_window(representative_model)
        # Worst-case single-prompt size ~ mean predicted prompt token rate
        # held constant; if p90 total prompt > cw, flag overflow.
        if p.p90_prompt_tokens and cw and p.p90_prompt_tokens > cw:
            overflow_risk = min(1.0, p.p90_prompt_tokens / cw - 1.0 + 0.5)
        elif p.p90_prompt_tokens and cw:
            overflow_risk = max(0.0, p.p90_prompt_tokens / cw - 0.7)

    # Budget overshoot risk: needs a registered cap.
    budget_risk = 0.0
    if budget_cap_usd is not None:
        # Tail share that ends up over the cap (using gaussian-ish proxy on p50/p90).
        if p.p99_usd <= budget_cap_usd:
            budget_risk = 0.0
        elif p.p50_usd >= budget_cap_usd:
            budget_risk = 1.0
        else:
            # Linear interpolate over the band.
            span = max(1e-9, p.p99_usd - p.p50_usd)
            budget_risk = min(1.0, max(0.0, (budget_cap_usd - p.p50_usd) / span))
            budget_risk = 1.0 - budget_risk

    # Cost-spike risk: next-step expected usd vs running per-step median.
    cost_spike_risk = 0.0
    if traj.step_count > 0:
        running_median_step = traj.total_usd / max(1, traj.step_count)
        if running_median_step > 0:
            cost_spike_risk = min(
                1.0, max(0.0, p.expected_next_step_usd / running_median_step / 2.0 - 0.5)
            )

    notes: list[str] = []
    if loop_risk > 0.5:
        notes.append(f"loop risk high (trailing-streak)")
    if overflow_risk > 0.5:
        notes.append(f"context-window pressure on {representative_model}")
    if budget_risk > 0.5:
        notes.append(f"likely to overshoot ${budget_cap_usd}")

    risk_pred = RiskPrediction(
        offrails_risk=p.offrails_score,
        loop_risk=loop_risk,
        context_overflow_risk=overflow_risk,
        budget_overshoot_risk=budget_risk,
        cost_spike_risk=cost_spike_risk,
        notes=notes,
    )

    meta = _Meta(
        tier=tier,  # type: ignore[arg-type]
        n_samples=p.n_samples,
        confidence=p.confidence,
        explanation=p.explanation,
    )

    return AgentPrediction(
        trace_id=traj.trace_id,
        meta=meta,
        time=time_pred,
        cost=cost_pred,
        resources=resources_pred,
        next=next_pred,
        risk=risk_pred,
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


def _zero_prefix() -> PrefixPrediction:
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

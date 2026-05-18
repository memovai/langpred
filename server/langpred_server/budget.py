"""
Budget service — register caps, evaluate on every ingestion, expose status.

Evaluation is cheap: pull the current Trajectory, ask the predictor for
``cost.p50``, and compare to ``cap - spent``. Once a budget is breached we
keep returning ``breached=True`` so SDK clients see a sticky signal even if
the predictor later relaxes.
"""
from __future__ import annotations

from dataclasses import dataclass

from .db import BudgetRecord, get_store
from .predict import get_service
from .schemas import BudgetRequest, BudgetStatus
from .trajectories import get_trajectory


def register(req: BudgetRequest) -> BudgetStatus:
    store = get_store()
    record = BudgetRecord(
        trace_id=req.trace_id,
        cap_usd=req.cap_usd,
        on_exceed=req.on_exceed,
    )
    store.set_budget(record)
    return evaluate(req.trace_id) or _initial_status(record)


def evaluate(trace_id: str) -> BudgetStatus | None:
    """Re-evaluate a budget against current trajectory. Returns ``None`` if no
    budget is registered for the trace."""
    store = get_store()
    rec = store.get_budget(trace_id)
    if rec is None:
        return None

    traj = get_trajectory(trace_id)
    spent = traj.total_usd if traj else 0.0
    pred = get_service().predict(trace_id, "cost")
    remaining_p50 = max(0.0, pred.p50 - spent)
    remaining_p90 = max(0.0, pred.p90 - spent)

    breached_now = False
    reason = rec.breach_reason
    if spent >= rec.cap_usd:
        breached_now = True
        reason = "hard breach (already spent ≥ cap)"
    elif (spent + remaining_p50) >= rec.cap_usd:
        breached_now = True
        reason = (
            f"predicted breach: spent ${spent:.4f} + p50 remaining "
            f"${remaining_p50:.4f} ≥ cap ${rec.cap_usd:.4f}"
        )

    if breached_now or rec.breached:
        rec.breached = True
        rec.breach_reason = reason
    rec.last_spent_usd = spent
    rec.last_predicted_remaining_p50 = remaining_p50
    rec.last_predicted_remaining_p90 = remaining_p90
    store.update_budget(rec)

    return BudgetStatus(
        trace_id=trace_id,
        cap_usd=rec.cap_usd,
        on_exceed=rec.on_exceed,  # type: ignore[arg-type]
        spent_usd=spent,
        predicted_remaining_p50_usd=remaining_p50,
        predicted_remaining_p90_usd=remaining_p90,
        breached=rec.breached,
        breach_reason=rec.breach_reason,
    )


def status(trace_id: str) -> BudgetStatus | None:
    return evaluate(trace_id)


def _initial_status(rec: BudgetRecord) -> BudgetStatus:
    return BudgetStatus(
        trace_id=rec.trace_id,
        cap_usd=rec.cap_usd,
        on_exceed=rec.on_exceed,  # type: ignore[arg-type]
        spent_usd=0.0,
        predicted_remaining_p50_usd=0.0,
        predicted_remaining_p90_usd=0.0,
        breached=False,
        breach_reason=None,
    )

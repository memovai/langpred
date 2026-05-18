"""Local dashboard endpoints.

The public API intentionally mirrors Langfuse for ingestion/prediction. This
module is the local operator surface: compact trace list, detail projection,
and a static UI shell served by the same FastAPI process.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from . import alerts as alerts_mod
from .auth import project_id_from_authorization
from .db import BudgetRecord, get_store
from .predict import get_service
from .trajectories import Step, Trajectory, all_trajectories, get_trajectory


STATIC_DIR = Path(__file__).with_name("static")

router = APIRouter(tags=["local-ui"])
api_router = APIRouter(prefix="/api/local", tags=["local-ui"])


@router.get("/ui", include_in_schema=False)
async def ui_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@router.get("/ui/", include_in_schema=False)
async def ui_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@api_router.get("/traces")
async def list_traces(
    authorization: str | None = Header(default=None),
    q: str | None = Query(default=None),
    status: str = Query(default="all"),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    project_id = project_id_from_authorization(authorization)
    traces = all_trajectories(project_id=project_id)
    traces = _filter_traces(traces, q=q, status=status)
    traces.sort(key=_trace_sort_key, reverse=True)
    visible = traces[:limit]

    return {
        "project_id": project_id,
        "count": len(traces),
        "summary": _summary(traces, project_id=project_id),
        "traces": [_trace_summary(t, project_id=project_id) for t in visible],
    }


@api_router.get("/traces/{trace_id}")
async def trace_detail(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    project_id = project_id_from_authorization(authorization)
    traj = get_trajectory(trace_id, project_id=project_id)
    if traj is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return {
        "project_id": project_id,
        "trace": _trace_detail(traj),
        "prediction": _prediction_dump(trace_id, project_id=project_id),
        "budget": _budget_dump(get_store().get_budget(trace_id, project_id=project_id)),
        "alerts": [
            _alert_dump(rule)
            for rule in alerts_mod.list_for(trace_id, project_id=project_id)
        ],
    }


@api_router.post("/rebuild")
async def rebuild_models() -> dict[str, Any]:
    get_service().rebuild()
    return {"ok": True}


def _filter_traces(
    traces: list[Trajectory],
    *,
    q: str | None,
    status: str,
) -> list[Trajectory]:
    needle = (q or "").strip().lower()
    allowed_statuses = {"all", "open", "ok", "error", "cancelled"}
    selected_status = status if status in allowed_statuses else "all"

    out: list[Trajectory] = []
    for trace in traces:
        if selected_status != "all" and trace.status != selected_status:
            continue
        if needle:
            haystack = " ".join(
                str(part or "")
                for part in (
                    trace.trace_id,
                    trace.name,
                    trace.user_id,
                    trace.session_id,
                    trace.release,
                    trace.version,
                )
            ).lower()
            if needle not in haystack:
                continue
        out.append(trace)
    return out


def _summary(traces: list[Trajectory], *, project_id: str) -> dict[str, Any]:
    total_cost = sum(t.total_usd for t in traces)
    total_tokens = sum(t.total_tokens for t in traces)
    open_count = sum(1 for t in traces if t.status == "open")
    errored_count = sum(1 for t in traces if t.status in {"error", "cancelled"})
    p90_costs: list[float] = []
    high_risk = 0

    for trace in traces[:100]:
        pred = _prediction_dump(trace.trace_id, project_id=project_id)
        if not pred:
            continue
        p90_costs.append(float(pred["cost"]["usd_total_p90"]))
        risk = pred.get("risk", {})
        if max(
            float(risk.get("offrails_risk") or 0),
            float(risk.get("loop_risk") or 0),
            float(risk.get("budget_overshoot_risk") or 0),
        ) >= 0.7:
            high_risk += 1

    return {
        "traces": len(traces),
        "open_traces": open_count,
        "errored_traces": errored_count,
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "p90_cost_usd": max(p90_costs) if p90_costs else 0.0,
        "high_risk_traces": high_risk,
    }


def _trace_summary(trace: Trajectory, *, project_id: str) -> dict[str, Any]:
    pred = _prediction_dump(trace.trace_id, project_id=project_id)
    headline = _headline_prediction(pred)
    return {
        "id": trace.trace_id,
        "name": trace.name,
        "status": trace.status,
        "user_id": trace.user_id,
        "session_id": trace.session_id,
        "start_ts": _iso(trace.start_ts),
        "end_ts": _iso(trace.end_ts),
        "updated_ts": _iso(_trace_updated_at(trace)),
        "elapsed_seconds": trace.elapsed_seconds,
        "step_count": trace.step_count,
        "llm_calls": trace.llm_call_count,
        "tool_calls": trace.tool_call_count,
        "total_tokens": trace.total_tokens,
        "total_usd": trace.total_usd,
        "release": trace.release,
        "version": trace.version,
        "prediction": headline,
    }


def _trace_detail(trace: Trajectory) -> dict[str, Any]:
    return {
        **_trace_summary(trace, project_id=trace.project_id),
        "input": trace.input,
        "metadata": trace.metadata,
        "observations": [_step_dump(step, index=i) for i, step in enumerate(trace.steps)],
    }


def _step_dump(step: Step, *, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "observation_id": step.observation_id,
        "kind": step.kind,
        "name": step.name,
        "tool_name": step.tool_name,
        "model": step.model,
        "start_ts": _iso(step.start_ts),
        "end_ts": _iso(step.end_ts),
        "latency_ms": step.latency_ms,
        "prompt_tokens": step.prompt_tokens,
        "completion_tokens": step.completion_tokens,
        "total_tokens": step.total_tokens,
        "usd": step.usd,
        "level": step.level,
        "status_message": step.status_message,
    }


def _prediction_dump(trace_id: str, *, project_id: str) -> dict[str, Any] | None:
    budget = get_store().get_budget(trace_id, project_id=project_id)
    try:
        pred = get_service().predict_all(
            trace_id,
            project_id=project_id,
            budget_cap_usd=budget.cap_usd if budget else None,
        )
    except Exception:
        return None
    return pred.model_dump(mode="json") if pred else None


def _headline_prediction(pred: dict[str, Any] | None) -> dict[str, Any] | None:
    if pred is None:
        return None
    risk = pred["risk"]
    return {
        "tier": pred["meta"]["tier"],
        "n_samples": pred["meta"]["n_samples"],
        "remaining_seconds_p50": pred["time"]["remaining_seconds_p50"],
        "usd_total_p50": pred["cost"]["usd_total_p50"],
        "usd_total_p90": pred["cost"]["usd_total_p90"],
        "steps_remaining_p50": pred["resources"]["steps_remaining_p50"],
        "risk": max(
            risk["offrails_risk"],
            risk["loop_risk"],
            risk["context_overflow_risk"],
            risk["budget_overshoot_risk"],
            risk["cost_spike_risk"],
        ),
        "next_kind_distribution": pred["next"]["next_kind_distribution"],
    }


def _budget_dump(budget: BudgetRecord | None) -> dict[str, Any] | None:
    if budget is None:
        return None
    return {
        "trace_id": budget.trace_id,
        "cap_usd": budget.cap_usd,
        "on_exceed": budget.on_exceed,
        "quantile": budget.quantile,
        "breached": budget.breached,
        "breach_reason": budget.breach_reason,
        "spent_usd": budget.last_spent_usd,
        "predicted_remaining_p50_usd": budget.last_predicted_remaining_p50,
        "predicted_remaining_p90_usd": budget.last_predicted_remaining_p90,
        "created_at": budget.created_at,
        "updated_at": budget.updated_at,
    }


def _alert_dump(rule: Any) -> dict[str, Any]:
    return {
        "id": rule.id,
        "trace_id": rule.trace_id,
        "condition": rule.condition,
        "webhook_url": rule.webhook_url,
        "last_fired_at": rule.last_fired_at,
        "fire_count": rule.fire_count,
        "last_value": rule.last_value,
    }


def _trace_sort_key(trace: Trajectory) -> float:
    dt = _trace_updated_at(trace)
    return _timestamp(dt) if dt else 0.0


def _trace_updated_at(trace: Trajectory) -> datetime | None:
    candidates: list[datetime | None] = [trace.end_ts]
    for step in reversed(trace.steps):
        candidates.extend([step.end_ts, step.start_ts])
        break
    candidates.append(trace.start_ts)
    return next((dt for dt in candidates if dt is not None), None)


def _timestamp(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()

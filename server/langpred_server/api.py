"""HTTP routes for prediction + budget."""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from . import alerts as alerts_mod
from .auth import project_id_from_authorization
from .budget import register as register_budget, status as budget_status
from .db import get_store
from .predict import get_service
from .schemas import (
    AgentPrediction,
    AlertRule,
    AlertRuleStatus,
    BudgetRequest,
    BudgetStatus,
    ForecastRequest,
    Prediction,
)


router = APIRouter(prefix="/api/public", tags=["predict"])


# -------------------------------------------------------- pre-trace forecast


@router.post("/forecast", response_model=AgentPrediction)
async def post_forecast(
    req: ForecastRequest,
    authorization: str | None = Header(default=None),
) -> AgentPrediction:
    """Forecast for a hypothetical trace — used for reject-upfront and
    route-at-start decisions before any KV cache exists."""
    return get_service().forecast(
        trace_name=req.trace_name,
        project_id=project_id_from_authorization(authorization),
        user_id=req.user_id,
        session_id=req.session_id,
        input=req.input,
        metadata=req.metadata,
    )


@router.get("/forecast", response_model=AgentPrediction)
async def get_forecast(
    trace_name: str,
    authorization: str | None = Header(default=None),
) -> AgentPrediction:
    return get_service().forecast(
        trace_name=trace_name,
        project_id=project_id_from_authorization(authorization),
    )


@router.get("/predict/{trace_id}", response_model=AgentPrediction)
async def predict_all(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> AgentPrediction:
    """Omnibus prediction: time, cost, resources, next action, and risk."""
    project_id = project_id_from_authorization(authorization)
    budget = get_store().get_budget(trace_id, project_id=project_id)
    cap = budget.cap_usd if budget else None
    result = get_service().predict_all(trace_id, project_id=project_id, budget_cap_usd=cap)
    if result is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return result


@router.get("/predict/{trace_id}/eta", response_model=Prediction)
async def predict_eta(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> Prediction:
    return get_service().predict(trace_id, "eta", project_id=project_id_from_authorization(authorization))


@router.get("/predict/{trace_id}/cost", response_model=Prediction)
async def predict_cost(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> Prediction:
    return get_service().predict(trace_id, "cost", project_id=project_id_from_authorization(authorization))


@router.get("/predict/{trace_id}/steps", response_model=Prediction)
async def predict_steps(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> Prediction:
    return get_service().predict(trace_id, "steps", project_id=project_id_from_authorization(authorization))


@router.get("/predict/{trace_id}/offrails", response_model=Prediction)
async def predict_offrails(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> Prediction:
    return get_service().predict(trace_id, "offrails", project_id=project_id_from_authorization(authorization))


# ------------------------------------------------------------- training admin


@router.post("/predict/rebuild")
async def predict_rebuild() -> dict:
    get_service().rebuild()
    return {"ok": True}


# ----------------------------------------------------------------- budgets


@router.post("/budgets", response_model=BudgetStatus)
async def create_budget(
    req: BudgetRequest,
    authorization: str | None = Header(default=None),
) -> BudgetStatus:
    return register_budget(req, project_id=project_id_from_authorization(authorization))


@router.get("/budgets/{trace_id}/status", response_model=BudgetStatus)
async def get_budget_status(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> BudgetStatus:
    st = budget_status(trace_id, project_id=project_id_from_authorization(authorization))
    if st is None:
        raise HTTPException(status_code=404, detail="no budget for trace")
    return st


# ------------------------------------------------------------------ alerts


@router.post("/alerts", response_model=AlertRuleStatus)
async def create_alert(
    rule: AlertRule,
    authorization: str | None = Header(default=None),
) -> AlertRuleStatus:
    try:
        return alerts_mod.register(rule, project_id=project_id_from_authorization(authorization))
    except alerts_mod.ConditionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/alerts/{trace_id}", response_model=list[AlertRuleStatus])
async def list_alerts(
    trace_id: str,
    authorization: str | None = Header(default=None),
) -> list[AlertRuleStatus]:
    return alerts_mod.list_for(trace_id, project_id=project_id_from_authorization(authorization))

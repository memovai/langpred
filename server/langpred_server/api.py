"""HTTP routes for prediction + budget."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .budget import register as register_budget, status as budget_status
from .db import get_store
from .predict import get_service
from .schemas import AgentPrediction, BudgetRequest, BudgetStatus, Prediction


router = APIRouter(prefix="/api/public", tags=["predict"])


@router.get("/predict/{trace_id}", response_model=AgentPrediction)
async def predict_all(trace_id: str) -> AgentPrediction:
    """Omnibus prediction: time, cost, resources, next action, and risk."""
    budget = get_store().get_budget(trace_id)
    cap = budget.cap_usd if budget else None
    result = get_service().predict_all(trace_id, budget_cap_usd=cap)
    if result is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return result


@router.get("/predict/{trace_id}/eta", response_model=Prediction)
async def predict_eta(trace_id: str) -> Prediction:
    return get_service().predict(trace_id, "eta")


@router.get("/predict/{trace_id}/cost", response_model=Prediction)
async def predict_cost(trace_id: str) -> Prediction:
    return get_service().predict(trace_id, "cost")


@router.get("/predict/{trace_id}/steps", response_model=Prediction)
async def predict_steps(trace_id: str) -> Prediction:
    return get_service().predict(trace_id, "steps")


@router.get("/predict/{trace_id}/offrails", response_model=Prediction)
async def predict_offrails(trace_id: str) -> Prediction:
    return get_service().predict(trace_id, "offrails")


# ------------------------------------------------------------- training admin


@router.post("/predict/rebuild")
async def predict_rebuild() -> dict:
    get_service().rebuild()
    return {"ok": True}


# ----------------------------------------------------------------- budgets


@router.post("/budgets", response_model=BudgetStatus)
async def create_budget(req: BudgetRequest) -> BudgetStatus:
    return register_budget(req)


@router.get("/budgets/{trace_id}/status", response_model=BudgetStatus)
async def get_budget_status(trace_id: str) -> BudgetStatus:
    st = budget_status(trace_id)
    if st is None:
        raise HTTPException(status_code=404, detail="no budget for trace")
    return st

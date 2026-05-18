"""
Langfuse-compatible ingestion endpoint.

The Langfuse client batches events into a `{"batch": [...]}` POST. We accept
the same shape, persist each event into the store, and return a 207-style
result map. We *also* fire the predictor / budget evaluator asynchronously so
the SDK can read budget status off the response header.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Response

from .db import get_store
from .budget import evaluate as evaluate_budget
from .schemas import (
    IngestionBatch,
    IngestionEvent,
    IngestionEventResult,
    IngestionResponse,
)
from .settings import SETTINGS


log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/public", tags=["ingestion"])


def _ts_iso(ts: datetime | None) -> str:
    if ts is None:
        return datetime.utcnow().isoformat()
    return ts.isoformat()


def _trace_id_from(ev: IngestionEvent) -> str | None:
    body = ev.body or {}
    if ev.type == "trace-create":
        return body.get("id")
    return body.get("traceId")


def _observation_id_from(ev: IngestionEvent) -> str | None:
    body = ev.body or {}
    return body.get("id")


def _store_event(ev: IngestionEvent) -> IngestionEventResult:
    try:
        trace_id = _trace_id_from(ev)
        observation_id = _observation_id_from(ev)
        get_store().append_event(
            ev_id=ev.id,
            ev_type=ev.type,
            trace_id=trace_id,
            observation_id=observation_id,
            ts=_ts_iso(ev.timestamp),
            body=ev.body or {},
        )
        return IngestionEventResult(id=ev.id, status=201)
    except Exception as exc:  # pragma: no cover
        log.exception("Failed to store ingestion event")
        return IngestionEventResult(id=ev.id, status=500, message=str(exc))


@router.post("/ingestion", response_model=IngestionResponse)
async def ingest(
    batch: IngestionBatch,
    response: Response,
    authorization: str | None = Header(default=None),
) -> IngestionResponse:
    successes: list[IngestionEventResult] = []
    errors: list[IngestionEventResult] = []
    touched_trace_ids: set[str] = set()

    for ev in batch.batch:
        result = _store_event(ev)
        (successes if result.status < 400 else errors).append(result)
        tid = _trace_id_from(ev)
        if tid:
            touched_trace_ids.add(tid)

    # Re-evaluate any budgets attached to touched traces. Emit a "breached"
    # header so the SDK can react on the very next response.
    breached: list[str] = []
    for tid in touched_trace_ids:
        st = evaluate_budget(tid)
        if st and st.breached:
            breached.append(tid)
    if breached:
        response.headers["X-Langpred-Budget"] = "breached"
        response.headers["X-Langpred-Budget-Traces"] = ",".join(breached)

    return IngestionResponse(successes=successes, errors=errors)


# ---- Health + version --------------------------------------------------


health_router = APIRouter(tags=["meta"])


@health_router.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "langpred",
        "version": "0.1.0",
        "traces": get_store().trace_count(),
        "database_url": SETTINGS.database_url,
    }

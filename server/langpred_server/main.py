"""FastAPI entry-point."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncIterator

from fastapi import FastAPI

from . import api as predict_api
from . import ingest as ingest_api
from .predict import get_service
from .settings import SETTINGS


log = logging.getLogger("langpred")
logging.basicConfig(level=SETTINGS.log_level.upper())


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Initial build (empty datasets are fine).
    get_service().rebuild()
    # Background rebuild loop.
    stop = asyncio.Event()

    async def _train_loop() -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=SETTINGS.train_interval_seconds)
            except asyncio.TimeoutError:
                try:
                    get_service().rebuild()
                except Exception:  # pragma: no cover
                    log.exception("predictor rebuild failed")

    task = asyncio.create_task(_train_loop())
    try:
        yield
    finally:
        stop.set()
        await task


app = FastAPI(
    title="Langpred",
    version="0.1.0",
    description=(
        "Langfuse-compatible ingestion + ETA/cost/budget prediction for "
        "agent trajectories."
    ),
    lifespan=lifespan,
)

app.include_router(ingest_api.router)
app.include_router(ingest_api.health_router)
app.include_router(predict_api.router)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": "langpred",
        "version": "0.1.0",
        "endpoints": [
            "POST /api/public/ingestion",
            "GET /api/public/predict/{trace_id}/eta",
            "GET /api/public/predict/{trace_id}/cost",
            "GET /api/public/predict/{trace_id}/steps",
            "GET /api/public/predict/{trace_id}/offrails",
            "POST /api/public/budgets",
            "GET /api/public/budgets/{trace_id}/status",
            "GET /healthz",
        ],
    }

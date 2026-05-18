"""
Pydantic schemas mirroring Langfuse's `/api/public/ingestion` event envelope.

We are intentionally permissive on inputs (every Body field is Optional[]) so
that any Langfuse SDK version that points at us can write without surprises.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---- Bodies ---------------------------------------------------------------


class _Permissive(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TraceBody(_Permissive):
    id: str | None = None
    timestamp: datetime | None = None
    environment: str | None = None
    name: str | None = None
    userId: str | None = None
    sessionId: str | None = None
    release: str | None = None
    version: str | None = None
    input: Any | None = None
    output: Any | None = None
    metadata: Any | None = None
    tags: list[str] | None = None
    public: bool | None = None


class _ObservationBodyBase(_Permissive):
    id: str | None = None
    traceId: str | None = None
    parentObservationId: str | None = None
    name: str | None = None
    startTime: datetime | None = None
    endTime: datetime | None = None
    metadata: Any | None = None
    input: Any | None = None
    output: Any | None = None
    level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None
    statusMessage: str | None = None
    version: str | None = None
    environment: str | None = None


class CreateSpanBody(_ObservationBodyBase):
    pass


class UpdateSpanBody(_ObservationBodyBase):
    pass


class CreateEventBody(_ObservationBodyBase):
    pass


class _Usage(_Permissive):
    input: int | None = None
    output: int | None = None
    total: int | None = None
    unit: str | None = None
    inputCost: float | None = None
    outputCost: float | None = None
    totalCost: float | None = None
    # Legacy Langfuse "usage" sometimes uses prompt/completion names:
    promptTokens: int | None = None
    completionTokens: int | None = None
    totalTokens: int | None = None


class CreateGenerationBody(_ObservationBodyBase):
    completionStartTime: datetime | None = None
    model: str | None = None
    modelParameters: dict[str, Any] | None = None
    usage: _Usage | dict[str, Any] | None = None
    usageDetails: dict[str, Any] | None = None
    costDetails: dict[str, Any] | None = None
    promptName: str | None = None
    promptVersion: int | None = None


class UpdateGenerationBody(CreateGenerationBody):
    pass


class ScoreBody(_Permissive):
    id: str | None = None
    traceId: str | None = None
    observationId: str | None = None
    sessionId: str | None = None
    name: str | None = None
    value: float | str | None = None
    comment: str | None = None
    source: str | None = None
    dataType: str | None = None


class SDKLogBody(_Permissive):
    log: Any | None = None


# ---- Envelope -------------------------------------------------------------


EventType = Literal[
    "trace-create",
    "span-create",
    "span-update",
    "generation-create",
    "generation-update",
    "event-create",
    "observation-create",
    "observation-update",
    "score-create",
    "sdk-log",
]


class IngestionEvent(_Permissive):
    id: str
    timestamp: datetime
    type: EventType
    body: dict[str, Any] = Field(default_factory=dict)
    metadata: Any | None = None


class IngestionBatch(_Permissive):
    batch: list[IngestionEvent]
    metadata: Any | None = None


class IngestionEventResult(BaseModel):
    id: str
    status: int  # 201 success, 400 bad input, 500 server error
    message: str | None = None


class IngestionResponse(BaseModel):
    successes: list[IngestionEventResult]
    errors: list[IngestionEventResult]


# ---- Langpred extension schemas -------------------------------------------


class Prediction(BaseModel):
    trace_id: str
    kind: Literal["eta", "cost", "offrails", "steps"]
    p50: float
    p90: float
    p99: float
    confidence: float = Field(
        ..., description="0..1 — heuristic uncertainty downgrade (lower = wider band)"
    )
    tier: Literal["heuristic", "knn", "gbm"] = "heuristic"
    n_samples: int = 0
    explanation: str | None = None


class BudgetRequest(BaseModel):
    trace_id: str
    cap_usd: float = Field(..., gt=0)
    on_exceed: Literal["kill", "downgrade", "warn"] = "kill"


class BudgetStatus(BaseModel):
    trace_id: str
    cap_usd: float
    on_exceed: Literal["kill", "downgrade", "warn"]
    spent_usd: float
    predicted_remaining_p50_usd: float
    predicted_remaining_p90_usd: float
    breached: bool
    breach_reason: str | None = None

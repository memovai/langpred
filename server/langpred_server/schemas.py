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
    """Single-kind prediction (backward-compatible v0 API).

    Replaced by :class:`AgentPrediction` for new callers, but the per-kind
    endpoints (``/predict/{tid}/{kind}``) still return this shape so existing
    integrations keep working.
    """

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


class _Meta(BaseModel):
    tier: Literal["heuristic", "knn", "gbm"] = "heuristic"
    n_samples: int = 0
    confidence: float = 0.0
    explanation: str | None = None


class TimePrediction(BaseModel):
    """Time-shape estimate. Everything is in seconds.

    - ``total_*`` is wall-time from trace start → trace end.
    - ``remaining_*`` is computed server-side from ``trace.start_ts``, so the
      caller doesn't need to track elapsed time on the client.
    - ``next_step_*`` is how long the *next* observation is expected to take.
    - ``compute_*`` is time spent inside LLM generations; ``io_*`` is the rest
      (tool calls, waiting for external services, etc.) — useful for UIs that
      want to show "LLM thinking" vs "running tools".
    """

    total_seconds_p50: float
    total_seconds_p90: float
    total_seconds_p99: float
    remaining_seconds_p50: float
    remaining_seconds_p90: float
    remaining_seconds_p99: float
    next_step_seconds_p50: float = 0.0
    next_step_seconds_p90: float = 0.0
    compute_seconds_p50: float = 0.0
    io_seconds_p50: float = 0.0
    elapsed_seconds: float = 0.0


class _UsdByModel(BaseModel):
    model: str
    usd_p50: float
    usd_p90: float


class CostPrediction(BaseModel):
    """USD-shape estimate, with per-model breakdown so callers can downgrade.

    ``next_step_usd_*`` lets a planner ask "is the *next* step alone about to
    cost more than the entire run so far?" — a useful spike detector.
    """

    usd_total_p50: float
    usd_total_p90: float
    usd_total_p99: float
    usd_remaining_p50: float
    usd_remaining_p90: float
    usd_remaining_p99: float
    next_step_usd_p50: float = 0.0
    next_step_usd_p90: float = 0.0
    usd_by_model: list[_UsdByModel] = Field(default_factory=list)
    spent_usd: float = 0.0


class _ToolCount(BaseModel):
    tool: str
    p50: float
    p90: float


class ResourcePrediction(BaseModel):
    """Counts: tokens, steps, LLM vs tool calls, and per-tool call frequency.

    ``tool_call_counts`` is the histogram of *additional* tool calls expected
    over the remaining steps, so a rate-limit-aware caller can ask "how many
    more web_search calls before this trace finishes?".
    """

    total_tokens_p50: float
    total_tokens_p90: float
    prompt_tokens_p50: float
    prompt_tokens_p90: float
    completion_tokens_p50: float
    completion_tokens_p90: float
    total_steps_p50: float
    total_steps_p90: float
    steps_remaining_p50: float
    steps_remaining_p90: float
    llm_calls_p50: float = 0.0
    tool_calls_p50: float = 0.0
    tool_call_counts: list[_ToolCount] = Field(default_factory=list)


class _ToolProb(BaseModel):
    tool: str
    probability: float


class NextActionPrediction(BaseModel):
    """The trajectory-conditional prediction — what is the agent about to do?

    - ``next_kind_distribution`` is a probability mass function over
      {generation, tool_call, end}.
    - ``top_next_tools`` lists the most likely tool names with their
      probabilities (only meaningful when next_kind=tool_call).
    - ``likely_next_model`` is the most-common model used at this prefix
      length in the cohort — useful for downgrade routing.
    - ``expected_next_step_*`` are aliases of CostPrediction.next_step_* and
      TimePrediction.next_step_* but co-located so a planner can read one
      object.
    """

    next_kind_distribution: dict[str, float] = Field(default_factory=dict)
    top_next_tools: list[_ToolProb] = Field(default_factory=list)
    likely_next_model: str | None = None
    p_finish_within_one_step: float = 0.0
    expected_next_step_usd_p50: float = 0.0
    expected_next_step_seconds_p50: float = 0.0


class RiskPrediction(BaseModel):
    """Probabilities that something goes wrong before the trace ends.

    Each field is a probability in [0, 1]:

    - ``offrails_risk`` — neighbour traces that ended in error/cancelled.
    - ``loop_risk`` — heuristic: trailing repeat-streak relative to step count.
    - ``context_overflow_risk`` — predicted total prompt tokens vs the
      generation model's context window.
    - ``budget_overshoot_risk`` — populated only if a budget is registered.
    - ``cost_spike_risk`` — probability that one upcoming step will cost
      ≥ 2× the running per-step median.
    """

    offrails_risk: float = 0.0
    loop_risk: float = 0.0
    context_overflow_risk: float = 0.0
    budget_overshoot_risk: float = 0.0
    cost_spike_risk: float = 0.0
    notes: list[str] = Field(default_factory=list)


class AgentPrediction(BaseModel):
    """Omnibus prediction for a single trace.

    The convenience endpoints (``/predict/{tid}/eta`` etc.) return slices of
    this; ``/predict/{tid}`` returns the whole thing. All five sub-predictions
    are populated even when the model has thin data — fields will just have
    zero values and ``meta.tier='heuristic'``.
    """

    trace_id: str
    meta: _Meta
    time: TimePrediction
    cost: CostPrediction
    resources: ResourcePrediction
    next: NextActionPrediction
    risk: RiskPrediction


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

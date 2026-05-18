"""Client-side prediction dataclasses.

Mirror :mod:`langpred_server.schemas` so users can ``isinstance`` check and
use attribute access. The legacy ``EtaPrediction`` / ``CostPrediction`` /
``OffRailsPrediction`` types are preserved (they back the per-kind endpoints)
and the new ``AgentPrediction`` is the omnibus surface returned by
:meth:`langpred.Trace.predict`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ----------------------- legacy per-kind types ----------------------------


@dataclass
class _PredictionBase:
    trace_id: str
    p50: float
    p90: float
    p99: float
    confidence: float
    tier: str
    n_samples: int
    explanation: str = ""

    @classmethod
    def from_response(cls, body: dict[str, Any]) -> "_PredictionBase":
        return cls(
            trace_id=body["trace_id"],
            p50=float(body["p50"]),
            p90=float(body["p90"]),
            p99=float(body["p99"]),
            confidence=float(body["confidence"]),
            tier=body.get("tier", "heuristic"),
            n_samples=int(body.get("n_samples", 0)),
            explanation=body.get("explanation", ""),
        )


@dataclass
class EtaPrediction(_PredictionBase):
    @property
    def seconds_p50(self) -> float:
        return self.p50

    @property
    def seconds_p90(self) -> float:
        return self.p90

    @property
    def seconds_p99(self) -> float:
        return self.p99


@dataclass
class CostPrediction(_PredictionBase):
    @property
    def usd_p50(self) -> float:
        return self.p50

    @property
    def usd_p90(self) -> float:
        return self.p90

    @property
    def usd_p99(self) -> float:
        return self.p99


@dataclass
class OffRailsPrediction(_PredictionBase):
    @property
    def score(self) -> float:
        return self.p50


# --------------------- new omnibus types ----------------------------------


@dataclass
class TimeForecast:
    total_seconds_p50: float
    total_seconds_p90: float
    total_seconds_p99: float
    remaining_seconds_p50: float
    remaining_seconds_p90: float
    remaining_seconds_p99: float
    next_step_seconds_p50: float
    next_step_seconds_p90: float
    compute_seconds_p50: float
    io_seconds_p50: float
    elapsed_seconds: float


@dataclass
class _UsdByModel:
    model: str
    usd_p50: float
    usd_p90: float


@dataclass
class CostForecast:
    usd_total_p50: float
    usd_total_p90: float
    usd_total_p99: float
    usd_remaining_p50: float
    usd_remaining_p90: float
    usd_remaining_p99: float
    next_step_usd_p50: float
    next_step_usd_p90: float
    spent_usd: float
    usd_by_model: list[_UsdByModel] = field(default_factory=list)


@dataclass
class _ToolCount:
    tool: str
    p50: float
    p90: float


@dataclass
class ResourceForecast:
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
    llm_calls_p50: float
    tool_calls_p50: float
    tool_call_counts: list[_ToolCount] = field(default_factory=list)


@dataclass
class _ToolProb:
    tool: str
    probability: float


@dataclass
class NextActionForecast:
    """The *trajectory-conditional* part of the prediction — what is the
    agent about to do?
    """

    next_kind_distribution: dict[str, float]
    top_next_tools: list[_ToolProb]
    likely_next_model: str | None
    p_finish_within_one_step: float
    expected_next_step_usd_p50: float
    expected_next_step_seconds_p50: float

    def most_likely_kind(self) -> str | None:
        if not self.next_kind_distribution:
            return None
        return max(self.next_kind_distribution.items(), key=lambda kv: kv[1])[0]

    def most_likely_tool(self) -> str | None:
        if not self.top_next_tools:
            return None
        return self.top_next_tools[0].tool


@dataclass
class RiskForecast:
    offrails_risk: float
    loop_risk: float
    context_overflow_risk: float
    budget_overshoot_risk: float
    cost_spike_risk: float
    notes: list[str] = field(default_factory=list)

    @property
    def any_high(self) -> bool:
        return any(
            v > 0.5
            for v in (
                self.offrails_risk,
                self.loop_risk,
                self.context_overflow_risk,
                self.budget_overshoot_risk,
                self.cost_spike_risk,
            )
        )


@dataclass
class Meta:
    tier: str
    n_samples: int
    confidence: float
    explanation: str | None = None


@dataclass
class AgentPrediction:
    """Omnibus prediction for one trace."""

    trace_id: str
    meta: Meta
    time: TimeForecast
    cost: CostForecast
    resources: ResourceForecast
    next: NextActionForecast
    risk: RiskForecast

    @classmethod
    def from_response(cls, body: dict[str, Any]) -> "AgentPrediction":
        t = body["time"]
        c = body["cost"]
        r = body["resources"]
        n = body["next"]
        risk = body["risk"]
        m = body["meta"]
        return cls(
            trace_id=body["trace_id"],
            meta=Meta(
                tier=m.get("tier", "heuristic"),
                n_samples=int(m.get("n_samples", 0)),
                confidence=float(m.get("confidence", 0.0)),
                explanation=m.get("explanation"),
            ),
            time=TimeForecast(
                total_seconds_p50=float(t["total_seconds_p50"]),
                total_seconds_p90=float(t["total_seconds_p90"]),
                total_seconds_p99=float(t["total_seconds_p99"]),
                remaining_seconds_p50=float(t["remaining_seconds_p50"]),
                remaining_seconds_p90=float(t["remaining_seconds_p90"]),
                remaining_seconds_p99=float(t["remaining_seconds_p99"]),
                next_step_seconds_p50=float(t.get("next_step_seconds_p50", 0.0)),
                next_step_seconds_p90=float(t.get("next_step_seconds_p90", 0.0)),
                compute_seconds_p50=float(t.get("compute_seconds_p50", 0.0)),
                io_seconds_p50=float(t.get("io_seconds_p50", 0.0)),
                elapsed_seconds=float(t.get("elapsed_seconds", 0.0)),
            ),
            cost=CostForecast(
                usd_total_p50=float(c["usd_total_p50"]),
                usd_total_p90=float(c["usd_total_p90"]),
                usd_total_p99=float(c["usd_total_p99"]),
                usd_remaining_p50=float(c["usd_remaining_p50"]),
                usd_remaining_p90=float(c["usd_remaining_p90"]),
                usd_remaining_p99=float(c["usd_remaining_p99"]),
                next_step_usd_p50=float(c.get("next_step_usd_p50", 0.0)),
                next_step_usd_p90=float(c.get("next_step_usd_p90", 0.0)),
                spent_usd=float(c.get("spent_usd", 0.0)),
                usd_by_model=[
                    _UsdByModel(
                        model=x["model"],
                        usd_p50=float(x["usd_p50"]),
                        usd_p90=float(x["usd_p90"]),
                    )
                    for x in c.get("usd_by_model", [])
                ],
            ),
            resources=ResourceForecast(
                total_tokens_p50=float(r["total_tokens_p50"]),
                total_tokens_p90=float(r["total_tokens_p90"]),
                prompt_tokens_p50=float(r["prompt_tokens_p50"]),
                prompt_tokens_p90=float(r["prompt_tokens_p90"]),
                completion_tokens_p50=float(r["completion_tokens_p50"]),
                completion_tokens_p90=float(r["completion_tokens_p90"]),
                total_steps_p50=float(r["total_steps_p50"]),
                total_steps_p90=float(r["total_steps_p90"]),
                steps_remaining_p50=float(r["steps_remaining_p50"]),
                steps_remaining_p90=float(r["steps_remaining_p90"]),
                llm_calls_p50=float(r.get("llm_calls_p50", 0.0)),
                tool_calls_p50=float(r.get("tool_calls_p50", 0.0)),
                tool_call_counts=[
                    _ToolCount(
                        tool=x["tool"], p50=float(x["p50"]), p90=float(x["p90"])
                    )
                    for x in r.get("tool_call_counts", [])
                ],
            ),
            next=NextActionForecast(
                next_kind_distribution=dict(n.get("next_kind_distribution", {})),
                top_next_tools=[
                    _ToolProb(tool=x["tool"], probability=float(x["probability"]))
                    for x in n.get("top_next_tools", [])
                ],
                likely_next_model=n.get("likely_next_model"),
                p_finish_within_one_step=float(n.get("p_finish_within_one_step", 0.0)),
                expected_next_step_usd_p50=float(n.get("expected_next_step_usd_p50", 0.0)),
                expected_next_step_seconds_p50=float(
                    n.get("expected_next_step_seconds_p50", 0.0)
                ),
            ),
            risk=RiskForecast(
                offrails_risk=float(risk.get("offrails_risk", 0.0)),
                loop_risk=float(risk.get("loop_risk", 0.0)),
                context_overflow_risk=float(risk.get("context_overflow_risk", 0.0)),
                budget_overshoot_risk=float(risk.get("budget_overshoot_risk", 0.0)),
                cost_spike_risk=float(risk.get("cost_spike_risk", 0.0)),
                notes=list(risk.get("notes", [])),
            ),
        )

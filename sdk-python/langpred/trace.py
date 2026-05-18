"""
Trace / Span / Generation / Score primitives — Langfuse-shape.

Each object knows how to:

- create itself on the wire (emits a ``*-create`` event),
- update itself (emits a ``*-update`` event),
- spawn children (``trace.span(...)``, ``trace.generation(...)``),
- call predictions / set budgets (Langpred extensions).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from ._ids import new_id
from .budget import BudgetGuard
from .predict import (
    AgentPrediction,
    CostPrediction,
    EtaPrediction,
    NextActionForecast,
    OffRailsPrediction,
)


if TYPE_CHECKING:
    from .transport import Transport


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _Observation:
    """Shared base for span / generation / event."""

    _create_event: str = "observation-create"
    _update_event: str = "observation-update"
    _kind: str = "span"

    def __init__(
        self,
        transport: "Transport",
        trace_id: str,
        parent_id: str | None = None,
        **kwargs: Any,
    ):
        self.transport = transport
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.id = kwargs.pop("id", None) or new_id()
        self.name = kwargs.pop("name", None)
        self.start_time = kwargs.pop("start_time", None) or _utc_now()
        self.end_time = kwargs.pop("end_time", None)
        self._extra = kwargs
        self._created = False
        self._create()

    def _body(self, **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": self.id,
            "traceId": self.trace_id,
            "parentObservationId": self.parent_id,
            "name": self.name,
            "startTime": self.start_time,
            "endTime": self.end_time,
            **self._extra,
            **overrides,
        }
        return {k: v for k, v in body.items() if v is not None}

    def _create(self) -> None:
        if self._created:
            return
        self.transport.enqueue(self._create_event, self._body())
        self._created = True

    # ----------------------------------------------------------------- API

    def update(self, **fields: Any) -> "_Observation":
        if "end_time" in fields:
            self.end_time = fields.pop("end_time")
        if "name" in fields:
            self.name = fields.pop("name")
        self._extra.update(fields)
        self.transport.enqueue(self._update_event, self._body())
        return self

    def end(self, **fields: Any) -> "_Observation":
        self.end_time = fields.pop("end_time", None) or _utc_now()
        return self.update(**fields)


class Span(_Observation):
    _create_event = "span-create"
    _update_event = "span-update"
    _kind = "span"


class Generation(_Observation):
    _create_event = "generation-create"
    _update_event = "generation-update"
    _kind = "generation"

    def __init__(
        self,
        transport: "Transport",
        trace_id: str,
        parent_id: str | None = None,
        *,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        model_parameters: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        **kwargs: Any,
    ):
        extra: dict[str, Any] = {}
        if model is not None:
            extra["model"] = model
        if usage is not None:
            extra["usage"] = usage
        if model_parameters is not None:
            extra["modelParameters"] = model_parameters
        if input is not None:
            extra["input"] = input
        if output is not None:
            extra["output"] = output
        extra.update(kwargs)
        super().__init__(transport, trace_id, parent_id, **extra)


class Event(_Observation):
    _create_event = "event-create"
    _update_event = "observation-update"
    _kind = "event"


class Score:
    """A scalar score attached to a trace or observation."""

    def __init__(
        self,
        transport: "Transport",
        trace_id: str,
        *,
        name: str,
        value: float | str,
        observation_id: str | None = None,
        comment: str | None = None,
        source: str | None = None,
        data_type: str | None = None,
    ):
        self.transport = transport
        self.trace_id = trace_id
        self.id = new_id()
        body = {
            "id": self.id,
            "traceId": trace_id,
            "observationId": observation_id,
            "name": name,
            "value": value,
            "comment": comment,
            "source": source,
            "dataType": data_type,
        }
        self.transport.enqueue("score-create", {k: v for k, v in body.items() if v is not None})


class Trace:
    """A Langfuse-shape trace + Langpred prediction extensions."""

    def __init__(
        self,
        transport: "Transport",
        *,
        id: str | None = None,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: Any | None = None,
        tags: list[str] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        release: str | None = None,
        version: str | None = None,
    ):
        self.transport = transport
        self.id = id or new_id()
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata
        self.tags = tags
        self._body_extra: dict[str, Any] = {}
        if input is not None:
            self._body_extra["input"] = input
        if output is not None:
            self._body_extra["output"] = output
        if release is not None:
            self._body_extra["release"] = release
        if version is not None:
            self._body_extra["version"] = version
        self.transport.enqueue("trace-create", self._build_body())

    def _build_body(self) -> dict[str, Any]:
        body = {
            "id": self.id,
            "name": self.name,
            "userId": self.user_id,
            "sessionId": self.session_id,
            "metadata": self.metadata,
            "tags": self.tags,
            "timestamp": _utc_now(),
            **self._body_extra,
        }
        return {k: v for k, v in body.items() if v is not None}

    # ------------------------------------------------------------- children

    def span(self, **kwargs: Any) -> Span:
        return Span(self.transport, self.id, **kwargs)

    def generation(self, **kwargs: Any) -> Generation:
        return Generation(self.transport, self.id, **kwargs)

    def event(self, **kwargs: Any) -> Event:
        return Event(self.transport, self.id, **kwargs)

    def score(self, **kwargs: Any) -> Score:
        return Score(self.transport, self.id, **kwargs)

    # -------------------------------------------------------------- mutate

    def update(self, **fields: Any) -> "Trace":
        if "name" in fields:
            self.name = fields["name"]
        if "user_id" in fields:
            self.user_id = fields.pop("user_id")
        if "session_id" in fields:
            self.session_id = fields.pop("session_id")
        if "metadata" in fields:
            self.metadata = fields.pop("metadata")
        if "tags" in fields:
            self.tags = fields.pop("tags")
        for k, v in fields.items():
            self._body_extra[k] = v
        self.transport.enqueue("trace-create", self._build_body())  # idempotent upsert
        return self

    # --------------------------------------------------- prediction methods

    def predict(self) -> AgentPrediction:
        """Full prediction: time + cost + resources + next-action + risk.

        Prefer this over the per-kind methods when you need more than one
        dimension — it's a single round-trip and the sub-predictions are
        internally consistent (e.g. cost & time come from the same neighbour
        cohort).
        """
        body = self.transport.get(f"/api/public/predict/{self.id}")
        return AgentPrediction.from_response(body)

    def predict_eta(self) -> EtaPrediction:
        body = self.transport.get(f"/api/public/predict/{self.id}/eta")
        return EtaPrediction.from_response(body)  # type: ignore[return-value]

    def predict_cost(self) -> CostPrediction:
        body = self.transport.get(f"/api/public/predict/{self.id}/cost")
        return CostPrediction.from_response(body)  # type: ignore[return-value]

    def predict_steps(self) -> EtaPrediction:
        body = self.transport.get(f"/api/public/predict/{self.id}/steps")
        return EtaPrediction.from_response(body)  # type: ignore[return-value]

    def predict_next_action(self) -> NextActionForecast:
        """What is the agent about to do next? (the trajectory-conditional
        thesis — distribution over next step kind, top-k tool names, likely
        model, and a one-step expected cost/time)."""
        return self.predict().next

    def is_off_rails(self, threshold: float = 0.5) -> bool:
        body = self.transport.get(f"/api/public/predict/{self.id}/offrails")
        return float(body.get("p50", 0.0)) >= threshold

    def offrails_score(self) -> OffRailsPrediction:
        body = self.transport.get(f"/api/public/predict/{self.id}/offrails")
        return OffRailsPrediction.from_response(body)  # type: ignore[return-value]

    # -------------------------------------------------------------- budget

    def set_budget(
        self,
        usd: float,
        on_exceed: str = "kill",
    ) -> BudgetGuard:
        self.transport.post(
            "/api/public/budgets",
            {"trace_id": self.id, "cap_usd": usd, "on_exceed": on_exceed},
        )
        return BudgetGuard(
            trace=self, transport=self.transport, cap_usd=usd, on_exceed=on_exceed
        )

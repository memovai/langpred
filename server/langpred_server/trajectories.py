"""
Build :class:`Trajectory` views over the event log.

A Trajectory is the per-trace flattened, time-ordered sequence of steps used
by the predictor. We *don't* store this — it's projected on demand from the
event log so we never lose information after a schema change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from .db import StoredEvent, Store, get_store
from .ml.pricing import price_step


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Step:
    observation_id: str | None
    kind: str  # "span" | "generation" | "event"
    name: str | None
    start_ts: datetime | None
    end_ts: datetime | None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usd: float = 0.0
    level: str | None = None
    status_message: str | None = None
    tool_name: str | None = None  # for spans / events that look like tool calls

    @property
    def latency_ms(self) -> float:
        if self.start_ts and self.end_ts:
            return max(0.0, (self.end_ts - self.start_ts).total_seconds() * 1000.0)
        return 0.0

    @property
    def is_tool(self) -> bool:
        return self.kind in ("span", "event") and (self.tool_name or self.name) is not None

    def descriptor(self) -> str:
        """Stable short token used by the predictor / loop detector."""
        if self.kind == "generation":
            return f"gen:{self.model or '?'}"
        return f"{self.kind}:{self.tool_name or self.name or '?'}"


@dataclass
class Trajectory:
    trace_id: str
    name: str | None
    start_ts: datetime | None
    end_ts: datetime | None
    steps: list[Step] = field(default_factory=list)
    status: str = "open"  # "open" | "ok" | "error" | "cancelled"
    user_id: str | None = None
    session_id: str | None = None

    @property
    def total_usd(self) -> float:
        return sum(s.usd for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def wall_seconds(self) -> float:
        if self.start_ts and self.end_ts:
            return max(0.0, (self.end_ts - self.start_ts).total_seconds())
        return 0.0

    @property
    def is_complete(self) -> bool:
        return self.status in ("ok", "error", "cancelled") and self.end_ts is not None

    def prefix(self, k: int) -> "Trajectory":
        clipped = Trajectory(
            trace_id=self.trace_id,
            name=self.name,
            start_ts=self.start_ts,
            end_ts=self.steps[k - 1].end_ts if k and self.steps[: k][-1:] else None,
            steps=self.steps[:k],
            status="open",
            user_id=self.user_id,
            session_id=self.session_id,
        )
        return clipped


def _extract_usage(body: dict[str, Any]) -> tuple[int, int, int]:
    """Return (prompt_tokens, completion_tokens, total_tokens) from a Langfuse body."""
    usage = body.get("usage") or {}
    details = body.get("usageDetails") or {}
    pt = (
        usage.get("input")
        or usage.get("promptTokens")
        or details.get("input")
        or details.get("promptTokens")
        or 0
    )
    ct = (
        usage.get("output")
        or usage.get("completionTokens")
        or details.get("output")
        or details.get("completionTokens")
        or 0
    )
    tt = (
        usage.get("total")
        or usage.get("totalTokens")
        or details.get("total")
        or (int(pt or 0) + int(ct or 0))
        or 0
    )
    return int(pt or 0), int(ct or 0), int(tt or 0)


def _extract_cost(body: dict[str, Any], pt: int, ct: int) -> float:
    usage = body.get("usage") or {}
    cd = body.get("costDetails") or {}
    explicit = (
        usage.get("totalCost")
        or cd.get("total")
        or cd.get("totalCost")
    )
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass
    # Otherwise price by model.
    return price_step(body.get("model"), pt, ct)


def build_trajectory(events: Iterable[StoredEvent]) -> Trajectory | None:
    """Project a sequence of stored events into a :class:`Trajectory`."""
    evs = sorted(events, key=lambda e: e.timestamp)
    if not evs:
        return None

    trace_id = next((e.trace_id for e in evs if e.trace_id), None)
    if not trace_id:
        return None

    traj = Trajectory(trace_id=trace_id, name=None, start_ts=None, end_ts=None)
    observations: dict[str, Step] = {}

    for ev in evs:
        t = ev.type
        body = ev.body or {}
        if t == "trace-create":
            traj.name = body.get("name") or traj.name
            traj.user_id = body.get("userId") or traj.user_id
            traj.session_id = body.get("sessionId") or traj.session_id
            ts = _parse_dt(body.get("timestamp")) or _parse_dt(ev.timestamp)
            if ts and (traj.start_ts is None or ts < traj.start_ts):
                traj.start_ts = ts
            # Trace-level completion: presence of output suggests "done"
            if body.get("output") is not None and traj.status == "open":
                traj.status = "ok"
            continue

        if t == "sdk-log":
            continue

        if t == "score-create":
            # Could be used to influence "off rails" — skip in trajectory shape.
            continue

        oid = body.get("id") or ev.observation_id
        if t.endswith("-create"):
            kind = "generation" if t.startswith("generation") else (
                "span" if t.startswith("span") else "event"
            )
            step = observations.get(oid) or Step(
                observation_id=oid,
                kind=kind,
                name=body.get("name"),
                start_ts=_parse_dt(body.get("startTime")) or _parse_dt(ev.timestamp),
                end_ts=_parse_dt(body.get("endTime")),
            )
            step.kind = kind
            step.name = body.get("name") or step.name
            step.model = body.get("model") or step.model
            step.level = body.get("level") or step.level
            step.status_message = body.get("statusMessage") or step.status_message
            step.tool_name = body.get("name") if kind != "generation" else step.tool_name
            pt, ct, tt = _extract_usage(body)
            if pt or ct or tt:
                step.prompt_tokens = pt or step.prompt_tokens
                step.completion_tokens = ct or step.completion_tokens
                step.total_tokens = tt or step.total_tokens
                step.usd = _extract_cost(body, pt, ct) or step.usd
            observations[oid] = step
            if step.start_ts and (traj.start_ts is None or step.start_ts < traj.start_ts):
                traj.start_ts = step.start_ts
            if step.end_ts and (traj.end_ts is None or step.end_ts > traj.end_ts):
                traj.end_ts = step.end_ts
        elif t.endswith("-update") or t == "observation-update":
            step = observations.get(oid)
            if step is None:
                step = Step(
                    observation_id=oid,
                    kind="generation" if "generation" in t else "span",
                    name=body.get("name"),
                    start_ts=_parse_dt(body.get("startTime")),
                    end_ts=_parse_dt(body.get("endTime")),
                )
                observations[oid] = step
            if body.get("endTime"):
                step.end_ts = _parse_dt(body.get("endTime")) or step.end_ts
            if body.get("model"):
                step.model = body.get("model")
            pt, ct, tt = _extract_usage(body)
            if pt or ct or tt:
                step.prompt_tokens = max(step.prompt_tokens, pt)
                step.completion_tokens = max(step.completion_tokens, ct)
                step.total_tokens = max(step.total_tokens, tt)
                step.usd = max(step.usd, _extract_cost(body, pt, ct))
            if body.get("level") == "ERROR" and traj.status == "open":
                traj.status = "error"
            if step.end_ts and (traj.end_ts is None or step.end_ts > traj.end_ts):
                traj.end_ts = step.end_ts

    # Order steps by start_ts for the predictor's prefix view.
    traj.steps = sorted(observations.values(), key=lambda s: s.start_ts or datetime.min)
    return traj


def get_trajectory(trace_id: str, store: Store | None = None) -> Trajectory | None:
    store = store or get_store()
    events = store.events_for_trace(trace_id)
    return build_trajectory(events)


def all_trajectories(store: Store | None = None) -> list[Trajectory]:
    store = store or get_store()
    out: list[Trajectory] = []
    for tid in store.all_trace_ids():
        traj = build_trajectory(store.events_for_trace(tid))
        if traj is not None:
            out.append(traj)
    return out

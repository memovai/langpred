"""
Alert rules — fire a webhook when a condition over the AgentPrediction goes
true. Lightweight rules engine, not a full DSL.

Condition format::

    "<dotted.path> <op> <number>"

with op in ``> >= < <= == !=`` and path being a dotted field of
:class:`langpred_server.schemas.AgentPrediction`. Examples::

    cost.usd_total_p50 > 0.5
    risk.loop_risk > 0.7
    resources.steps_remaining_p50 > 20
    time.remaining_seconds_p90 > 300

Rules are re-evaluated on every ingestion. A rule fires its webhook at most
once per ``min_interval_seconds`` to avoid flooding.
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from .db import AlertRuleRecord, get_store
from .predict import get_service
from .schemas import AgentPrediction, AlertRule, AlertRuleStatus


log = logging.getLogger("langpred.alerts")

_COND_RE = re.compile(
    r"^\s*([A-Za-z_][\w.]*)\s*(>=|<=|==|!=|>|<)\s*([-+]?\d*\.?\d+)\s*$"
)


class ConditionError(ValueError):
    pass


def parse_condition(s: str) -> tuple[str, str, float]:
    m = _COND_RE.match(s or "")
    if not m:
        raise ConditionError(
            f"can't parse condition {s!r} — expected 'path op number' "
            f"(e.g. 'cost.usd_total_p50 > 0.5')"
        )
    return m.group(1), m.group(2), float(m.group(3))


def field_value(pred: AgentPrediction, path: str) -> float:
    obj: Any = pred
    for part in path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            raise ConditionError(f"unknown field: {path}")
        if obj is None:
            return 0.0
    try:
        return float(obj)
    except (TypeError, ValueError) as e:
        raise ConditionError(f"non-numeric field at {path}: {obj!r}") from e


def evaluate_condition(pred: AgentPrediction, condition: str) -> tuple[bool, float, float]:
    """Return ``(matched, value, threshold)`` — `matched` is True when the
    condition is currently satisfied. We always return the raw value so the
    webhook payload can include it."""
    path, op, threshold = parse_condition(condition)
    value = field_value(pred, path)
    if op == ">":
        return value > threshold, value, threshold
    if op == ">=":
        return value >= threshold, value, threshold
    if op == "<":
        return value < threshold, value, threshold
    if op == "<=":
        return value <= threshold, value, threshold
    if op == "==":
        return value == threshold, value, threshold
    if op == "!=":
        return value != threshold, value, threshold
    raise ConditionError(f"unknown op {op}")


# ---------------------------------------------------------------- registration


def register(rule: AlertRule) -> AlertRuleStatus:
    # Validate syntactically up-front so we fail loud at registration time.
    parse_condition(rule.condition)
    rec = AlertRuleRecord(
        id=rule.id or str(uuid.uuid4()),
        trace_id=rule.trace_id,
        condition=rule.condition,
        webhook_url=rule.webhook_url,
        min_interval_seconds=rule.min_interval_seconds,
    )
    get_store().add_alert(rec)
    return _to_status(rec)


def list_for(trace_id: str) -> list[AlertRuleStatus]:
    return [_to_status(r) for r in get_store().alerts_for(trace_id)]


def _to_status(r: AlertRuleRecord) -> AlertRuleStatus:
    return AlertRuleStatus(
        id=r.id,
        trace_id=r.trace_id,
        condition=r.condition,
        webhook_url=r.webhook_url,
        last_fired_at=r.last_fired_at,
        fire_count=r.fire_count,
        last_value=r.last_value,
    )


# -------------------------------------------------------------- evaluation


def evaluate_for_trace(trace_id: str) -> list[str]:
    """Evaluate all rules attached to a trace. Fire webhooks where due.
    Returns the list of alert rule ids that fired during this call."""
    store = get_store()
    rules = store.alerts_for(trace_id)
    if not rules:
        return []
    pred = get_service().predict_all(trace_id)
    if pred is None:
        return []
    fired: list[str] = []
    now = datetime.now(timezone.utc)
    for r in rules:
        try:
            matched, value, threshold = evaluate_condition(pred, r.condition)
        except ConditionError as e:
            log.warning("alert %s has bad condition: %s", r.id, e)
            continue
        r.last_value = value
        if not matched:
            store.update_alert(r)
            continue
        if r.last_fired_at:
            try:
                last = datetime.fromisoformat(r.last_fired_at)
                if (now - last).total_seconds() < r.min_interval_seconds:
                    store.update_alert(r)
                    continue
            except ValueError:
                pass
        # Fire the webhook.
        _fire_webhook(r, pred, value, threshold)
        r.last_fired_at = now.isoformat()
        r.fire_count += 1
        store.update_alert(r)
        fired.append(r.id)
    return fired


# Use a thread pool so webhook latency never blocks ingestion. Best-effort —
# if the pool is full we drop the fire (alerts are advisory, not transactional).
_POOL_LOCK = threading.Lock()
_POOL: list[threading.Thread] = []
_POOL_MAX = 8


def _fire_webhook(
    r: AlertRuleRecord, pred: AgentPrediction, value: float, threshold: float
) -> None:
    payload = {
        "trace_id": r.trace_id,
        "rule_id": r.id,
        "condition": r.condition,
        "value": value,
        "threshold": threshold,
        "prediction": pred.model_dump(mode="json"),
        "fired_at": datetime.now(timezone.utc).isoformat(),
    }

    def _post() -> None:
        try:
            with httpx.Client(timeout=5.0) as c:
                c.post(r.webhook_url, json=payload)
        except Exception:
            log.exception("alert webhook %s failed for rule %s", r.webhook_url, r.id)

    with _POOL_LOCK:
        _POOL[:] = [t for t in _POOL if t.is_alive()]
        if len(_POOL) >= _POOL_MAX:
            log.warning("alert pool full; dropping fire for rule %s", r.id)
            return
        t = threading.Thread(target=_post, daemon=True)
        t.start()
        _POOL.append(t)

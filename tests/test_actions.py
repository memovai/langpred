"""Tests for the pre-emptive action menu:
- forecast (reject-upfront / route-at-start)
- alert_when (webhook fires on condition)
- on_scope_reduce (callback fires when budget+scope_reduce trips)
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _now():
    return datetime.now(timezone.utc)


def _seed_completed(client, name: str, n: int):
    base = _now()
    events = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": base.isoformat(),
            "type": "trace-create",
            "body": {"id": "trace-" + uuid.uuid4().hex, "name": name,
                     "timestamp": base.isoformat()},
        }
    ]
    return events


def _seed_trace(client, name: str, n_steps: int, completed: bool = True,
                model: str = "claude-sonnet-4-6"):
    tid = "trace-" + uuid.uuid4().hex
    base = _now()
    events = [{
        "id": str(uuid.uuid4()),
        "timestamp": base.isoformat(),
        "type": "trace-create",
        "body": {"id": tid, "name": name, "timestamp": base.isoformat()},
    }]
    for k in range(n_steps):
        start = (base + timedelta(seconds=k)).isoformat()
        end = (base + timedelta(seconds=k + 1)).isoformat()
        events.append({
            "id": str(uuid.uuid4()),
            "timestamp": start,
            "type": "generation-create",
            "body": {
                "id": "obs-" + uuid.uuid4().hex,
                "traceId": tid,
                "name": f"gen_{k}",
                "model": model,
                "startTime": start, "endTime": end,
                "usage": {"input": 600, "output": 200, "total": 800},
            },
        })
    if completed:
        events.append({
            "id": str(uuid.uuid4()),
            "timestamp": (base + timedelta(seconds=n_steps + 1)).isoformat(),
            "type": "trace-create",
            "body": {"id": tid, "name": name, "output": "done"},
        })
    client.post("/api/public/ingestion", json={"batch": events})
    return tid


# --------------------------------------------------------------- forecast


def test_forecast_returns_cohort_prediction(client):
    """forecast() works without any trace_id — it's a pre-trace decision tool."""
    # Seed enough completed traces of the named shape.
    for _ in range(10):
        _seed_trace(client, "shopping_agent", n_steps=5)
    client.post("/api/public/predict/rebuild")

    r = client.post("/api/public/forecast", json={"trace_name": "shopping_agent"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Cohort-based forecast: shows what a typical run looks like.
    assert body["cost"]["usd_total_p50"] > 0
    assert body["resources"]["total_steps_p50"] > 0
    assert body["meta"]["n_samples"] == 10
    # No KV cache yet => no "spent" or "elapsed".
    assert body["cost"]["spent_usd"] == 0.0
    assert body["time"]["elapsed_seconds"] == 0.0


def test_forecast_unknown_shape_falls_back_to_global(client):
    for _ in range(5):
        _seed_trace(client, "agent_a", n_steps=4)
    client.post("/api/public/predict/rebuild")

    r = client.post("/api/public/forecast", json={"trace_name": "unseen_shape"})
    assert r.status_code == 200
    body = r.json()
    # Falls back to the global cohort (agent_a traces).
    assert body["meta"]["n_samples"] == 5


# ------------------------------------------------------------ alert webhook


class _CaptureHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        self.__class__.received.append(json.loads(raw))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *a, **kw):
        pass  # silence


def _start_capture_server() -> tuple[str, ThreadingHTTPServer]:
    _CaptureHandler.received = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{port}", srv


def test_alert_webhook_fires_on_condition(client):
    url, srv = _start_capture_server()
    try:
        # Seed cohort so the predictor returns non-zero numbers.
        for _ in range(8):
            _seed_trace(client, "agent", n_steps=5)
        client.post("/api/public/predict/rebuild")

        # Register a rule that should match (predicted cost > $0 once we have
        # a partial trace).
        tid = _seed_trace(client, "agent", n_steps=2, completed=False)
        r = client.post("/api/public/alerts", json={
            "trace_id": tid,
            "condition": "cost.usd_total_p50 > 0.0",
            "webhook_url": url,
            "min_interval_seconds": 0,
        })
        assert r.status_code == 200, r.text

        # Push one more event to trigger evaluation.
        client.post("/api/public/ingestion", json={"batch": [{
            "id": str(uuid.uuid4()),
            "timestamp": _now().isoformat(),
            "type": "generation-create",
            "body": {
                "id": "obs-" + uuid.uuid4().hex,
                "traceId": tid,
                "name": "trigger",
                "model": "claude-sonnet-4-6",
                "startTime": _now().isoformat(),
                "endTime": _now().isoformat(),
                "usage": {"input": 100, "output": 50, "total": 150},
            },
        }]})

        # Webhook fires asynchronously — wait briefly.
        for _ in range(20):
            if _CaptureHandler.received:
                break
            time.sleep(0.05)
        assert _CaptureHandler.received, "webhook didn't fire"
        payload = _CaptureHandler.received[0]
        assert payload["trace_id"] == tid
        assert payload["condition"] == "cost.usd_total_p50 > 0.0"
        assert payload["value"] > 0
        assert "prediction" in payload
    finally:
        srv.shutdown()


def test_alert_rejects_bad_condition(client):
    r = client.post("/api/public/alerts", json={
        "trace_id": "tid",
        "condition": "this is not parseable",
        "webhook_url": "http://example.com",
    })
    assert r.status_code == 400


# ------------------------------------------------------ scope-reduce action


def test_scope_reduce_header_set_on_breach(client):
    tid = _seed_trace(client, "spendy", n_steps=1, completed=False)
    client.post("/api/public/budgets", json={
        "trace_id": tid, "cap_usd": 0.000001, "on_exceed": "scope_reduce",
    })
    # Trigger evaluation via another ingestion event.
    r = client.post("/api/public/ingestion", json={"batch": [{
        "id": str(uuid.uuid4()),
        "timestamp": _now().isoformat(),
        "type": "generation-create",
        "body": {
            "id": "obs-" + uuid.uuid4().hex,
            "traceId": tid,
            "name": "expensive",
            "model": "claude-opus-4-7",
            "startTime": _now().isoformat(),
            "endTime": _now().isoformat(),
            "usage": {"input": 4000, "output": 2000, "total": 6000},
        },
    }]})
    assert r.status_code == 200
    # We expect the scope-reduce header (NOT the kill/breached header).
    assert "x-langpred-scope-reduce" in {k.lower() for k in r.headers}
    assert tid in r.headers.get("x-langpred-scope-reduce",
                                  r.headers.get("X-Langpred-Scope-Reduce", ""))
    # And NOT the breached header — that's only for on_exceed=kill.
    assert "x-langpred-budget" not in {k.lower() for k in r.headers}


def test_budget_rejects_downgrade_action(client):
    # We intentionally removed mid-trace "downgrade" from the menu. Verify
    # the schema rejects it.
    r = client.post("/api/public/budgets", json={
        "trace_id": "tid", "cap_usd": 0.5, "on_exceed": "downgrade",
    })
    assert r.status_code == 422  # pydantic validation


# --------------------------------------------------------- condition parser


def test_condition_parser_basic():
    from langpred_server.alerts import parse_condition, ConditionError
    assert parse_condition("cost.usd_total_p50 > 0.5") == ("cost.usd_total_p50", ">", 0.5)
    assert parse_condition("risk.loop_risk>=0.7") == ("risk.loop_risk", ">=", 0.7)
    assert parse_condition("time.remaining_seconds_p90 < 30") == (
        "time.remaining_seconds_p90", "<", 30.0
    )

    import pytest
    with pytest.raises(ConditionError):
        parse_condition("definitely not a condition")

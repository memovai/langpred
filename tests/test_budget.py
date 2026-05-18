"""Budget flow: register a budget, ingest expensive steps, budget breaches."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _expensive_step(tid: str, model: str = "claude-opus-4-7") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": _now(),
        "type": "generation-create",
        "body": {
            "id": "obs-" + uuid.uuid4().hex,
            "traceId": tid,
            "name": "step",
            "model": model,
            "startTime": _now(),
            "endTime": _now(),
            "usage": {"input": 5000, "output": 3000, "total": 8000},
        },
    }


def test_budget_breaches_when_predicted_total_exceeds_cap(client):
    tid = "trace-" + uuid.uuid4().hex
    # Trace + budget.
    client.post("/api/public/ingestion", json={"batch": [
        {"id": str(uuid.uuid4()), "timestamp": _now(),
         "type": "trace-create", "body": {"id": tid, "name": "spendy"}}
    ]})
    st = client.post(
        "/api/public/budgets",
        json={"trace_id": tid, "cap_usd": 0.05, "on_exceed": "kill"},
    )
    assert st.status_code == 200
    assert not st.json()["breached"]

    # Ingest some expensive steps until the budget trips.
    breached = False
    for _ in range(8):
        r = client.post("/api/public/ingestion", json={"batch": [_expensive_step(tid)]})
        assert r.status_code == 200
        if r.headers.get("x-langpred-budget") == "breached":
            breached = True
            break
    assert breached, "budget should breach on expensive opus-4-7 steps"

    status = client.get(f"/api/public/budgets/{tid}/status").json()
    assert status["breached"]
    assert status["spent_usd"] > 0


def test_budget_does_not_breach_for_cheap_run(client):
    tid = "trace-" + uuid.uuid4().hex
    client.post("/api/public/ingestion", json={"batch": [
        {"id": str(uuid.uuid4()), "timestamp": _now(),
         "type": "trace-create", "body": {"id": tid, "name": "cheap"}}
    ]})
    client.post(
        "/api/public/budgets",
        json={"trace_id": tid, "cap_usd": 5.0, "on_exceed": "kill"},
    )
    # A few cheap haiku steps.
    for _ in range(3):
        ev = _expensive_step(tid, model="claude-haiku-4-5")
        ev["body"]["usage"] = {"input": 200, "output": 80, "total": 280}
        client.post("/api/public/ingestion", json={"batch": [ev]})
    status = client.get(f"/api/public/budgets/{tid}/status").json()
    assert not status["breached"]
    assert status["spent_usd"] < 5.0

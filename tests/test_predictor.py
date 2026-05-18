"""End-to-end test of the prediction path: ingest some completed traces,
rebuild, then ask for ETA / cost on a fresh partial trace."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone


def _now():
    return datetime.now(timezone.utc)


def _seed_completed_trace(client, name: str, n_steps: int, model: str = "claude-sonnet-4-6"):
    tid = "trace-" + uuid.uuid4().hex
    base = _now()
    events = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": base.isoformat(),
            "type": "trace-create",
            "body": {"id": tid, "name": name, "timestamp": base.isoformat()},
        }
    ]
    for k in range(n_steps):
        start = (base + timedelta(seconds=k)).isoformat()
        end = (base + timedelta(seconds=k + 1)).isoformat()
        gid = "obs-" + uuid.uuid4().hex
        events.append({
            "id": str(uuid.uuid4()),
            "timestamp": start,
            "type": "generation-create",
            "body": {
                "id": gid, "traceId": tid, "name": f"step_{k}",
                "model": model,
                "startTime": start, "endTime": end,
                "usage": {"input": 500, "output": 200, "total": 700},
            },
        })
    # Mark the trace as completed.
    events.append({
        "id": str(uuid.uuid4()),
        "timestamp": (base + timedelta(seconds=n_steps + 1)).isoformat(),
        "type": "trace-create",
        "body": {"id": tid, "name": name, "output": "done"},
    })
    r = client.post("/api/public/ingestion", json={"batch": events})
    assert r.status_code == 200, r.text
    return tid


def test_predictor_returns_band_after_seeding(client):
    for _ in range(8):
        _seed_completed_trace(client, name="agent", n_steps=6)
    client.post("/api/public/predict/rebuild")

    # Now ingest a partial trace (3 steps) and ask for cost.
    partial = _seed_completed_trace(client, name="agent", n_steps=3)
    r = client.get(f"/api/public/predict/{partial}/cost")
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] == partial
    assert body["kind"] == "cost"
    assert body["p50"] >= 0
    assert body["p90"] >= body["p50"]
    assert body["p99"] >= body["p90"]
    assert body["tier"] in ("heuristic", "knn", "gbm")


def test_predictor_eta_and_offrails(client):
    for _ in range(6):
        _seed_completed_trace(client, name="research", n_steps=5)
    client.post("/api/public/predict/rebuild")

    partial = _seed_completed_trace(client, name="research", n_steps=2)
    eta = client.get(f"/api/public/predict/{partial}/eta").json()
    offrails = client.get(f"/api/public/predict/{partial}/offrails").json()
    assert eta["p50"] >= 0
    assert 0 <= offrails["p50"] <= 1

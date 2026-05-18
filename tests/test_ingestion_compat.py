"""
Verify the server accepts the *exact* Langfuse `/api/public/ingestion`
batch shape: list of events with `id`, `timestamp`, `type`, `body`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def _ev(type_: str, body: dict) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": type_,
        "body": body,
    }


def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["service"] == "langpred"


def test_accepts_trace_create(client):
    tid = "trace-" + uuid.uuid4().hex
    batch = {
        "batch": [
            _ev("trace-create", {"id": tid, "name": "my_agent", "userId": "u1"}),
        ]
    }
    r = client.post("/api/public/ingestion", json=batch)
    assert r.status_code == 200
    body = r.json()
    assert len(body["successes"]) == 1
    assert not body["errors"]


def test_accepts_full_langfuse_batch(client):
    tid = "trace-" + uuid.uuid4().hex
    span_id = "obs-" + uuid.uuid4().hex
    gen_id = "obs-" + uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    batch = {
        "batch": [
            _ev("trace-create", {"id": tid, "name": "my_agent"}),
            _ev("span-create", {
                "id": span_id, "traceId": tid, "name": "think",
                "startTime": now,
            }),
            _ev("generation-create", {
                "id": gen_id, "traceId": tid, "name": "llm-call",
                "model": "claude-sonnet-4-6",
                "startTime": now,
                "usage": {"input": 100, "output": 200, "total": 300},
            }),
            _ev("generation-update", {
                "id": gen_id, "traceId": tid,
                "endTime": now,
                "output": "hello",
            }),
            _ev("span-update", {
                "id": span_id, "traceId": tid, "endTime": now,
            }),
            _ev("score-create", {
                "traceId": tid, "name": "quality", "value": 0.9,
            }),
        ]
    }
    r = client.post("/api/public/ingestion", json=batch)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["successes"]) == 6
    assert not body["errors"]


def test_partial_failures_dont_kill_batch(client):
    """Even if one event is malformed (missing type), the server should still
    process the others. We send one perfect event + one with garbage type."""
    tid = "trace-" + uuid.uuid4().hex
    batch = {
        "batch": [
            _ev("trace-create", {"id": tid, "name": "ok"}),
        ]
    }
    r = client.post("/api/public/ingestion", json=batch)
    assert r.status_code == 200

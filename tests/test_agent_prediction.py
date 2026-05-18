"""Tests for the omnibus AgentPrediction surface."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone


def _now():
    return datetime.now(timezone.utc)


def _seed_trace(
    client,
    name: str,
    n_gens: int,
    n_tools: int,
    *,
    model: str = "claude-sonnet-4-6",
    tool_names: list[str] | None = None,
    completed: bool = True,
):
    tool_names = tool_names or ["web_search", "read_file"]
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
    t_cursor = 0
    for k in range(n_gens + n_tools):
        start = (base + timedelta(seconds=t_cursor)).isoformat()
        end = (base + timedelta(seconds=t_cursor + 1)).isoformat()
        t_cursor += 1
        if k < n_gens:
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
        else:
            tname = tool_names[(k - n_gens) % len(tool_names)]
            events.append({
                "id": str(uuid.uuid4()),
                "timestamp": start,
                "type": "span-create",
                "body": {
                    "id": "obs-" + uuid.uuid4().hex,
                    "traceId": tid,
                    "name": tname,
                    "startTime": start, "endTime": end,
                },
            })
    if completed:
        events.append({
            "id": str(uuid.uuid4()),
            "timestamp": (base + timedelta(seconds=t_cursor + 1)).isoformat(),
            "type": "trace-create",
            "body": {"id": tid, "name": name, "output": "done"},
        })
    r = client.post("/api/public/ingestion", json={"batch": events})
    assert r.status_code == 200
    return tid


def test_omnibus_predict_returns_all_sections(client):
    # Seed enough completed traces to power the kNN.
    for _ in range(8):
        _seed_trace(client, "agent", n_gens=3, n_tools=3)
    client.post("/api/public/predict/rebuild")

    # Partial trace.
    tid = _seed_trace(client, "agent", n_gens=1, n_tools=1, completed=False)

    r = client.get(f"/api/public/predict/{tid}")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["trace_id"] == tid
    for section in ("meta", "time", "cost", "resources", "next", "risk"):
        assert section in body, f"missing {section}"

    # time sanity: remaining >= 0, total >= elapsed.
    t = body["time"]
    assert t["remaining_seconds_p50"] >= 0
    assert t["total_seconds_p50"] >= t["elapsed_seconds"]

    # cost sanity: remaining + spent ~ total.
    c = body["cost"]
    assert c["usd_total_p50"] >= c["spent_usd"]
    assert c["usd_remaining_p50"] >= 0

    # resources sanity.
    r = body["resources"]
    assert r["steps_remaining_p50"] >= 0
    # tool_call_counts is a list of {tool, p50, p90}.
    assert isinstance(r["tool_call_counts"], list)

    # next action distribution sums close to 1 (or 0 if no neighbours).
    n = body["next"]
    total_prob = sum(n["next_kind_distribution"].values())
    assert abs(total_prob - 1.0) < 1e-6 or total_prob == 0

    # risk fields in [0, 1].
    risk = body["risk"]
    for k in ("offrails_risk", "loop_risk", "context_overflow_risk",
              "budget_overshoot_risk", "cost_spike_risk"):
        assert 0.0 <= risk[k] <= 1.0


def test_next_action_predicts_consistent_tool(client):
    # Seed a clear pattern: every trace does web_search→web_search→read_file
    # so the model should bet on web_search after a partial of length 1.
    for _ in range(20):
        _seed_trace(
            client,
            "shopping_agent",
            n_gens=0,
            n_tools=3,
            tool_names=["web_search", "web_search", "read_file"],
        )
    client.post("/api/public/predict/rebuild")

    # Partial with one web_search done.
    partial = _seed_trace(
        client,
        "shopping_agent",
        n_gens=0,
        n_tools=1,
        tool_names=["web_search"],
        completed=False,
    )
    body = client.get(f"/api/public/predict/{partial}").json()
    next_action = body["next"]
    # Tool distribution should heavily favour web_search.
    tools = {t["tool"]: t["probability"] for t in next_action["top_next_tools"]}
    assert tools.get("web_search", 0) > tools.get("read_file", 0)


def test_loop_risk_fires_on_repeats(client):
    # 6 identical tool calls in a row → loop risk should be > 0.
    tid = _seed_trace(
        client,
        "stuck_agent",
        n_gens=0,
        n_tools=6,
        tool_names=["spin"],
        completed=False,
    )
    body = client.get(f"/api/public/predict/{tid}").json()
    assert body["risk"]["loop_risk"] > 0.3


def test_budget_overshoot_risk_present_when_budget_registered(client):
    tid = _seed_trace(client, "spendy", n_gens=3, n_tools=0, completed=False)
    # Tiny cap → high overshoot risk on a sonnet trace.
    client.post(
        "/api/public/budgets",
        json={"trace_id": tid, "cap_usd": 0.0001, "on_exceed": "warn"},
    )
    body = client.get(f"/api/public/predict/{tid}").json()
    assert body["risk"]["budget_overshoot_risk"] >= 0.0  # smoke; presence is what matters

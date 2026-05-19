"""Regression tests for production-facing architecture improvements."""
from __future__ import annotations

import base64
import uuid
from datetime import datetime, timedelta, timezone


def _auth(project_id: str) -> dict[str, str]:
    raw = base64.b64encode(f"{project_id}:secret".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


def _now():
    return datetime.now(timezone.utc)


def _seed_trace(
    client,
    name: str,
    n_steps: int,
    *,
    project_id: str = "default",
    completed: bool = True,
    metadata: dict | None = None,
    tool_names: list[str] | None = None,
    total_cost: float | None = None,
) -> str:
    tid = "trace-" + uuid.uuid4().hex
    base = _now()
    headers = _auth(project_id) if project_id != "default" else None
    events = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": base.isoformat(),
            "type": "trace-create",
            "body": {
                "id": tid,
                "name": name,
                "timestamp": base.isoformat(),
                "metadata": metadata,
            },
        }
    ]
    for k in range(n_steps):
        start = (base + timedelta(seconds=k)).isoformat()
        end = (base + timedelta(seconds=k + 1)).isoformat()
        if tool_names:
            events.append(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": start,
                    "type": "span-create",
                    "body": {
                        "id": "obs-" + uuid.uuid4().hex,
                        "traceId": tid,
                        "name": tool_names[k % len(tool_names)],
                        "startTime": start,
                        "endTime": end,
                    },
                }
            )
            continue
        usage = {"input": 100, "output": 50, "total": 150}
        if total_cost is not None:
            usage["totalCost"] = total_cost
        events.append(
            {
                "id": str(uuid.uuid4()),
                "timestamp": start,
                "type": "generation-create",
                "body": {
                    "id": "obs-" + uuid.uuid4().hex,
                    "traceId": tid,
                    "name": f"gen_{k}",
                    "model": "claude-sonnet-4-6",
                    "startTime": start,
                    "endTime": end,
                    "usage": usage,
                },
            }
        )
    if completed:
        events.append(
            {
                "id": str(uuid.uuid4()),
                "timestamp": (base + timedelta(seconds=n_steps + 1)).isoformat(),
                "type": "trace-create",
                "body": {"id": tid, "name": name, "output": "done"},
            }
        )
    r = client.post("/api/public/ingestion", json={"batch": events}, headers=headers)
    assert r.status_code == 200, r.text
    return tid


def test_prediction_cohorts_are_project_isolated(client):
    for _ in range(6):
        _seed_trace(client, "agent", 2, project_id="pk-a")
        _seed_trace(client, "agent", 9, project_id="pk-b")
    client.post("/api/public/predict/rebuild")

    a = client.post(
        "/api/public/forecast",
        json={"trace_name": "agent"},
        headers=_auth("pk-a"),
    ).json()
    b = client.post(
        "/api/public/forecast",
        json={"trace_name": "agent"},
        headers=_auth("pk-b"),
    ).json()

    assert a["meta"]["n_samples"] == 6
    assert b["meta"]["n_samples"] == 6
    assert a["resources"]["total_steps_p50"] == 2
    assert b["resources"]["total_steps_p50"] == 9


def test_budget_can_use_tail_quantile(client):
    tid_p50 = _seed_trace(
        client, "cold", 1, completed=False, total_cost=0.01
    )
    st_p50 = client.post(
        "/api/public/budgets",
        json={
            "trace_id": tid_p50,
            "cap_usd": 0.03,
            "on_exceed": "kill",
            "quantile": "p50",
        },
    ).json()

    tid_p90 = _seed_trace(
        client, "cold", 1, completed=False, total_cost=0.01
    )
    st_p90 = client.post(
        "/api/public/budgets",
        json={
            "trace_id": tid_p90,
            "cap_usd": 0.03,
            "on_exceed": "kill",
            "quantile": "p90",
        },
    ).json()

    assert not st_p50["breached"]
    assert st_p90["breached"]
    assert st_p90["quantile"] == "p90"


def test_forecast_conditions_on_request_metadata(client):
    for _ in range(20):
        _seed_trace(client, "refactor", 2, metadata={"file_count": 1})
        _seed_trace(client, "refactor", 10, metadata={"file_count": 1000})
    client.post("/api/public/predict/rebuild")

    small = client.post(
        "/api/public/forecast",
        json={"trace_name": "refactor", "metadata": {"file_count": 1}},
    ).json()
    large = client.post(
        "/api/public/forecast",
        json={"trace_name": "refactor", "metadata": {"file_count": 1000}},
    ).json()

    assert small["resources"]["total_steps_p50"] == 2
    assert large["resources"]["total_steps_p50"] == 10
    assert "request-conditioned" in small["meta"]["explanation"]


def test_tool_counts_are_remaining_not_full_trace(client):
    for _ in range(8):
        _seed_trace(
            client,
            "shopping",
            3,
            tool_names=["web_search", "web_search", "read_file"],
        )
    client.post("/api/public/predict/rebuild")

    partial = _seed_trace(
        client,
        "shopping",
        1,
        completed=False,
        tool_names=["web_search"],
    )
    body = client.get(f"/api/public/predict/{partial}").json()
    tools = {item["tool"]: item["p50"] for item in body["resources"]["tool_call_counts"]}

    assert tools["web_search"] == 1
    assert tools["read_file"] == 1

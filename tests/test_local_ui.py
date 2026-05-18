from __future__ import annotations

import base64
import uuid
from datetime import datetime, timedelta, timezone


def _auth(project_id: str) -> dict[str, str]:
    raw = base64.b64encode(f"{project_id}:secret".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


def _seed_trace(client, *, project_id: str = "default", completed: bool = True) -> str:
    trace_id = "trace-" + uuid.uuid4().hex
    obs_id = "obs-" + uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    start = now + timedelta(seconds=1)
    end = now + timedelta(seconds=3)
    events = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": now.isoformat(),
            "type": "trace-create",
            "body": {
                "id": trace_id,
                "name": "research_agent",
                "timestamp": now.isoformat(),
                "userId": "user-1",
                "sessionId": "session-1",
                "metadata": {"file_count": 12},
            },
        },
        {
            "id": str(uuid.uuid4()),
            "timestamp": start.isoformat(),
            "type": "generation-create",
            "body": {
                "id": obs_id,
                "traceId": trace_id,
                "name": "draft",
                "model": "claude-sonnet-4-6",
                "startTime": start.isoformat(),
                "endTime": end.isoformat(),
                "usage": {
                    "input": 100,
                    "output": 40,
                    "total": 140,
                    "totalCost": 0.012,
                },
            },
        },
    ]
    if completed:
        events.append(
            {
                "id": str(uuid.uuid4()),
                "timestamp": (end + timedelta(seconds=1)).isoformat(),
                "type": "trace-create",
                "body": {"id": trace_id, "name": "research_agent", "output": "done"},
            }
        )

    headers = _auth(project_id) if project_id != "default" else None
    response = client.post("/api/public/ingestion", json={"batch": events}, headers=headers)
    assert response.status_code == 200, response.text
    return trace_id


def test_local_ui_shell_is_served(client):
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "Langpred Local" in response.text
    assert "/ui/assets/app.js" in response.text


def test_local_trace_list_and_detail(client):
    trace_id = _seed_trace(client)

    response = client.get("/api/local/traces")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["project_id"] == "default"
    assert body["summary"]["traces"] == 1
    assert body["summary"]["total_tokens"] == 140
    assert body["traces"][0]["id"] == trace_id
    assert body["traces"][0]["prediction"]["usd_total_p50"] >= 0.012

    detail = client.get(f"/api/local/traces/{trace_id}")
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert payload["trace"]["name"] == "research_agent"
    assert payload["trace"]["observations"][0]["model"] == "claude-sonnet-4-6"
    assert payload["prediction"]["trace_id"] == trace_id


def test_local_trace_list_uses_langfuse_project_key(client):
    trace_id = _seed_trace(client, project_id="pk-local")

    assert client.get("/api/local/traces").json()["summary"]["traces"] == 0
    scoped = client.get("/api/local/traces", headers=_auth("pk-local")).json()
    assert scoped["project_id"] == "pk-local"
    assert scoped["summary"]["traces"] == 1
    assert scoped["traces"][0]["id"] == trace_id

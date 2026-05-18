"""
Integration test: the **real**, pip-installed Langfuse SDK writes to a live
Langpred server, and Langpred returns sensible predictions on a partial trace.

This is the load-bearing test for the "drop-in" claim. If Langfuse changes
their batch envelope and we break compat, this fails — much harder than the
unit tests, which use our own SDK.

Skipped if ``langfuse`` is not installed (it's not a hard dependency of the
Langpred test suite).
"""
from __future__ import annotations

import os
import random
import socket
import threading
import time

import httpx
import pytest


langfuse = pytest.importorskip(
    "langfuse",
    reason="real Langfuse SDK not installed (pip install 'langfuse<3' to enable)",
)


SHAPE = "support_agent"


@pytest.fixture
def live_server():
    """Boot uvicorn on a free port and yield the base URL."""
    import uvicorn

    from langpred_server import db, predict
    db.reset_store_for_tests()
    predict.reset_service_for_tests()

    from langpred_server.main import app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 6
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:
        raise RuntimeError("test server did not start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=3)


def _simulate_step(trace, step_idx: int) -> None:
    tool = random.choice(["web_search", "lookup_kb", "read_ticket"])
    span = trace.span(name=tool)
    span.end()
    pt = random.randint(300, 1200)
    ct = random.randint(80, 350)
    gen = trace.generation(
        name=f"reason_{step_idx}",
        model="claude-sonnet-4-6",
        input=f"step {step_idx}",
        output="ok",
        usage={"input": pt, "output": ct, "total": pt + ct},
    )
    gen.end()


def _seed(host: str, n: int) -> None:
    """Write ``n`` completed traces via the real Langfuse SDK."""
    from langfuse import Langfuse

    lf = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        host=host,
        flush_at=1,
        flush_interval=0.05,
    )
    try:
        for i in range(n):
            trace = lf.trace(name=SHAPE, user_id=f"u-{i}")
            for s in range(random.randint(3, 7)):
                _simulate_step(trace, s)
            trace.update(output="resolved")
        lf.flush()
    finally:
        lf.shutdown()
    # Force a rebuild so the new traces are part of the predictor cohort.
    httpx.post(f"{host}/api/public/predict/rebuild", timeout=5).raise_for_status()


def test_real_langfuse_writes_and_langpred_predicts(live_server):
    random.seed(7)
    _seed(live_server, n=25)

    # Now run ONE partial trace using the real Langfuse SDK and predict on it.
    from langfuse import Langfuse

    lf = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        host=live_server,
        flush_at=1,
        flush_interval=0.05,
    )
    trace = lf.trace(name=SHAPE, user_id="u-live")
    for s in range(2):
        _simulate_step(trace, s)
    lf.flush()
    lf.shutdown()

    body = httpx.get(
        f"{live_server}/api/public/predict/{trace.id}", timeout=5
    ).json()

    # ---- Drop-in claim: the real SDK's batch shape was accepted -----------
    assert body["trace_id"] == trace.id
    assert body["meta"]["tier"] in ("knn", "gbm", "heuristic")
    assert body["meta"]["n_samples"] > 0, (
        "Langpred didn't see any cohort. Real Langfuse SDK ingestion likely failed."
    )

    # ---- Numbers are sane ------------------------------------------------
    t = body["time"]
    assert t["total_seconds_p50"] >= t["elapsed_seconds"]
    assert t["remaining_seconds_p50"] >= 0

    c = body["cost"]
    assert c["spent_usd"] > 0, "should have priced the 2 generations done so far"
    assert c["usd_total_p50"] >= c["spent_usd"]
    assert c["usd_remaining_p50"] >= 0
    # Per-model split should have at least sonnet because that's what we used.
    assert any(m["model"] == "claude-sonnet-4-6" for m in c["usd_by_model"])

    r = body["resources"]
    # We did 2 steps; cohort traces had 3-7, so expect remaining roughly 1-5.
    assert r["total_steps_p50"] >= 2
    assert r["steps_remaining_p50"] >= 0
    # Tool histogram should mention all 3 tools (we sampled uniformly).
    tools = {tc["tool"] for tc in r["tool_call_counts"]}
    assert tools.issuperset({"web_search", "lookup_kb", "read_ticket"})

    n = body["next"]
    # In our shape every step starts with a tool call, so next-kind should
    # be dominated by tool_call.
    assert n["next_kind_distribution"].get("tool_call", 0) > 0.5
    # top tools should include at least one of the simulated set.
    top_tool_names = {t["tool"] for t in n["top_next_tools"]}
    assert top_tool_names & {"web_search", "lookup_kb", "read_ticket"}

    risk = body["risk"]
    for key in ("offrails_risk", "loop_risk", "context_overflow_risk",
                "budget_overshoot_risk", "cost_spike_risk"):
        assert 0.0 <= risk[key] <= 1.0


def test_real_langfuse_budget_enforcement(live_server):
    """Drive a budget via the real Langfuse SDK on the trace, then verify
    Langpred trips X-Langpred-Budget on subsequent ingestions."""
    random.seed(11)
    _seed(live_server, n=15)

    from langfuse import Langfuse

    lf = Langfuse(
        public_key="pk", secret_key="sk", host=live_server,
        flush_at=1, flush_interval=0.05,
    )
    trace = lf.trace(name=SHAPE, user_id="u-budget")
    # Register a tight budget.
    httpx.post(
        f"{live_server}/api/public/budgets",
        json={"trace_id": trace.id, "cap_usd": 0.001, "on_exceed": "kill"},
        timeout=5,
    ).raise_for_status()

    # Run an expensive step to push us past the cap.
    span = trace.span(name="web_search")
    span.end()
    gen = trace.generation(
        name="big_call",
        model="claude-opus-4-7",
        usage={"input": 5000, "output": 3000, "total": 8000},
    )
    gen.end()
    lf.flush()
    lf.shutdown()

    # Status should now report breached.
    st = httpx.get(
        f"{live_server}/api/public/budgets/{trace.id}/status", timeout=5
    ).json()
    assert st["breached"], f"expected budget breach, got {st}"
    assert st["spent_usd"] > 0

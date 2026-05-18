"""Make sure the SDK's langfuse_compat surface emits batches the server eats."""
from __future__ import annotations

import threading
import time

import pytest


@pytest.fixture
def live_server():
    """Spin up the FastAPI app on a real socket so the httpx-based SDK can
    talk to it. Returns the base URL."""
    import socket
    import uvicorn
    from langpred_server.main import app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for startup.
    deadline = time.time() + 5
    import httpx
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/healthz")
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:
        raise RuntimeError("test server did not start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=3)


def test_langfuse_compat_v2_roundtrip(live_server):
    from langpred.langfuse_compat import Langfuse

    lf = Langfuse(host=live_server)
    trace = lf.trace(name="compat_smoke", user_id="u-1")
    g = trace.generation(
        name="reason",
        model="claude-sonnet-4-6",
        input="hello",
        output="hi",
        usage={"input": 10, "output": 20, "total": 30},
    )
    g.end()
    trace.update(output="done")
    lf.flush()

    cost = trace.predict_cost()
    assert cost.trace_id == trace.id
    assert cost.p50 >= 0


def test_observe_decorator(live_server):
    from langpred.langfuse_compat import get_client, observe

    client = get_client(host=live_server)

    @observe(as_type="span", name="outer")
    def outer(x: int) -> int:
        return inner(x) + 1

    @observe(as_type="span", name="inner")
    def inner(x: int) -> int:
        return x * 2

    assert outer(3) == 7
    client.flush()

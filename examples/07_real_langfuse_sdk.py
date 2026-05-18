"""
Example 07 — Drive Langpred with the **real** pip-installed Langfuse SDK.

This is the integration test of the drop-in claim: we use the actual
``langfuse`` package (not our compat shim), point ``LANGFUSE_HOST`` at a
local Langpred server, run a realistic support-agent trace, and ask
Langpred for predictions.

What it demonstrates end-to-end:

  1. The official Langfuse SDK (v2) writes to ``/api/public/ingestion``
     on Langpred without modification.
  2. Langpred stores the events, builds trajectories, trains predictors.
  3. We hit Langpred's ``/api/public/predict/{tid}`` endpoint and get back
     a full AgentPrediction.

Run:
    # Terminal 1
    uvicorn langpred_server.main:app --port 7187

    # Terminal 2
    pip install 'langfuse<3' httpx
    python examples/07_real_langfuse_sdk.py
"""
from __future__ import annotations

import os
import random
import time
import uuid

# *** This is the only configuration needed to migrate an existing Langfuse
# *** app to Langpred. Point the host at us; keep your existing SDK.
os.environ["LANGFUSE_HOST"] = os.environ.get("LANGFUSE_HOST", "http://localhost:7187")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-local")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-local")

import httpx
from langfuse import Langfuse  # <-- the REAL Langfuse SDK


HOST = os.environ["LANGFUSE_HOST"]
SHAPE = "support_agent"


def simulate_step(trace, step_idx: int, *, expensive: bool = False) -> None:
    """One realistic agent step: a tool call followed by an LLM generation."""
    # Tool span.
    tool = random.choice(["web_search", "lookup_kb", "read_ticket"])
    span = trace.span(name=tool)
    time.sleep(0.01)  # pretend tool took a moment
    span.end()

    # LLM generation.
    model = "claude-opus-4-7" if expensive else "claude-sonnet-4-6"
    pt = random.randint(300, 1200) if not expensive else random.randint(2000, 5000)
    ct = random.randint(80, 350) if not expensive else random.randint(500, 1500)
    gen = trace.generation(
        name=f"reason_{step_idx}",
        model=model,
        input=f"step {step_idx}",
        output="thought + answer",
        usage={"input": pt, "output": ct, "total": pt + ct},
    )
    time.sleep(0.005)
    gen.end()


def seed_history(n_traces: int = 25) -> None:
    """Emit ``n_traces`` complete support-agent traces using the REAL Langfuse SDK
    so the predictor has a cohort to lean on.
    """
    print(f"Seeding {n_traces} completed historical traces using the real Langfuse SDK...")
    lf = Langfuse(flush_at=1, flush_interval=0.1)  # flush aggressively
    for i in range(n_traces):
        trace = lf.trace(name=SHAPE, user_id=f"hist-{i}", session_id=f"sess-{i}")
        n_steps = random.randint(3, 7)
        for s in range(n_steps):
            simulate_step(trace, s)
        trace.update(output="resolved")
    lf.flush()
    lf.shutdown()
    print("  …seeded. Asking Langpred to rebuild predictors.")
    httpx.post(f"{HOST}/api/public/predict/rebuild").raise_for_status()


def run_one_live_trace() -> str:
    """Run a fresh PARTIAL trace and return its id so we can predict against it."""
    lf = Langfuse(flush_at=1, flush_interval=0.1)
    trace = lf.trace(name=SHAPE, user_id="live-customer", session_id="sess-live")
    print(f"\nLive trace id = {trace.id}")
    # Execute only 2 of the typical 3-7 steps — we want a partial trajectory
    # so Langpred has something to predict the *remainder* of.
    for s in range(2):
        simulate_step(trace, s)
    lf.flush()
    lf.shutdown()
    return trace.id


def print_prediction(trace_id: str) -> None:
    """Ask Langpred's prediction API directly via HTTP — this is what the
    Langpred SDK does under the hood; here we're showing the raw shape.
    """
    r = httpx.get(f"{HOST}/api/public/predict/{trace_id}", timeout=10)
    r.raise_for_status()
    p = r.json()

    print()
    print(f"=== Langpred AgentPrediction for trace {trace_id[:8]} ===")
    print(f"meta:    tier={p['meta']['tier']}  n={p['meta']['n_samples']}  "
          f"confidence={p['meta']['confidence']:.2f}")
    print(f"        ({p['meta']['explanation']})")
    print()

    t = p["time"]
    print("⏱  time")
    print(f"   elapsed       : {t['elapsed_seconds']:.2f}s")
    print(f"   total p50/p90 : {t['total_seconds_p50']:.2f}s / {t['total_seconds_p90']:.2f}s")
    print(f"   remaining p50 : {t['remaining_seconds_p50']:.2f}s")
    print(f"   next step ~   : {t['next_step_seconds_p50']:.3f}s")

    c = p["cost"]
    print("\n💵  cost")
    print(f"   spent         : ${c['spent_usd']:.4f}")
    print(f"   total p50/p90 : ${c['usd_total_p50']:.4f} / ${c['usd_total_p90']:.4f}")
    print(f"   remaining p50 : ${c['usd_remaining_p50']:.4f}")
    for m in c["usd_by_model"]:
        print(f"   model {m['model']:<22} p50=${m['usd_p50']:.4f}")

    r = p["resources"]
    print("\n📦  resources")
    print(f"   total steps p50/p90 : {r['total_steps_p50']:.1f} / {r['total_steps_p90']:.1f}")
    print(f"   steps remaining p50 : {r['steps_remaining_p50']:.1f}")
    print(f"   total tokens p50    : {r['total_tokens_p50']:.0f}")
    for tc in r["tool_call_counts"][:4]:
        print(f"   tool {tc['tool']:<14} p50={tc['p50']:.1f}  p90={tc['p90']:.1f}")

    n = p["next"]
    print("\n🧭  next action")
    print(f"   kind dist     : {n['next_kind_distribution']}")
    print(f"   likely model  : {n['likely_next_model']}")
    print(f"   P(finish in 1): {n['p_finish_within_one_step']:.2f}")
    for tt in n["top_next_tools"][:3]:
        print(f"   tool guess    : {tt['tool']:<14} p={tt['probability']:.2f}")

    risk = p["risk"]
    print("\n⚠️  risk")
    for k, v in risk.items():
        if k != "notes":
            print(f"   {k:<24}: {v:.3f}")
    if risk.get("notes"):
        print(f"   notes: {risk['notes']}")


def main() -> None:
    random.seed(42)

    # Make sure langpred is reachable first.
    try:
        httpx.get(f"{HOST}/healthz", timeout=2).raise_for_status()
    except Exception as e:
        raise SystemExit(
            f"Langpred not reachable at {HOST}. Start it with:\n"
            f"  uvicorn langpred_server.main:app --port 7187\n"
            f"(error was: {e})"
        )

    seed_history(n_traces=25)
    tid = run_one_live_trace()
    print_prediction(tid)


if __name__ == "__main__":
    main()

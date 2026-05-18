"""
Example 08 — The five pre-emptive actions.

langpred's value vs Langfuse is **acting before bad outcomes**. This script
shows all five actions you can take:

  1. reject-upfront   — don't even start; predicted cost too high
  2. route-at-start   — pick the cheapest viable model BEFORE the trace runs
  3. alert            — fire a webhook when a risk/cost threshold trips
  4. scope-reduce     — ask the agent to shrink remaining work (keep model)
  5. kill             — hard stop; SDK raises BudgetExceeded

We deliberately don't ship "downgrade-mid-trace" — switching models after
KV cache + chain-of-thought has built up usually loses money and degrades
output. Pick the model upfront instead.

Run:
    uvicorn langpred_server.main:app --port 7187 &
    python examples/08_action_menu.py
"""
from __future__ import annotations

import os
import random

os.environ.setdefault("LANGFUSE_HOST", "http://localhost:7187")

from langpred import BudgetExceeded, Langpred


SHAPE = "research_agent"


def seed_history(lp: Langpred, n: int = 30) -> None:
    """Pretend we've already run many of these so the cohort is non-trivial."""
    for _ in range(n):
        trace = lp.trace(name=SHAPE)
        for k in range(random.randint(3, 7)):
            trace.span(name=random.choice(["web_search", "lookup_kb", "read_doc"])).end()
            trace.generation(
                model=random.choice(["claude-sonnet-4-6", "claude-haiku-4-5"]),
                usage={"input": random.randint(300, 1200),
                       "output": random.randint(80, 400),
                       "total": 0},
            ).end()
        trace.update(output="ok")
    lp.flush()
    import httpx
    httpx.post("http://localhost:7187/api/public/predict/rebuild").raise_for_status()


def action_1_reject_upfront(lp: Langpred) -> None:
    print("=" * 60)
    print("ACTION 1: reject-upfront")
    print("=" * 60)
    forecast = lp.forecast(trace_name=SHAPE)
    print(f"  cohort p50: ${forecast.cost.usd_total_p50:.4f}   "
          f"p90: ${forecast.cost.usd_total_p90:.4f}")
    customer_budget = 0.0001  # absurdly low to demonstrate rejection
    if forecast.cost.usd_total_p90 > customer_budget:
        print(f"  → REJECT: p90 ${forecast.cost.usd_total_p90:.4f} > budget "
              f"${customer_budget:.4f}. Don't even start.")
    else:
        print(f"  → accept; proceed")


def action_2_route_at_start(lp: Langpred) -> None:
    print("\n" + "=" * 60)
    print("ACTION 2: route-at-start (pick model BEFORE the trace runs)")
    print("=" * 60)
    forecast = lp.forecast(trace_name=SHAPE)
    p90 = forecast.cost.usd_total_p90
    if p90 < 0.005:
        chosen = "claude-haiku-4-5"
    elif p90 < 0.05:
        chosen = "claude-sonnet-4-6"
    else:
        chosen = "claude-opus-4-7"
    print(f"  cohort cost p90 = ${p90:.4f}  →  routing to {chosen}")
    print(f"  (no KV cache to lose — decision made before step 0)")


def action_3_alert_webhook(lp: Langpred) -> None:
    print("\n" + "=" * 60)
    print("ACTION 3: alert webhook")
    print("=" * 60)
    trace = lp.trace(name=SHAPE)
    trace.alert_when(
        "cost.usd_total_p50 > 0.001",
        webhook_url="https://example.com/hooks/slack",
        min_interval_seconds=0,
    )
    trace.alert_when(
        "risk.loop_risk > 0.7",
        webhook_url="https://example.com/hooks/oncall",
    )
    trace.span(name="web_search").end()
    trace.generation(
        model="claude-sonnet-4-6",
        usage={"input": 800, "output": 300, "total": 1100},
    ).end()
    lp.flush()
    print(f"  trace {trace.id[:8]} has 2 alert rules registered.")
    print("  cost rule will fire on the next ingestion event (predicted p50 > $0.001).")


def action_4_scope_reduce(lp: Langpred) -> None:
    print("\n" + "=" * 60)
    print("ACTION 4: scope-reduce hint (no model switch)")
    print("=" * 60)
    trace = lp.trace(name=SHAPE)

    scope_reduce_fired = {"yes": False}

    def shrink_remaining_work() -> None:
        scope_reduce_fired["yes"] = True
        print("  ⤵ callback fired: agent should now shrink max_tokens / drop optional steps")

    trace.on_scope_reduce(shrink_remaining_work)
    trace.set_budget(usd=0.0001, on_exceed="scope_reduce")

    # Run an expensive step to trip the budget.
    trace.generation(
        model="claude-opus-4-7",
        usage={"input": 4000, "output": 2000, "total": 6000},
    ).end()
    lp.flush()
    if scope_reduce_fired["yes"]:
        print("  → KV cache preserved (same model). Agent code received the hint.")
    else:
        print("  → no breach; budget was OK")


def action_5_kill(lp: Langpred) -> None:
    print("\n" + "=" * 60)
    print("ACTION 5: hard kill")
    print("=" * 60)
    trace = lp.trace(name=SHAPE)
    guard = trace.set_budget(usd=0.0001, on_exceed="kill")

    try:
        # First step to push us over.
        trace.generation(
            model="claude-opus-4-7",
            usage={"input": 4000, "output": 2000, "total": 6000},
        ).end()
        lp.flush()
        guard.check()
        print("  → no breach (this would be a bug)")
    except BudgetExceeded as e:
        print(f"  → BudgetExceeded raised: {e}")
        print("  → agent loop bails out cleanly")


def main() -> None:
    random.seed(13)
    lp = Langpred()
    print("Seeding 30 historical traces so the cohort is non-trivial...\n")
    seed_history(lp, n=30)

    action_1_reject_upfront(lp)
    action_2_route_at_start(lp)
    action_3_alert_webhook(lp)
    action_4_scope_reduce(lp)
    action_5_kill(lp)

    print("\n" + "=" * 60)
    print("note: 'downgrade-mid-trace' is intentionally NOT in this menu.")
    print("KV cache loss + reasoning incoherence usually wipes the savings.")
    print("Use route-at-start (action 2) instead.")
    print("=" * 60)

    lp.shutdown()


if __name__ == "__main__":
    main()

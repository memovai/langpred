"""
Example 02 — Budget guard.

Cap a single agent run at $0.05. Langpred re-evaluates the budget every time
your SDK flushes an event (cheap, async). When the *predicted* total spend
exceeds the cap, the next `guard.check()` raises `BudgetExceeded` and your
agent loop bails out cleanly — *before* the dollar is actually spent.

Compare to the status quo: `max_steps=50` kills good runs and lets bad ones
through. A budget kills bad ones and lets good ones through.

Run:
    python examples/02_budget_guard.py
"""
from __future__ import annotations

import os
import random

os.environ.setdefault("LANGFUSE_HOST", "http://localhost:7187")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-local")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-local")

from langpred import BudgetExceeded, Langpred


def runaway_agent(lp: Langpred, cap_usd: float) -> str:
    trace = lp.trace(name="research_agent", user_id="customer-runaway")
    guard = trace.set_budget(usd=cap_usd, on_exceed="kill")

    try:
        for step in range(1, 200):
            # Pretend each step does a chunky generation.
            gen = trace.generation(
                name=f"step_{step}",
                model="claude-opus-4-7",  # expensive on purpose
                input=f"step {step}",
                output="..." * 200,
                usage={"input": 2000, "output": 1500, "total": 3500},
            )
            gen.end()
            lp.flush()  # forces budget re-evaluation server-side

            guard.check()  # may raise BudgetExceeded

            if random.random() < 0.02:  # would normally exit on its own
                return "natural completion"
        return "max steps without budget breach"
    except BudgetExceeded as exc:
        trace.update(output={"status": "killed_by_budget", "reason": str(exc)})
        lp.flush()
        return f"killed at step {step}: {exc}"


def main() -> None:
    lp = Langpred()
    result = runaway_agent(lp, cap_usd=0.05)
    print(f"agent finished: {result}")
    lp.shutdown()


if __name__ == "__main__":
    main()

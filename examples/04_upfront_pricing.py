"""
Example 04 — Quote a fixed price to a customer at trace-start.

Pattern:
  1. Create a trace with the user's request, *don't* execute yet.
  2. Run a tiny "scout" step so Langpred has a non-empty prefix.
  3. Ask for `predict_cost()` → use p90 as the quote (covers tail risk).
  4. Wait for the customer to accept, *then* run the rest.

Run:
    python examples/04_upfront_pricing.py
"""
from __future__ import annotations

import os

os.environ.setdefault("LANGFUSE_HOST", "http://localhost:7187")

from langpred import Langpred


def quote(lp: Langpred, request: str, margin: float = 1.4) -> tuple[str, float]:
    """Return (trace_id, quoted_usd). margin=1.4 means we charge 40% over p90."""
    trace = lp.trace(name="refactor_repo", input={"request": request})
    # Cheap scout: a small generation so the predictor has a prefix to compare.
    scout = trace.generation(
        name="scout",
        model="claude-haiku-4-5",
        input=request,
        output="planned",
        usage={"input": 400, "output": 100, "total": 500},
    )
    scout.end()
    lp.flush()
    cost = trace.predict_cost()
    quoted = cost.usd_p90 * margin
    print(
        f"trace={trace.id}  "
        f"cost p50=${cost.usd_p50:.4f}  p90=${cost.usd_p90:.4f}  "
        f"p99=${cost.usd_p99:.4f}  tier={cost.tier}  n={cost.n_samples}"
    )
    print(f"-> quote (p90 × {margin}): ${quoted:.4f}")
    return trace.id, quoted


def main() -> None:
    lp = Langpred()
    quote(lp, "refactor my repo's auth layer to use JWT instead of sessions")
    quote(lp, "summarise this 3000-word report")
    lp.shutdown()


if __name__ == "__main__":
    main()

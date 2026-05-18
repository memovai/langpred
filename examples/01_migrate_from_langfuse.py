"""
Example 01 — Migrate from Langfuse in one line.

The point of this example is to **show the diff** that an existing Langfuse
user makes to start getting Langpred predictions. Two layers:

1. Zero code change — just point LANGFUSE_HOST at a Langpred server. The
   existing `langfuse` library writes to us; predictions show up in the
   dashboard.

2. One-import code change — `from langpred.langfuse_compat import Langfuse`
   instead of `from langfuse import Langfuse`. Now `predict_eta()`,
   `predict_cost()`, `set_budget()` are available on the trace object.

Run:
    python examples/01_migrate_from_langfuse.py
    (assumes `langpred-server` is running on http://localhost:7187)
"""
from __future__ import annotations

import os
import random
import time

# Pretend we did the env-var migration:
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:7187")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-local")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-local")

# ---- BEFORE ----------------------------------------------------------------
# from langfuse import Langfuse
# ---- AFTER (the one-line diff) --------------------------------------------
from langpred.langfuse_compat import Langfuse


def fake_llm_call(prompt: str, model: str) -> tuple[str, int, int]:
    """Pretend to call an LLM. Returns (output, prompt_tokens, completion_tokens)."""
    pt = len(prompt.split()) * 4
    ct = random.randint(80, 400)
    time.sleep(random.uniform(0.05, 0.2))
    return f"reply to: {prompt[:40]}…", pt, ct


def run_agent_step(trace, step: int) -> None:
    span = trace.span(name=f"think_step_{step}")
    out, pt, ct = fake_llm_call(f"step {step}: plan next move", "claude-sonnet-4-6")
    gen = trace.generation(
        name="reason",
        model="claude-sonnet-4-6",
        input=f"step {step}: plan next move",
        output=out,
        usage={"input": pt, "output": ct, "total": pt + ct},
    )
    gen.end()
    span.end()


def main() -> None:
    langfuse = Langfuse()

    trace = langfuse.trace(
        name="research_agent",
        user_id="customer-42",
        session_id="s-001",
        metadata={"feature": "migration_demo"},
    )

    # Existing-style instrumentation — no diff vs Langfuse below this line.
    for step in range(1, 6):
        run_agent_step(trace, step)

    trace.update(output="done")
    langfuse.flush()

    # --- NEW capabilities (additive, ignore if you don't care) -------------
    eta = trace.predict_eta()
    cost = trace.predict_cost()
    print(f"trace.id = {trace.id}")
    print(f"predicted ETA   : p50={eta.seconds_p50:.1f}s  p90={eta.seconds_p90:.1f}s  p99={eta.seconds_p99:.1f}s  (tier={eta.tier}, n={eta.n_samples})")
    print(f"predicted cost  : p50=${cost.usd_p50:.4f}  p90=${cost.usd_p90:.4f}  p99=${cost.usd_p99:.4f}  (tier={cost.tier}, n={cost.n_samples})")
    print(f"off-rails score : {trace.offrails_score().score:.2f}")

    langfuse.shutdown()


if __name__ == "__main__":
    main()

"""
Example 03 — Render a real ETA progress bar.

Most agent UIs show `step N / 50` which is a fiction. With Langpred you can
show a calibrated `"~38s remaining (p90: 4m12s)"`.

This script simulates an agent run and prints the live ETA after each step.

Run:
    python examples/03_eta_in_ui.py
"""
from __future__ import annotations

import os
import random
import time

os.environ.setdefault("LANGFUSE_HOST", "http://localhost:7187")

from langpred import Langpred


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def main() -> None:
    lp = Langpred()
    trace = lp.trace(name="research_agent", user_id="ui-demo")
    n_steps = random.randint(6, 14)

    for step in range(1, n_steps + 1):
        gen = trace.generation(
            name=f"step_{step}",
            model="claude-sonnet-4-6",
            input=f"step {step}",
            output="ok",
            usage={"input": 800, "output": 250, "total": 1050},
        )
        gen.end()
        lp.flush()
        time.sleep(random.uniform(0.1, 0.4))

        eta = trace.predict_eta()
        steps = trace.predict_steps()
        print(
            f"step {step:>2}: ETA ~{fmt(eta.seconds_p50)} "
            f"(p90 {fmt(eta.seconds_p90)}, p99 {fmt(eta.seconds_p99)}) "
            f"| steps total ~{steps.p50:.0f} (p90 {steps.p90:.0f}) "
            f"| tier={eta.tier} n={eta.n_samples}"
        )

    trace.update(output="ui demo done")
    lp.shutdown()


if __name__ == "__main__":
    main()

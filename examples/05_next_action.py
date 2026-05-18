"""
Example 05 — Predict the *next* action.

What langpred has that pure observability tools don't: a distribution over
what the agent is about to do. Useful for:

- **Pre-warming caches** ("80% chance the next call is `web_search`, prefetch")
- **Model routing** ("likely_next_model is opus — downgrade to sonnet")
- **Showing the user a smarter spinner** ("looking things up..." vs "writing...")

Run:
    python examples/05_next_action.py
"""
from __future__ import annotations

import os

os.environ.setdefault("LANGFUSE_HOST", "http://localhost:7187")

from langpred import Langpred


def main() -> None:
    lp = Langpred()
    trace = lp.trace(name="research_agent", user_id="next-action-demo")

    # Simulate a partial trajectory of one tool call + one generation.
    trace.span(name="web_search").end()
    trace.generation(
        name="reason",
        model="claude-sonnet-4-6",
        usage={"input": 800, "output": 250, "total": 1050},
    ).end()
    lp.flush()

    next_action = trace.predict_next_action()
    print(f"next-kind distribution : {next_action.next_kind_distribution}")
    print(f"most likely kind       : {next_action.most_likely_kind()}")
    if next_action.top_next_tools:
        for t in next_action.top_next_tools:
            print(f"  tool {t.tool:<14} p={t.probability:.2f}")
    print(f"likely next model      : {next_action.likely_next_model}")
    print(f"P(finish within 1 step): {next_action.p_finish_within_one_step:.2f}")
    print(f"expected next-step ${next_action.expected_next_step_usd_p50:.4f} "
          f"in {next_action.expected_next_step_seconds_p50:.2f}s")

    lp.shutdown()


if __name__ == "__main__":
    main()

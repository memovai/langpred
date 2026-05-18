"""
Example 06 — Forecast resources, not just time.

Pulls a full :class:`AgentPrediction` in one call and prints:

- time remaining (calibrated band)
- cost remaining + per-model breakdown
- token budget pressure
- per-tool call counts predicted for the rest of the run
- risk flags (loop, off-rails, context-window, budget overshoot, cost spike)

Run:
    python examples/06_resource_forecast.py
"""
from __future__ import annotations

import os

os.environ.setdefault("LANGFUSE_HOST", "http://localhost:7187")

from langpred import Langpred


def fmt_seconds(s: float) -> str:
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}m"


def main() -> None:
    lp = Langpred()
    trace = lp.trace(name="research_agent", user_id="forecast-demo")

    # A partial trajectory.
    trace.span(name="web_search").end()
    trace.generation(
        model="claude-sonnet-4-6",
        usage={"input": 1200, "output": 300, "total": 1500},
    ).end()
    trace.span(name="read_file").end()
    lp.flush()

    p = trace.predict()

    print(f"--- trace {p.trace_id[:8]} (tier={p.meta.tier}, n={p.meta.n_samples}) ---")
    print()
    print("⏱  Time")
    print(f"   elapsed         : {fmt_seconds(p.time.elapsed_seconds)}")
    print(f"   remaining p50   : {fmt_seconds(p.time.remaining_seconds_p50)}")
    print(f"   remaining p90   : {fmt_seconds(p.time.remaining_seconds_p90)}")
    print(f"   next step ~     : {p.time.next_step_seconds_p50:.2f}s")
    print(f"   compute vs io   : {p.time.compute_seconds_p50:.1f}s LLM / {p.time.io_seconds_p50:.1f}s I/O")
    print()
    print("💵  Cost")
    print(f"   spent           : ${p.cost.spent_usd:.4f}")
    print(f"   remaining p50   : ${p.cost.usd_remaining_p50:.4f}")
    print(f"   remaining p90   : ${p.cost.usd_remaining_p90:.4f}")
    print(f"   next step ~     : ${p.cost.next_step_usd_p50:.4f}")
    if p.cost.usd_by_model:
        for m in p.cost.usd_by_model:
            print(f"   by model: {m.model:<22} p50=${m.usd_p50:.4f}  p90=${m.usd_p90:.4f}")
    print()
    print("📦  Resources")
    print(f"   total tokens p50: {p.resources.total_tokens_p50:.0f}  p90: {p.resources.total_tokens_p90:.0f}")
    print(f"   steps remaining : ~{p.resources.steps_remaining_p50:.0f} (p90 {p.resources.steps_remaining_p90:.0f})")
    if p.resources.tool_call_counts:
        for tc in p.resources.tool_call_counts:
            print(f"   tool: {tc.tool:<14} p50={tc.p50:.1f}  p90={tc.p90:.1f}")
    print()
    print("🧭  Next action")
    print(f"   kind dist       : {p.next.next_kind_distribution}")
    print(f"   most likely     : {p.next.most_likely_kind()}")
    if p.next.top_next_tools:
        print(f"   top tool guess  : {p.next.top_next_tools[0].tool} "
              f"(p={p.next.top_next_tools[0].probability:.2f})")
    print(f"   P(finish in 1)  : {p.next.p_finish_within_one_step:.2f}")
    print()
    print("⚠️  Risk")
    print(f"   off-rails       : {p.risk.offrails_risk:.2f}")
    print(f"   loop            : {p.risk.loop_risk:.2f}")
    print(f"   ctx overflow    : {p.risk.context_overflow_risk:.2f}")
    print(f"   budget overshoot: {p.risk.budget_overshoot_risk:.2f}")
    print(f"   cost spike      : {p.risk.cost_spike_risk:.2f}")
    if p.risk.notes:
        print(f"   notes           : {p.risk.notes}")

    lp.shutdown()


if __name__ == "__main__":
    main()

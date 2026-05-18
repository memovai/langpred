# langpred

**Predict the *next chunk* of your agent's behavior — before it runs.**
Drop-in for [Langfuse](https://langfuse.com): same SDK, same ingestion endpoint, plus a per-trace forecast across **five dimensions**.

```text
$ python examples/06_resource_forecast.py
--- trace 3a9f1c8b (tier=knn, n=20) ---

⏱  Time
   elapsed         : 3s
   remaining p50   : 11s        next step ~ 1.7s   (LLM 8s / I/O 6s)
💵  Cost
   spent $0.012  remaining p50 $0.018  p90 $0.041  next step ~$0.003
   by model: claude-sonnet-4-6  p50=$0.024  p90=$0.052
📦  Resources
   ~6 steps left  (p90: 11)  tokens p50 8400 / p90 14200
   tool: web_search     p50=3.0  p90=5.0
   tool: read_file      p50=1.0  p90=2.0
🧭  Next action
   kind dist        : {'tool_call': 0.65, 'generation': 0.30, 'end': 0.05}
   most likely tool : web_search (p=0.42)
   likely model     : claude-sonnet-4-6
⚠️  Risk
   off-rails 0.05   loop 0.00   ctx-overflow 0.00   budget 0.00   spike 0.07
```

---

## What you can do with it

### 1. Show a real ETA — not a fake `step N/50` bar
```python
p = trace.predict()
print(f"~{p.time.remaining_seconds_p50:.0f}s left  (p90: {p.time.remaining_seconds_p90:.0f}s)")
```

### 2. Kill a runaway loop *before* the dollar lands
```python
with trace.set_budget(usd=0.50, on_exceed="kill"):
    agent.run()   # raises BudgetExceeded when predicted total > $0.50
```

### 3. Quote a fixed price *upfront*
```python
trace.generation(...)                  # one cheap scout step
quote = trace.predict().cost.usd_total_p90 * 1.4    # margin over p90
# → show "$X.XX" to the customer, run only if they accept
```

### 4. Decide what the agent is *about to do*
```python
na = trace.predict_next_action()
if na.most_likely_kind() == "tool_call" and na.top_next_tools[0].tool == "web_search":
    prefetch_search_cache()        # warm up before the call
if na.likely_next_model == "claude-opus-4-7":
    downgrade_to_sonnet()          # save 5× on the next step
```

### 5. Catch off-rails / context-overflow / cost-spike risks
```python
risk = trace.predict().risk
if risk.loop_risk > 0.5 or risk.context_overflow_risk > 0.5:
    abort()
if risk.cost_spike_risk > 0.5:
    notify_oncall()
```

---

## The five things langpred predicts

| Dimension | What it answers |
|---|---|
| **time** | `total_seconds`, `remaining_seconds`, `next_step_seconds`, `compute` vs `io` split |
| **cost** | `usd_total`, `usd_remaining`, `next_step_usd`, **`usd_by_model`** breakdown |
| **resources** | `total_tokens` (prompt / completion), `steps_remaining`, `llm_calls`, **per-tool call counts** |
| **next action** | Distribution over `{generation, tool_call, end}`; top-k next tool names; likely next model |
| **risk** | `offrails`, `loop`, `context_overflow`, `budget_overshoot`, `cost_spike` |

All five come from a single round-trip: `trace.predict() → AgentPrediction`. They share one kNN cohort, so they're internally consistent.

Per-dimension shortcuts exist for compatibility: `predict_eta()`, `predict_cost()`, `predict_steps()`, `predict_next_action()`, `is_off_rails()`.

---

## Install — 30 seconds

```bash
git clone https://github.com/memovai/langpred && cd langpred
pip install -e ./server -e ./sdk-python
uvicorn langpred_server.main:app --port 7187 &
```

SQLite at `./langpred.db`. No Postgres, no Redis, no GPU.

```bash
python examples/01_migrate_from_langfuse.py    # smoke test
```

---

## Migrate from Langfuse — literally one line

**A — zero code change.** Point your env at us:
```bash
export LANGFUSE_HOST=http://localhost:7187
```
The existing `langfuse.Langfuse()` SDK now writes to Langpred. Predictions show up server-side immediately.

**B — one import change.** Get the new methods in code:
```diff
- from langfuse import Langfuse
+ from langpred.langfuse_compat import Langfuse
```
Same constructor. Same `trace / span / generation / update / flush / @observe`. Now also `predict / predict_eta / predict_cost / predict_next_action / set_budget / is_off_rails`.

---

## Six runnable examples

| File | Shows |
|---|---|
| [`01_migrate_from_langfuse.py`](./examples/01_migrate_from_langfuse.py) | The exact one-import diff |
| [`02_budget_guard.py`](./examples/02_budget_guard.py) | Kill switch mid-loop |
| [`03_eta_in_ui.py`](./examples/03_eta_in_ui.py) | Live ETA updating per step |
| [`04_upfront_pricing.py`](./examples/04_upfront_pricing.py) | Scout-then-quote pattern |
| [`05_next_action.py`](./examples/05_next_action.py) | Predict the next tool / model |
| [`06_resource_forecast.py`](./examples/06_resource_forecast.py) | Full 5-dimension forecast |

---

## How it works — the 30-second version

Every Langfuse event becomes part of a **trajectory**. For a partial trajectory, we find the **k=20 nearest finished trajectories** (16-dim prefix feature L2, same-`trace.name` halved) and read off:

- their **final** outcomes → time/cost/resources quantiles
- their step at **prefix_len+1** → next-action distribution
- their **per-tool histograms** and **per-model cost split** → resource sub-forecasts
- their **status field** → off-rails risk

One kNN query powers all five sub-predictions, so they stay coherent. After 1000 finished traces per shape it auto-promotes to gradient-boosted quantile regressors for the scalar dimensions, while still pulling histograms from kNN. Full architecture and pain-point research: [DESIGN.md](./DESIGN.md).

---

## Status

v0.2: omnibus `predict()` + 5-dimension forecast, 14/14 tests green, benchmark MAE \$0.10, p90 coverage 79% (conformal calibration is next). Roadmap: OTel ingest, dashboard UI, TypeScript SDK, multi-tenant auth.

MIT — see [LICENSE](./LICENSE).

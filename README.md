# langpred

**Predict your agent's bill, ETA, and runaway loops — before they happen.**
Drop-in for [Langfuse](https://langfuse.com): same SDK, same ingestion endpoint, plus three things Langfuse doesn't do.

```text
$ python examples/03_eta_in_ui.py
step  1: ETA ~12s   cost p50 $0.018   p90 $0.052   tier=heuristic
step  3: ETA  ~9s   cost p50 $0.022   p90 $0.058   tier=knn  n=18
step  6: ETA  ~4s   cost p50 $0.028   p90 $0.061   tier=knn  n=18  ← about to finish
```

---

## What you can do with it

### 1. Show a real ETA — not a fake `step N/50` bar
```python
trace = lp.trace(name="research_agent")
# ... your agent code runs steps ...
eta = trace.predict_eta()
print(f"~{eta.seconds_p50:.0f}s left  (p90: {eta.seconds_p90:.0f}s)")
```
> **Effect:** users stop staring at spinners. Calibrated band updates every step.

### 2. Kill a runaway loop *before* the dollar lands
```python
trace = lp.trace(name="research_agent")
with trace.set_budget(usd=0.50, on_exceed="kill"):
    agent.run()   # raises BudgetExceeded when predicted total > $0.50
```
> **Effect:** replaces `max_steps=50`. Good runs finish; bad ones die early. No $1k surprise bill.

### 3. Quote a fixed price *upfront*
```python
trace = lp.trace(name="refactor_repo", input=request)
trace.generation(...)              # one cheap "scout" step
quote = trace.predict_cost().usd_p90 * 1.4    # margin over p90 covers tail
# → show "$X.XX" to the customer, run the rest only if they accept
```
> **Effect:** sell agent features at a fixed price instead of token-passthrough.

---

## Install — actually 30 seconds

```bash
git clone <this-repo> langpred && cd langpred
pip install -e ./server -e ./sdk-python
uvicorn langpred_server.main:app --port 7187 &      # the server
```

That's it. SQLite at `./langpred.db`, no Postgres, no Redis, no GPU.

```bash
python examples/01_migrate_from_langfuse.py    # smoke test
```

---

## Migrate from Langfuse — literally one line

Already using Langfuse? Two ways in:

**A — zero code change.** Point your env var at us:
```bash
export LANGFUSE_HOST=http://localhost:7187
```
The existing `langfuse.Langfuse()` SDK now writes to Langpred. Predictions are visible server-side immediately.

**B — one import change.** Get the new methods in code:
```diff
- from langfuse import Langfuse
+ from langpred.langfuse_compat import Langfuse
```
Same constructor. Same `trace / span / generation / update / flush / @observe`. Now also `predict_eta`, `predict_cost`, `predict_steps`, `set_budget`, `is_off_rails`.

---

## Four runnable examples (each < 70 lines)

| File | Shows |
|---|---|
| [`examples/01_migrate_from_langfuse.py`](./examples/01_migrate_from_langfuse.py) | The exact one-import diff, side-by-side |
| [`examples/02_budget_guard.py`](./examples/02_budget_guard.py) | Kill switch tripping mid-loop on an Opus runaway |
| [`examples/03_eta_in_ui.py`](./examples/03_eta_in_ui.py) | Live ETA updating after every step |
| [`examples/04_upfront_pricing.py`](./examples/04_upfront_pricing.py) | Scout-then-quote pattern for fixed-price SaaS |

---

## How it works — the 30-second version

Every Langfuse event Langpred ingests is stored as part of a **trajectory**. Per `trace.name`:

- **<50 finished traces** → heuristic floor (median step-rate × steps-remaining).
- **≥50 finished traces** → k-NN on a 16-dim prefix feature vector; report empirical p50/p90/p99 of nearest-neighbour outcomes.
- **≥1000 finished traces** → auto-promote to gradient-boosted quantile regressors (sklearn `HistGradientBoostingRegressor(loss="quantile")`).

Budgets re-evaluate asynchronously on every ingestion event; the server flags `X-Langpred-Budget: breached` on the next response, so the SDK kill is zero-RTT.

Full architecture, pain-point research, and trade-offs: [DESIGN.md](./DESIGN.md).

---

## Status

v0.1, runnable end-to-end. 10/10 tests green. Benchmark on synthetic trajectories: cost MAE \$0.10, p90 coverage 79% (still under-calibrated — conformal wrap is next). Roadmap: OTel ingest, dashboard UI, TypeScript SDK, multi-tenant auth.

MIT — see [LICENSE](./LICENSE).

# langpred

**Predict your agent's behavior — then act on it, before bad outcomes land.**
Drop-in for [Langfuse](https://langfuse.com): keep your SDK, redirect ingestion at us, and get **pre-emptive intervention** that Langfuse can't give you because Langfuse only looks backward.

```text
$ python examples/08_action_menu.py
ACTION 1: reject-upfront   →  cohort p90 $0.030 > budget $0.001. Don't start.
ACTION 2: route-at-start   →  cohort cost p90 = $0.030 → routing to sonnet
ACTION 3: alert webhook    →  POST https://hooks/slack on risk.loop_risk > 0.7
ACTION 4: scope-reduce     →  callback fired; agent shrinks max_tokens. Same model.
ACTION 5: hard kill        →  BudgetExceeded raised; agent loop bails out cleanly.
```

---

## The moat: act *before* the bad outcome

| | Langfuse | langpred |
|---|---|---|
| When you learn it cost \$500 | After the run | Step 3 of 50 |
| What you can do about it | Adjust next quote | Stop, reduce scope, alert, reroute, refuse to start |
| Customer experience | Surprise bill | Bounded bill, calibrated ETA, fixed quote |

Information without the ability to act is just expensive observability. langpred predicts **and** gives you a menu of pre-emptive actions for every outcome the prediction flags.

---

## The five actions

Each one matches a different time-in-trace and confidence level:

| Action | When | What it does | KV cache |
|---|---|---|---|
| **reject-upfront** | Before trace starts | Decline the request; predicted cost or risk too high | n/a (no trace) |
| **route-at-start** | Before trace starts | Pick the cheapest viable model based on the cohort | n/a (no trace) |
| **alert** | Any step | POST a webhook (Slack / PagerDuty / yours) on a threshold | preserved |
| **scope-reduce** | Mid-trace | Signal the agent to shrink remaining work (max_tokens etc.) | **preserved** |
| **kill** | Any step (early is best) | SDK raises `BudgetExceeded`; agent loop exits | invalidated |

> ⚠️  We **don't ship mid-trace model downgrade**. Switching models invalidates Anthropic prompt caching and breaks chain-of-thought coherence — the math usually loses. Use `route-at-start` instead; pick the model *before* any KV state has been built.

---

## Code for each action

### 1. reject-upfront — don't even start
```python
forecast = lp.forecast(trace_name="research_agent")
if forecast.cost.usd_total_p90 > customer.budget:
    refuse_quote(reason=f"predicted cost ${forecast.cost.usd_total_p90:.2f}")
```

### 2. route-at-start — pick the model upfront
```python
forecast = lp.forecast(trace_name="research_agent")
model = "claude-haiku-4-5" if forecast.cost.usd_total_p90 < 0.01 \
   else "claude-sonnet-4-6" if forecast.cost.usd_total_p90 < 0.10 \
   else "claude-opus-4-7"
trace = lp.trace(name="research_agent")   # then run with `model`
```

### 3. alert — fire a webhook on any threshold
```python
trace.alert_when("cost.usd_total_p90 > 1.00", webhook_url="https://hooks.slack.com/...")
trace.alert_when("risk.loop_risk > 0.70",    webhook_url="https://events.pagerduty.com/...")
trace.alert_when("time.remaining_seconds_p90 > 300", webhook_url="...")
```
Webhooks POST `{trace_id, condition, value, threshold, prediction}`. Re-fires at most once per 30s by default.

### 4. scope-reduce — shrink the work, keep the model
```python
trace.on_scope_reduce(lambda: agent.set_max_tokens(256))   # or skip optional steps
trace.set_budget(usd=0.50, on_exceed="scope_reduce")       # callback fires on breach
```
KV cache and chain-of-thought stay intact; the agent just does less.

### 5. kill — hard stop
```python
with trace.set_budget(usd=0.50, on_exceed="kill"):
    agent.run()       # raises BudgetExceeded when predicted_total > $0.50
```

---

## What it predicts — five dimensions, one round-trip

| | answers |
|---|---|
| **time** | total / **remaining** seconds, next-step time, compute vs I/O split |
| **cost** | total / **remaining** USD, next-step USD, **per-model breakdown** |
| **resources** | tokens (prompt + completion), steps remaining, **per-tool call counts** |
| **next action** | distribution over `{generation, tool_call, end}`, top-k next tools, likely model |
| **risk** | `off-rails` · `loop` · `context-overflow` · `budget-overshoot` · `cost-spike` |

All five from one `trace.predict()` call against a shared kNN cohort — internally consistent.

---

## Migrate from Langfuse — one line

**Zero code change** — point your env var at us:

```bash
export LANGFUSE_HOST=http://localhost:7187
```

Your existing `langfuse.Langfuse()` calls now write to Langpred; predictions appear immediately.

**One import change** — get the action menu in code:

```diff
- from langfuse import Langfuse
+ from langpred.langfuse_compat import Langfuse
```

Same constructor, same `trace / span / generation / @observe / flush`. New: `lp.forecast()`, `trace.predict()`, `trace.alert_when()`, `trace.on_scope_reduce()`, `trace.set_budget()`.

> ✓ **Validated against the real Langfuse SDK.** `tests/test_real_langfuse_integration.py` runs the actual pip-installed `langfuse` package against a live Langpred server. CI catches upstream envelope changes.

---

## Install — 30 seconds

```bash
git clone https://github.com/memovai/langpred && cd langpred
pip install -e ./server -e ./sdk-python
uvicorn langpred_server.main:app --port 7187 &
open http://localhost:7187/ui/
python examples/08_action_menu.py     # walks through all 5 actions live
```

SQLite by default. No Postgres, no Redis, no GPU.

---

## Eight runnable examples

| | shows |
|---|---|
| [`01_migrate_from_langfuse.py`](./examples/01_migrate_from_langfuse.py) | The exact one-import diff |
| [`02_budget_guard.py`](./examples/02_budget_guard.py) | Hard kill mid-loop |
| [`03_eta_in_ui.py`](./examples/03_eta_in_ui.py) | Live ETA per step |
| [`04_upfront_pricing.py`](./examples/04_upfront_pricing.py) | Scout-then-quote |
| [`05_next_action.py`](./examples/05_next_action.py) | Predict next tool / model |
| [`06_resource_forecast.py`](./examples/06_resource_forecast.py) | Full 5-dimension forecast |
| [`07_real_langfuse_sdk.py`](./examples/07_real_langfuse_sdk.py) | Driven by the **real** Langfuse SDK |
| [`08_action_menu.py`](./examples/08_action_menu.py) | All 5 pre-emptive actions in one script |

---

## How it works (30 seconds)

Every Langfuse event becomes part of a **trajectory**. For a partial trace we find the **k=20 nearest finished trajectories** (16-dim prefix L2; same-`trace.name` halved) and aggregate them five ways at once:

- final outcomes → quantile bands (time / cost / resources)
- step at `prefix_len+1` → next-action distribution
- per-tool histograms → expected remaining tool calls
- per-model cost split → upfront routing target
- status field → off-rails risk

For pre-trace `forecast()`, we skip the kNN step and aggregate the same-name cohort directly. After 1000 finished traces per shape, gradient-boosted quantile regressors auto-promote for scalar dimensions while distributions stay kNN. No GPU. Full architecture: **[DESIGN.md](./DESIGN.md)**.

---

## Status

| | |
|---|---|
| **Tests** | 28 / 28 green (+ optional real-Langfuse SDK integration when installed) |
| **Cost MAE** | \$0.10 on synthetic benchmark |
| **p90 coverage** | 79% (conformal calibration is on deck) |
| **Local UI** | `http://localhost:7187/ui/` — trace list, detail timeline, prediction cards, budgets, forecast |
| **Stack** | FastAPI + SQLite + scikit-learn · Python ≥3.10 · no GPU |

**Roadmap:** OTel ingest · TypeScript SDK · conformal calibration · optional proxy add-on for hard budget enforcement.

---

## Next steps

- **Read the design**: [DESIGN.md](./DESIGN.md) — pain-point research, why kNN-then-GBM, why no mid-trace downgrade.
- **Run the action menu live**: `python examples/08_action_menu.py` after starting the server.
- **Integrate**: change one env var; predictions and the action menu both land immediately.

MIT — see [LICENSE](./LICENSE).

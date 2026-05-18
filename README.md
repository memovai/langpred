# langpred

**Predict your agent's bill, ETA, runaway loops, and next action — before they run.**
Drop-in for [Langfuse](https://langfuse.com): keep your SDK, point one env var at us, get five dimensions of prediction.

```text
$ python examples/06_resource_forecast.py
--- trace 3a9f1c8b  (tier=knn, n=20, confidence=0.72) ---
⏱   remaining 11s  (p90 28s)         next step ~1.7s
💵   remaining $0.018  (p90 $0.041)   next step ~$0.003
📦   ~6 steps left  ·  3× web_search, 1× read_file expected
🧭   next: tool_call (p=0.65) → most likely "web_search"
⚠️   risk: 0.05 off-rails · 0.00 loop · 0.07 cost-spike
```

---

## Who this is for

Teams shipping LLM agents (LangChain, LangGraph, Autogen, raw SDKs) tired of:

- **\$1k surprise bills** from runaway loops that `max_steps=50` didn't catch
- **Spinners with no ETA** because progress bars on agent runs are a lie
- **Token-passthrough pricing** because per-customer cost is unbounded
- **Hand-rolled loop detection** that breaks every time the agent shape changes

If you already use Langfuse, migration is literally `export LANGFUSE_HOST=…`. If you don't, install in 30 seconds.

---

## What it predicts — five dimensions, one round-trip

| | answers |
|---|---|
| **time** | total / **remaining** seconds, next-step time, compute vs I/O split |
| **cost** | total / **remaining** USD, next-step USD, **per-model breakdown** |
| **resources** | tokens (prompt + completion), steps remaining, **per-tool call counts** |
| **next action** | distribution over `{generation, tool_call, end}`, top-k next tools, likely model |
| **risk** | `off-rails` · `loop` · `context-overflow` · `budget-overshoot` · `cost-spike` |

All five come from one `trace.predict()` call against a shared kNN cohort, so the numbers are internally consistent (cost-remaining never disagrees with steps-remaining × per-step rate).

---

## Migrate from Langfuse — one line

**Zero code change** — point your env var at us:

```bash
export LANGFUSE_HOST=http://localhost:7187
```

Your existing `langfuse.Langfuse()` calls now write to Langpred; predictions appear immediately.

**One import change** — get the new methods in code:

```diff
- from langfuse import Langfuse
+ from langpred.langfuse_compat import Langfuse
```

Same constructor, same `trace / span / generation / @observe / flush`. New: `trace.predict()`, `trace.predict_next_action()`, `trace.set_budget()`, `trace.is_off_rails()`.

> ✓ **Validated against the real Langfuse SDK.** `tests/test_real_langfuse_integration.py` runs the actual pip-installed `langfuse` package against a live Langpred server. If upstream changes their batch envelope, CI catches it.

---

## Install — 30 seconds

```bash
git clone https://github.com/memovai/langpred && cd langpred
pip install -e ./server -e ./sdk-python
uvicorn langpred_server.main:app --port 7187 &
python examples/01_migrate_from_langfuse.py     # smoke test
```

SQLite by default. No Postgres, no Redis, no GPU.

---

## The three things that change how you build

### 1. Kill runaway loops *before* the dollar lands
```python
with trace.set_budget(usd=0.50, on_exceed="kill"):
    agent.run()   # raises BudgetExceeded when predicted total > $0.50
```
Replaces `max_steps=50`. Good runs finish, bad ones die early. The cap trips on **predicted** total spend, not realized — so you stop the bleeding one step before it lands.

### 2. Quote a fixed price upfront
```python
trace.generation(...)              # one cheap scout step
quote = trace.predict().cost.usd_total_p90 * 1.4
```
Sell agent features at a fixed price (Devin-style) without token-passthrough roulette. The 1.4× margin over p90 covers ~99% of tail outcomes.

### 3. Route on what the agent is *about* to do
```python
na = trace.predict_next_action()
if na.likely_next_model == "claude-opus-4-7":
    downgrade_to("claude-sonnet-4-6")
if na.most_likely_tool() == "web_search":
    prefetch_search_cache()
```
Pre-warm caches. Downgrade to a cheaper model when you can predict it's safe. Show smarter spinners ("looking things up…" vs "writing…").

---

## Seven runnable examples

| | shows |
|---|---|
| [`01_migrate_from_langfuse.py`](./examples/01_migrate_from_langfuse.py) | The exact one-import diff |
| [`02_budget_guard.py`](./examples/02_budget_guard.py) | Kill switch tripping mid-loop |
| [`03_eta_in_ui.py`](./examples/03_eta_in_ui.py) | Live ETA per step |
| [`04_upfront_pricing.py`](./examples/04_upfront_pricing.py) | Scout-then-quote pattern |
| [`05_next_action.py`](./examples/05_next_action.py) | Predict next tool / model |
| [`06_resource_forecast.py`](./examples/06_resource_forecast.py) | Full 5-dimension forecast |
| [`07_real_langfuse_sdk.py`](./examples/07_real_langfuse_sdk.py) | Drives Langpred with the **real** Langfuse SDK |

---

## How it works — the 30-second version

Every Langfuse event becomes part of a **trajectory**. For a partial trace we find the **k=20 nearest finished trajectories** (16-dim prefix feature L2; same-`trace.name` halved) and aggregate them five ways at once:

- final outcomes → quantile bands (time / cost / resources)
- step at `prefix_len+1` → next-action distribution
- per-tool histograms → expected remaining tool calls
- per-model cost split → downgrade target
- status field → off-rails risk

After 1000 finished traces per shape, gradient-boosted quantile regressors auto-promote for the scalar dimensions while distributions stay kNN. Zero GPU. Full architecture and pain-point research: **[DESIGN.md](./DESIGN.md)**.

---

## Status

| | |
|---|---|
| **Tests** | 16 / 16 green (including real-Langfuse SDK integration) |
| **Cost MAE** | \$0.10 on synthetic benchmark |
| **p90 coverage** | 79% (conformal calibration is on deck) |
| **Stack** | FastAPI + SQLite + scikit-learn · Python ≥3.10 · no GPU |

**Roadmap:** OTel ingest · dashboard UI · TypeScript SDK · multi-tenant auth · conformal calibration · optional proxy add-on for hard budget enforcement.

---

## Next steps

- **Read the design**: [DESIGN.md](./DESIGN.md) — pain-point research, prediction model trade-offs, why kNN-then-GBM and not transformers.
- **Run an example**: `python examples/06_resource_forecast.py` after starting the server.
- **Integrate**: change one env var; predictions land in the dashboard.

MIT — see [LICENSE](./LICENSE).

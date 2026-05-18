# Langpred — Design Doc

> **One-liner.** Langfuse looks **backward** (what did my agent do?). Langpred looks **forward** (what is it about to do, how long will it take, how much will it cost, and should I stop it now?). We ingest the same trajectories Langfuse already collects and emit predictions you can put in a UI, a price tag, or a kill switch.

---

## 1. Pain — what's actually broken in 2026

We talked to / read complaints from teams shipping agentic features (Cursor / Cline / Devin / Replit Agent / internal copilots / RAG-with-tools). The single highest-signal complaint is **"the agent surprised me"** — surprise in cost, surprise in duration, or surprise in behaviour. Concretely:

### 1.1 Cost runaway (the $10k bill thread)
Agentic loops have **no implicit token budget**. A bug in a planner can re-enter a tool 600 times and burn $1,200 in 90 minutes. Today's mitigations are awful:

- **Hardcoded `max_steps=50`** — kills useful runs and doesn't kill bad ones (50 dumb steps can still cost $80).
- **After-the-fact alerts** in Langfuse / Helicone — they tell you *yesterday's* bill, not *this run's projected* bill.
- **Pre-spend rate limits** at the provider (OpenAI / Anthropic) — coarse, org-level, do not understand "this customer" or "this feature".

What's missing: a **live budget that understands the trajectory shape** and trips *before* the dollar lands.

### 1.2 ETA UX (the spinner problem)
SaaS agent products show a spinner. The user has no idea whether they're 10% or 90% done. NPS data from agent-first products (Devin, Cursor Composer, Cognition) repeatedly cites "I don't know when to come back" as a top-3 frustration. Fixed-duration progress bars (`steps 3/50`) are a lie — most agents finish at step 7 or step 41.

What's missing: a **probabilistic ETA** (`p50: 38s, p90: 4m12s, p99: 18m`) that updates every step.

### 1.3 Upfront pricing (the SaaS-margin trap)
Agent products that pass through token cost can't quote fixed prices. Devin charges $500/mo flat and loses money on power users; Lindy / Relevance / others charge per-action because they're scared of token blow-ups. Customers want **"I'll pay $X to refactor this repo"** — not "spin the wheel and we'll send a Stripe invoice".

What's missing: a way to **quote a price at trace-start** with a calibrated confidence band, so the seller can margin-up p90 and absorb p99.

### 1.4 Runaway-loop detection
Stuck agents repeat near-identical tool calls. Current detection is hand-rolled per app (n-gram on tool names, last-3-calls equality). Brittle, doesn't generalise across agent shapes.

What's missing: a **trajectory-level "is this run going off the rails"** signal trained on past failures.

### 1.5 Cost attribution
Multi-tenant agent SaaS needs per-customer / per-feature / per-prompt-version cost. Langfuse can group, but only post-hoc. Decisions ("downgrade this customer to Haiku") need to be made *during* the request, not in a dashboard tomorrow.

### 1.6 Why Langfuse users specifically
Langfuse adoption is wide; the ingest format is already glue in many stacks. **The trajectory data Langfuse hoards is exactly the training data Langpred needs.** Asking these teams to instrument *again* is fatal — we have to be a drop-in. So: same `LANGFUSE_HOST` switch, same env vars, same `/api/public/ingestion` payload, same SDK call shape. Anything else loses the wedge.

---

## 2. Product — what Langpred does

```
                  ┌────────────────────────────────────────────────┐
                  │              your agent app code               │
                  │   (LangChain, LangGraph, Autogen, raw SDK…)    │
                  └────────────────┬───────────────────────────────┘
                                   │
                  langpred SDK (or langfuse-compat shim)
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────────┐
        │  POST /api/public/ingestion   (Langfuse-compatible)      │
        │  trace-create / span-create / generation-create …        │
        └──────────────────┬───────────────────────────────────────┘
                           ▼
                  ┌────────────────────┐
                  │   Trajectory DB    │  (events keyed by traceId)
                  └────────┬───────────┘
                           ▼
                  ┌────────────────────┐
                  │  Feature builder   │  prefix→feature vector
                  └────────┬───────────┘
                           ▼
            ┌────────────────────────────┐
            │  Predictor (kNN + GBM)     │  per-project models
            │  - cost remaining          │
            │  - steps remaining         │
            │  - wall time remaining     │
            │  - off-rails score         │
            └────────────┬───────────────┘
                         ▼
        ┌────────────────────────────────────────────┐
        │  GET  /api/public/predict/{traceId}/eta    │ → use in UI
        │  GET  /api/public/predict/{traceId}/cost   │ → upfront price
        │  POST /api/public/budgets                  │ → kill switch
        │  Webhooks: on_budget_breach, on_offrails   │
        └────────────────────────────────────────────┘
```

The user-facing API surface adds **four verbs** on top of Langfuse:

1. `trace.predict_eta()` → `{p50, p90, p99, steps_left_p50, …}`
2. `trace.predict_cost()` → `{usd_p50, usd_p90, usd_p99, breakdown}`
3. `trace.set_budget(usd=0.5, on_exceed="kill")` → returns a `BudgetGuard` you check (or raises `BudgetExceeded` inside a context manager)
4. `trace.is_off_rails(threshold=0.7)` → boolean / score

That's it. Everything else (`trace`, `span`, `generation`, `update`, `flush`) is **identical** to Langfuse and forwards 1:1.

---

## 3. Migration — the one-line story

For a team already on Langfuse, day-1 migration is a single env var:

```bash
# before
LANGFUSE_HOST=https://cloud.langfuse.com

# after
LANGFUSE_HOST=https://api.langpred.com   # or http://localhost:7187 for self-host
```

…and they keep using the Langfuse SDK they already have. Predictions are visible in our dashboard immediately, and dual-write to Langfuse Cloud is supported (`LANGPRED_MIRROR_TO=https://cloud.langfuse.com`) so they don't lose history.

Day-2: `pip install langpred` and replace the import to get the prediction methods.

```python
# day 2 — additive only
- from langfuse import Langfuse
+ from langpred.langfuse_compat import Langfuse      # drop-in

langfuse = Langfuse()
trace = langfuse.trace(name="research_agent")
# … existing code unchanged …

# NEW capabilities, ignore if you don't care:
eta  = trace.predict_eta()         # {"p50": 38.1, "p90": 252.0, ...}
cost = trace.predict_cost()        # {"usd_p50": 0.023, "usd_p90": 0.087}
trace.set_budget(usd=0.50, on_exceed="kill")
```

---

## 4. Prediction model — what's actually inside

We deliberately ship two tiers and let the data choose.

### Tier 0 — heuristic floor
Used in the first N traces of a project (cold start) and as a sanity floor everywhere:

- `cost_remaining_p50 = (median_cost_per_step_for_this_trace_name) × (median_steps_remaining_for_this_trace_name)`
- `cost_remaining_p90 = same with 90th-percentile multipliers`
- ETA replaces "cost" with "wall-time".

This is **always** computable, has no training, and beats `max_steps=50` on day one.

### Tier 1 — kNN on prefixes (default once ≥ 50 completed trajectories per `trace.name`)
For a partial trajectory of length *k*, build a **prefix feature vector**:

```
step_count                k
elapsed_ms                t_now - t_start
total_tokens_so_far       Σ prompt + completion
total_cost_so_far         Σ priced
tool_name_history_hash    locality-sensitive hash of last 8 tool names
unique_tool_count         |distinct tools|
last_tool_repeat_streak   how many of last-N steps used the same tool (loop signal)
avg_latency_per_step      EMA
prompt_size_growth_slope  linear regression on prompt sizes over time
```

Find the **k=20 nearest completed trajectories** with the same `trace.name` whose prefix at length *k* is closest under L2 in the feature space. Their *final* outcomes give us the empirical distribution → we report p50 / p90 / p99 from the quantiles.

Why kNN first: zero-train, interpretable ("we predict $0.18 because trace-id `abc123` and 19 others looked just like this and cost $0.16–$0.22"), and incrementally updated by appending to an index.

### Tier 2 — gradient-boosted regressors (auto-promoted at ≥ 1,000 trajectories per shape)
Three independent regressors, each `prefix_features → final outcome`:

- `GBR_cost` → total USD
- `GBR_steps` → total step count
- `GBR_time` → total wall time

Quantile regression heads at τ ∈ {0.5, 0.9, 0.99} give us calibrated bands. We use scikit-learn `HistGradientBoostingRegressor(loss="quantile")` — small, no GPU, retrains in seconds.

`off_rails_score` is a binary classifier: prefix features → P(trajectory ended in `error`, `cancelled`, or `steps > p99_for_shape`).

### What we did **not** do
- **No transformer.** Sequence transformers are overkill for the data volumes most teams have, expensive to host, and not interpretable. We leave it on the roadmap behind a flag for >10⁵ trajectories per shape.
- **No LLM-based "ask Claude to estimate"**. That's a circular dependency (predicting LLM cost with an LLM) and unbounded latency.

---

## 5. Budget enforcement — how it actually kills a run

The naive design ("on each step, ask the server `is_over_budget?`") adds 50–200ms per step. We do it differently:

1. SDK starts the trace and POSTs `set_budget(usd=0.5, on_exceed="kill")`. Server registers the budget and returns an `enforcement_token`.
2. On every step, the SDK includes the running cost in the ingestion payload (already does, for token counts).
3. Server runs the predictor **asynchronously** on each new event. If `spent_so_far + predicted_remaining_p50 > cap`, it sets `budgets[trace_id].breached = True` and (a) sends a webhook, (b) returns `X-Langpred-Budget: breached` on the *next* ingestion response.
4. SDK reads that header (or, in synchronous mode, calls a fast `GET /budgets/{trace_id}/status` between steps) and raises `BudgetExceeded`.

This adds **zero** synchronous round-trips in the happy path. When the budget is tripped, the SDK reacts on the next event boundary — typically <1 step of latency.

Three `on_exceed` modes:
- `"kill"` — raise `BudgetExceeded` inside the SDK.
- `"downgrade"` — emit a callback the app handles (swap model, shrink context).
- `"warn"` — just fire the webhook, don't interrupt.

---

## 6. Data model

We mirror Langfuse's event types so ingestion is byte-compatible. Internal representation:

```
Trace(id, project_id, name, user_id, session_id, start_ts, end_ts, status, metadata)
Observation(id, trace_id, parent_id, kind, name, start_ts, end_ts,
            model, prompt_tokens, completion_tokens, input, output, level, status_message)
Score(id, trace_id, observation_id, name, value, comment)
Budget(id, trace_id, cap_usd, on_exceed, status, spent_so_far, predicted_remaining_p50, …)
Prediction(trace_id, step_count, kind, p50, p90, p99, created_at)
```

Persisted by default to SQLite for self-hosters (zero ops). The schema is split into a writeable `events` log + a `materialised` view that the predictor reads, so the predictor never blocks ingestion.

---

## 7. What's in this repo

```
langpred/
├── DESIGN.md                       ← you are here
├── README.md                       ← quickstart + migration
├── docker-compose.yml              ← one-command self-host
├── server/                         ← FastAPI ingestion + prediction
│   └── langpred_server/
│       ├── ingest.py               ← Langfuse-compatible /api/public/ingestion
│       ├── predict.py              ← /api/public/predict/{trace_id}/*
│       ├── budget.py               ← /api/public/budgets
│       ├── trajectories.py         ← event-log → Trajectory
│       └── ml/
│           ├── featurize.py        ← prefix feature builder
│           ├── knn.py              ← Tier-1 predictor
│           ├── gbm.py              ← Tier-2 predictor (HistGBR quantile)
│           └── pricing.py          ← token → USD lookup table
├── sdk-python/
│   └── langpred/
│       ├── client.py
│       ├── trace.py                ← Langfuse-shape Trace/Span/Generation + predict_*
│       ├── transport.py            ← batched HTTP with retry
│       └── langfuse_compat/        ← `from langpred.langfuse_compat import Langfuse`
├── examples/
│   ├── 01_migrate_from_langfuse.py ← before / after diff
│   ├── 02_budget_guard.py          ← protect production
│   ├── 03_eta_in_ui.py             ← show a real progress bar
│   └── 04_upfront_pricing.py       ← quote a fixed price to a customer
├── benchmarks/
│   └── eval_predictions.py         ← MAE + calibration on synthetic traces
└── tests/
```

---

## 8. Non-goals (for now)

- Full Langfuse UI parity. We ship a minimal dashboard; users keep Langfuse for drill-down (or use mirror mode).
- Eval / scoring / experiments (Langfuse owns this; we focus on prediction).
- Auto-prompt-optimisation. Adjacent, but a different product.
- Anything that requires GPU.

---

## 9. Open questions / next decisions

- **Cross-project transfer learning.** Can a new project bootstrap from a similar one's model? (Probably yes for `trace.name`-shape clusters; needs work.)
- **Online learning loop.** Right now we retrain on a schedule. Going incremental would tighten cold-start.
- **OTel ingest.** Langfuse itself is pivoting to OTel. We should accept OTel spans in v2 of the ingest endpoint.
- **Price sources.** We hard-code the per-model price table — fine, but it goes stale. Pull from a community source?

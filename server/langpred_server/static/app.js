const state = {
  view: "traces",
  traces: [],
  selectedId: null,
  auth: {
    publicKey: localStorage.getItem("langpred.publicKey") || "",
    secretKey: localStorage.getItem("langpred.secretKey") || "",
  },
};

const els = {
  pageTitle: document.getElementById("pageTitle"),
  statusLine: document.getElementById("statusLine"),
  notice: document.getElementById("notice"),
  publicKey: document.getElementById("publicKey"),
  secretKey: document.getElementById("secretKey"),
  projectForm: document.getElementById("projectForm"),
  tracesView: document.getElementById("tracesView"),
  forecastView: document.getElementById("forecastView"),
  traceList: document.getElementById("traceList"),
  detailPane: document.getElementById("detailPane"),
  searchInput: document.getElementById("searchInput"),
  statusFilter: document.getElementById("statusFilter"),
  refreshBtn: document.getElementById("refreshBtn"),
  rebuildBtn: document.getElementById("rebuildBtn"),
  metricTraces: document.getElementById("metricTraces"),
  metricOpen: document.getElementById("metricOpen"),
  metricCost: document.getElementById("metricCost"),
  metricRisk: document.getElementById("metricRisk"),
  forecastForm: document.getElementById("forecastForm"),
  forecastName: document.getElementById("forecastName"),
  forecastMetadata: document.getElementById("forecastMetadata"),
  forecastInput: document.getElementById("forecastInput"),
  forecastResult: document.getElementById("forecastResult"),
};

els.publicKey.value = state.auth.publicKey;
els.secretKey.value = state.auth.secretKey;

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    setView(button.dataset.view);
  });
});

els.projectForm.addEventListener("submit", (event) => {
  event.preventDefault();
  state.auth.publicKey = els.publicKey.value.trim();
  state.auth.secretKey = els.secretKey.value;
  localStorage.setItem("langpred.publicKey", state.auth.publicKey);
  localStorage.setItem("langpred.secretKey", state.auth.secretKey);
  state.selectedId = null;
  loadTraces();
});

els.refreshBtn.addEventListener("click", () => {
  if (state.view === "traces") {
    loadTraces();
  }
});

els.rebuildBtn.addEventListener("click", async () => {
  try {
    setBusy("Rebuilding prediction models.");
    await apiFetch("/api/local/rebuild", { method: "POST" });
    await loadTraces();
    showNotice("");
  } catch (error) {
    showNotice(error.message);
  }
});

els.searchInput.addEventListener("input", debounce(loadTraces, 180));
els.statusFilter.addEventListener("change", loadTraces);

els.forecastForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runForecast();
});

loadTraces();

function setView(view) {
  state.view = view;
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  els.tracesView.hidden = view !== "traces";
  els.forecastView.hidden = view !== "forecast";
  els.pageTitle.textContent = view === "forecast" ? "Forecast" : "Traces";
  els.statusLine.textContent =
    view === "forecast"
      ? "Quote a run before trace creation."
      : `Project ${projectLabel()}`;
}

async function loadTraces() {
  try {
    const params = new URLSearchParams();
    if (els.searchInput.value.trim()) params.set("q", els.searchInput.value.trim());
    params.set("status", els.statusFilter.value);
    params.set("limit", "100");
    const data = await apiFetch(`/api/local/traces?${params.toString()}`);
    state.traces = data.traces || [];
    renderMetrics(data.summary || {});
    els.statusLine.textContent = `Project ${data.project_id || projectLabel()} - ${data.count || 0} traces`;
    if (!state.traces.some((trace) => trace.id === state.selectedId)) {
      state.selectedId = state.traces[0] ? state.traces[0].id : null;
    }
    renderTraceList();
    if (state.selectedId) {
      await selectTrace(state.selectedId, { preserveList: true });
    } else {
      renderEmptyDetail();
    }
    showNotice("");
  } catch (error) {
    showNotice(error.message);
  }
}

async function selectTrace(traceId, options = {}) {
  state.selectedId = traceId;
  if (!options.preserveList) {
    renderTraceList();
  }
  try {
    const detail = await apiFetch(`/api/local/traces/${encodeURIComponent(traceId)}`);
    renderDetail(detail);
    showNotice("");
  } catch (error) {
    showNotice(error.message);
  }
}

function renderMetrics(summary) {
  els.metricTraces.textContent = number(summary.traces || 0, 0);
  els.metricOpen.textContent = number(summary.open_traces || 0, 0);
  els.metricCost.textContent = money(summary.total_cost_usd || 0);
  els.metricRisk.textContent = number(summary.high_risk_traces || 0, 0);
}

function renderTraceList() {
  if (!state.traces.length) {
    els.traceList.innerHTML = `
      <div class="empty-state">
        <h2>No traces</h2>
        <p>Start the server, point Langfuse traffic here, then refresh.</p>
      </div>
    `;
    return;
  }

  els.traceList.innerHTML = state.traces
    .map((trace) => {
      const pred = trace.prediction || {};
      const risk = pred.risk || 0;
      const statusClass = trace.status || "open";
      return `
        <button class="trace-row ${trace.id === state.selectedId ? "active" : ""}" data-trace-id="${escapeAttr(trace.id)}" type="button">
          <div class="row-top">
            <span class="row-title">${escapeHtml(trace.name || "unnamed trace")}</span>
            <span class="pill ${statusClass}">${escapeHtml(trace.status || "open")}</span>
          </div>
          <div class="row-meta">
            <span class="muted">${escapeHtml(shortId(trace.id))}</span>
            ${trace.user_id ? `<span class="pill">${escapeHtml(trace.user_id)}</span>` : ""}
            ${trace.session_id ? `<span class="pill">${escapeHtml(trace.session_id)}</span>` : ""}
          </div>
          <div class="row-stats">
            <span>${number(trace.step_count, 0)} steps</span>
            <span>${money(trace.total_usd || 0)}</span>
            <span>${duration(pred.remaining_seconds_p50 || 0)} left</span>
            <span class="pill ${risk >= 0.7 ? "high" : risk >= 0.35 ? "warn" : ""}">risk ${percent(risk)}</span>
          </div>
        </button>
      `;
    })
    .join("");

  els.traceList.querySelectorAll(".trace-row").forEach((row) => {
    row.addEventListener("click", () => selectTrace(row.dataset.traceId));
  });
}

function renderDetail(payload) {
  const trace = payload.trace;
  const pred = payload.prediction;
  const budget = payload.budget;
  const observations = trace.observations || [];

  els.detailPane.innerHTML = `
    <div class="detail-header">
      <div class="row-top">
        <h2>${escapeHtml(trace.name || "unnamed trace")}</h2>
        <span class="pill ${escapeAttr(trace.status || "open")}">${escapeHtml(trace.status || "open")}</span>
      </div>
      <div class="detail-meta">
        <span class="muted">${escapeHtml(trace.id)}</span>
        <span>${number(trace.step_count, 0)} steps</span>
        <span>${number(trace.total_tokens, 0)} tokens</span>
        <span>${money(trace.total_usd || 0)}</span>
        <span>${duration(trace.elapsed_seconds || 0)} elapsed</span>
      </div>
    </div>

    ${renderPrediction(pred)}
    ${renderBudget(trace, budget)}

    <div class="split">
      <section>
        <h3 class="section-title">Observations</h3>
        ${renderTimeline(observations)}
      </section>
      <aside>
        <h3 class="section-title">Next Action</h3>
        ${renderNextAction(pred)}
        <h3 class="section-title">Trace Data</h3>
        <div class="kv-grid">
          <pre class="json-block">${escapeHtml(formatJson(trace.input))}</pre>
          <pre class="json-block">${escapeHtml(formatJson(trace.metadata))}</pre>
        </div>
      </aside>
    </div>
  `;

  const budgetForm = document.getElementById("budgetForm");
  if (budgetForm) {
    budgetForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      await saveBudget(trace.id);
    });
  }
}

function renderPrediction(pred) {
  if (!pred) {
    return `<div class="notice">Prediction unavailable for this trace.</div>`;
  }
  const risk = pred.risk || {};
  const maxRisk = Math.max(
    risk.offrails_risk || 0,
    risk.loop_risk || 0,
    risk.context_overflow_risk || 0,
    risk.budget_overshoot_risk || 0,
    risk.cost_spike_risk || 0
  );
  return `
    <section class="prediction-grid">
      <div class="prediction-card"><span>ETA p50</span><strong>${duration(pred.time.remaining_seconds_p50 || 0)}</strong></div>
      <div class="prediction-card"><span>Cost p90</span><strong>${money(pred.cost.usd_total_p90 || 0)}</strong></div>
      <div class="prediction-card"><span>Steps left</span><strong>${number(pred.resources.steps_remaining_p50 || 0, 1)}</strong></div>
      <div class="prediction-card"><span>Risk</span><strong>${percent(maxRisk)}</strong></div>
      <div class="prediction-card"><span>Model tier</span><strong>${escapeHtml(pred.meta.tier)}</strong></div>
    </section>
    ${pred.risk.notes && pred.risk.notes.length ? `<div class="chips">${pred.risk.notes.map((note) => `<span class="pill warn">${escapeHtml(note)}</span>`).join("")}</div>` : ""}
  `;
}

function renderBudget(trace, budget) {
  return `
    <h3 class="section-title">Budget Guard</h3>
    <form class="budget-form" id="budgetForm">
      <label>
        Cap USD
        <input id="budgetCap" type="number" min="0.0001" step="0.0001" value="${budget ? escapeAttr(String(budget.cap_usd)) : ""}" placeholder="0.50" required />
      </label>
      <label>
        Action
        <select id="budgetAction">
          ${option("kill", budget && budget.on_exceed)}
          ${option("scope_reduce", budget && budget.on_exceed)}
          ${option("warn", budget && budget.on_exceed)}
        </select>
      </label>
      <label>
        Quantile
        <select id="budgetQuantile">
          ${option("p50", budget && budget.quantile)}
          ${option("p90", budget && budget.quantile)}
          ${option("p99", budget && budget.quantile)}
        </select>
      </label>
      <button type="submit">Save</button>
    </form>
    ${budget ? `<div class="chips"><span class="pill ${budget.breached ? "high" : "ok"}">${budget.breached ? "breached" : "active"}</span><span class="muted">${escapeHtml(budget.breach_reason || "no breach")}</span></div>` : ""}
  `;
}

function renderTimeline(observations) {
  if (!observations.length) {
    return `<div class="empty-state"><h2>No observations</h2><p>This trace has no spans, generations, or events yet.</p></div>`;
  }
  return `
    <div class="timeline">
      ${observations
        .map((step) => `
          <div class="timeline-row">
            <span class="pill">${number(step.index + 1, 0)}</span>
            <div class="timeline-name">${escapeHtml(step.name || step.tool_name || step.kind)}</div>
            <span class="hide-small">${escapeHtml(step.kind)}</span>
            <span class="hide-small">${duration((step.latency_ms || 0) / 1000)}</span>
            <span class="hide-small">${money(step.usd || 0)}</span>
          </div>
        `)
        .join("")}
    </div>
  `;
}

function renderNextAction(pred) {
  if (!pred) {
    return `<pre class="json-block">No prediction</pre>`;
  }
  const next = pred.next || {};
  const distribution = next.next_kind_distribution || {};
  const tools = next.top_next_tools || [];
  const rows = Object.entries(distribution)
    .sort((a, b) => b[1] - a[1])
    .map(([kind, probability]) => `<span class="pill">${escapeHtml(kind)} ${percent(probability)}</span>`)
    .join("");
  const toolRows = tools
    .map((tool) => `<span class="pill">${escapeHtml(tool.tool)} ${percent(tool.probability)}</span>`)
    .join("");
  return `
    <div class="chips">${rows || '<span class="muted">No next action distribution</span>'}</div>
    <pre class="json-block">${escapeHtml(formatJson({
      likely_next_model: next.likely_next_model,
      expected_next_step_usd_p50: next.expected_next_step_usd_p50,
      expected_next_step_seconds_p50: next.expected_next_step_seconds_p50,
      top_next_tools: tools,
    }))}</pre>
    <div class="chips">${toolRows}</div>
  `;
}

function renderEmptyDetail() {
  els.detailPane.innerHTML = `
    <div class="empty-state">
      <h2>No trace selected</h2>
      <p>Ingest traces and they will appear here with live predictions.</p>
    </div>
  `;
}

async function saveBudget(traceId) {
  try {
    const cap = Number(document.getElementById("budgetCap").value);
    const onExceed = document.getElementById("budgetAction").value;
    const quantile = document.getElementById("budgetQuantile").value;
    await apiFetch("/api/public/budgets", {
      method: "POST",
      body: JSON.stringify({
        trace_id: traceId,
        cap_usd: cap,
        on_exceed: onExceed,
        quantile,
      }),
    });
    await selectTrace(traceId, { preserveList: true });
  } catch (error) {
    showNotice(error.message);
  }
}

async function runForecast() {
  try {
    const traceName = els.forecastName.value.trim();
    const metadata = parseLooseJson(els.forecastMetadata.value);
    const input = parseLooseJson(els.forecastInput.value);
    const pred = await apiFetch("/api/public/forecast", {
      method: "POST",
      body: JSON.stringify({
        trace_name: traceName,
        metadata,
        input,
      }),
    });
    els.forecastResult.innerHTML = `
      ${renderPrediction(pred)}
      <h3 class="section-title">Cost by Model</h3>
      <div class="chips">
        ${(pred.cost.usd_by_model || []).map((item) => `<span class="pill">${escapeHtml(item.model)} ${money(item.usd_p90)}</span>`).join("") || '<span class="muted">No model split yet</span>'}
      </div>
      <h3 class="section-title">Raw Forecast</h3>
      <pre class="json-block">${escapeHtml(formatJson(pred))}</pre>
    `;
    showNotice("");
  } catch (error) {
    showNotice(error.message);
  }
}

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (options.body) {
    headers.set("Content-Type", "application/json");
  }
  const authHeader = authValue();
  if (authHeader) {
    headers.set("Authorization", authHeader);
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) {
      // keep HTTP message
    }
    throw new Error(message);
  }
  const text = await response.text();
  return text ? JSON.parse(text) : {};
}

function authValue() {
  if (!state.auth.publicKey && !state.auth.secretKey) {
    return "";
  }
  return `Basic ${btoa(`${state.auth.publicKey}:${state.auth.secretKey}`)}`;
}

function projectLabel() {
  return state.auth.publicKey || "default";
}

function setBusy(message) {
  els.statusLine.textContent = message;
}

function showNotice(message) {
  if (!message) {
    els.notice.hidden = true;
    els.notice.textContent = "";
    return;
  }
  els.notice.hidden = false;
  els.notice.textContent = message;
}

function parseLooseJson(value) {
  const trimmed = value.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed);
  } catch (_) {
    return trimmed;
  }
}

function formatJson(value) {
  if (value === null || value === undefined || value === "") {
    return "null";
  }
  return JSON.stringify(value, null, 2);
}

function option(value, selected) {
  return `<option value="${escapeAttr(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(value)}</option>`;
}

function duration(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  if (s < 1) return "0s";
  if (s < 60) return `${Math.round(s)}s`;
  const minutes = Math.floor(s / 60);
  const rest = Math.round(s % 60);
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function money(value) {
  return `$${(Number(value) || 0).toFixed(4)}`;
}

function number(value, digits = 0) {
  return (Number(value) || 0).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function percent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function shortId(value) {
  if (!value) return "";
  return value.length > 14 ? `${value.slice(0, 7)}...${value.slice(-5)}` : value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#96;");
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), wait);
  };
}

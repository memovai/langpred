"""Client-side budget guard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .trace import Trace
    from .transport import Transport


class BudgetExceeded(RuntimeError):
    """Raised inside a :class:`BudgetGuard` when the budget is breached."""

    def __init__(self, trace_id: str, status: dict[str, Any]):
        self.trace_id = trace_id
        self.status = status
        super().__init__(
            f"Langpred budget exceeded for trace {trace_id}: "
            f"{status.get('breach_reason') or 'cap reached'}"
        )


@dataclass
class BudgetGuard:
    """Context manager that polls the budget status and raises on breach.

    The transport already reads the ``X-Langpred-Budget`` header on every
    ingestion response, so most checks are zero-RTT. ``check()`` falls back to
    a GET when no recent header has been seen.
    """

    trace: "Trace"
    transport: "Transport"
    cap_usd: float
    on_exceed: str
    quantile: str = "p50"

    # ------------------------------------------------------------- check

    def check(self) -> dict[str, Any]:
        # Cheap path: SDK transport saved the last response headers — if the
        # server flagged a breach, react immediately.
        h = self.transport.last_headers
        breached_header = h.get("x-langpred-budget", h.get("X-Langpred-Budget"))
        breached_traces = h.get(
            "x-langpred-budget-traces", h.get("X-Langpred-Budget-Traces", "")
        )
        if breached_header == "breached" and self.trace.id in breached_traces.split(","):
            return self._fetch_and_maybe_raise()
        # Otherwise, on the slow path, ask the server.
        return self._fetch_and_maybe_raise()

    def _fetch_and_maybe_raise(self) -> dict[str, Any]:
        path = f"/api/public/budgets/{self.trace.id}/status"
        try:
            body = self.transport.get(path)
        except Exception:
            return {"breached": False, "trace_id": self.trace.id}
        if body.get("breached") and self.on_exceed == "kill":
            raise BudgetExceeded(self.trace.id, body)
        return body

    # ----------------------------------------------------- context manager

    def __enter__(self) -> "BudgetGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # On exit, do a final check so the caller can react if a late-breaking
        # event just tripped the budget.
        try:
            self.check()
        except BudgetExceeded:
            if exc_type is None:
                raise
            # If we're already propagating an exception, don't shadow it.

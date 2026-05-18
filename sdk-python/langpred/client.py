"""Top-level Langpred client."""
from __future__ import annotations

from typing import Any

from ._ids import env
from .predict import AgentPrediction
from .trace import Trace
from .transport import Transport


_DEFAULT_HOST = "http://localhost:7187"


class Langpred:
    """High-level client. Mirrors ``langfuse.Langfuse`` constructor + adds
    prediction methods through the returned :class:`Trace`."""

    def __init__(
        self,
        *,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        flush_at: int = 30,
        flush_interval: float = 1.0,
        timeout: float = 10.0,
    ) -> None:
        host = host or env("LANGPRED_HOST", "LANGFUSE_HOST", default=_DEFAULT_HOST)
        public_key = public_key or env("LANGPRED_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or env("LANGPRED_SECRET_KEY", "LANGFUSE_SECRET_KEY")
        self.transport = Transport(
            host=host or _DEFAULT_HOST,
            public_key=public_key,
            secret_key=secret_key,
            flush_at=flush_at,
            flush_interval_seconds=flush_interval,
            timeout=timeout,
        )

    # ----------------------------------------------------------- factories

    def trace(self, **kwargs: Any) -> Trace:
        return Trace(self.transport, **kwargs)

    # ------------------------------------------------------- pre-trace forecast

    def forecast(
        self,
        trace_name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any | None = None,
    ) -> AgentPrediction:
        """Forecast a hypothetical trace **before** creating it. Powers
        reject-upfront ("don't even start, predicted cost is too high") and
        route-at-start ("use Sonnet, not Opus, given the cohort profile")
        — both safe to act on because no KV cache exists yet.

        Returns the same :class:`AgentPrediction` shape as :meth:`Trace.predict`.
        """
        body = self.transport.post(
            "/api/public/forecast",
            {
                "trace_name": trace_name,
                "user_id": user_id,
                "session_id": session_id,
                "input": input,
            },
        )
        return AgentPrediction.from_response(body)

    # ------------------------------------------------------- lifecycle ops

    def flush(self) -> None:
        self.transport.flush()

    def shutdown(self) -> None:
        self.transport.shutdown()

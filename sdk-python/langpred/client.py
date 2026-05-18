"""Top-level Langpred client."""
from __future__ import annotations

from typing import Any

from ._ids import env
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

    # ------------------------------------------------------- lifecycle ops

    def flush(self) -> None:
        self.transport.flush()

    def shutdown(self) -> None:
        self.transport.shutdown()

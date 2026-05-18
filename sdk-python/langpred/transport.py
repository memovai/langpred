"""
HTTP transport for the Langpred / Langfuse-compatible SDK.

Buffer events in memory, flush either on size threshold, time threshold, or
explicit :func:`flush`. We expose the most recent response headers so
:mod:`langpred.budget` can read ``X-Langpred-Budget`` without an extra GET.
"""
from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx


log = logging.getLogger("langpred")


def _iso(ts: datetime | None = None) -> str:
    return (ts or datetime.now(timezone.utc)).isoformat()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
    return obj


class Transport:
    """Buffered HTTP transport for ingestion + prediction calls."""

    def __init__(
        self,
        host: str,
        public_key: str | None = None,
        secret_key: str | None = None,
        flush_at: int = 30,
        flush_interval_seconds: float = 1.0,
        timeout: float = 10.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.public_key = public_key or ""
        self.secret_key = secret_key or ""
        self.flush_at = flush_at
        self.flush_interval = flush_interval_seconds
        self.timeout = timeout
        self._buf: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._client = httpx.Client(timeout=timeout)
        self._last_response_headers: dict[str, str] = {}
        self._last_flush = time.monotonic()
        self._stopped = False
        # Best-effort flush at interpreter shutdown.
        atexit.register(self.shutdown)

    # ----------------------------------------------------------------- write

    def enqueue(self, event_type: str, body: dict[str, Any]) -> None:
        from ._ids import new_event_id

        ev = {
            "id": new_event_id(),
            "type": event_type,
            "timestamp": _iso(),
            "body": _json_safe(body),
        }
        flush_now = False
        with self._lock:
            self._buf.append(ev)
            if len(self._buf) >= self.flush_at:
                flush_now = True
            elif (time.monotonic() - self._last_flush) > self.flush_interval:
                flush_now = True
        if flush_now:
            self.flush()

    # ----------------------------------------------------------------- flush

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                self._last_flush = time.monotonic()
                return
            batch, self._buf = self._buf, []
        try:
            url = f"{self.host}/api/public/ingestion"
            r = self._client.post(
                url,
                json={"batch": batch},
                auth=(self.public_key, self.secret_key) if self.public_key else None,
            )
            self._last_response_headers = dict(r.headers)
            if r.status_code >= 400 and r.status_code != 207:
                log.warning("Langpred ingestion %s: %s", r.status_code, r.text[:200])
        except Exception:  # pragma: no cover
            log.exception("Langpred ingestion failed")
        finally:
            self._last_flush = time.monotonic()

    # --------------------------------------------------------------- shutdown

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self.flush()
        finally:
            try:
                self._client.close()
            except Exception:
                pass

    # ----------------------------------------------------------------- read

    def get(self, path: str) -> dict[str, Any]:
        self.flush()
        url = f"{self.host}{path}"
        r = self._client.get(
            url, auth=(self.public_key, self.secret_key) if self.public_key else None
        )
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.flush()
        url = f"{self.host}{path}"
        r = self._client.post(
            url,
            json=body,
            auth=(self.public_key, self.secret_key) if self.public_key else None,
        )
        r.raise_for_status()
        return r.json()

    # ---------------------------------------------------------------- headers

    @property
    def last_headers(self) -> dict[str, str]:
        return dict(self._last_response_headers)

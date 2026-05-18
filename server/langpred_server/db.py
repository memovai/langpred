"""
Minimal storage layer for Langpred.

By default we use an **in-memory** store so the test suite + examples run with
zero ops. When `LANGPRED_DATABASE_URL` is set to `sqlite:///path.db` we
serialise the same dicts to a single SQLite file via stdlib `sqlite3` — no
SQLAlchemy dependency, keeps the container small.

The store is intentionally a thin dict-of-events; the predictor reads from
:func:`Trajectory` built on top of it. We keep one big append-only event log
plus three indices (`by_trace`, `traces`, `budgets`) that are rebuilt on
restart from the log.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from .settings import SETTINGS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json_safe(obj: Any) -> Any:
    """Make pydantic / datetime / dict mix JSON-serialisable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


@dataclass
class StoredEvent:
    id: str
    type: str
    trace_id: str | None
    observation_id: str | None
    timestamp: str
    body: dict[str, Any]

    def to_row(self) -> tuple[str, str, str | None, str | None, str, str]:
        return (
            self.id,
            self.type,
            self.trace_id,
            self.observation_id,
            self.timestamp,
            json.dumps(_to_json_safe(self.body)),
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "StoredEvent":
        return cls(
            id=row[0],
            type=row[1],
            trace_id=row[2],
            observation_id=row[3],
            timestamp=row[4],
            body=json.loads(row[5]) if row[5] else {},
        )


@dataclass
class BudgetRecord:
    trace_id: str
    cap_usd: float
    on_exceed: str
    breached: bool = False
    breach_reason: str | None = None
    last_spent_usd: float = 0.0
    last_predicted_remaining_p50: float = 0.0
    last_predicted_remaining_p90: float = 0.0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


class Store:
    """Thread-safe in-memory + (optional) sqlite event log."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or SETTINGS.database_url
        self._lock = threading.RLock()
        self._events: list[StoredEvent] = []
        self._by_trace: dict[str, list[int]] = {}  # trace_id -> indices
        self._budgets: dict[str, BudgetRecord] = {}
        self._sqlite: sqlite3.Connection | None = None
        if self.database_url.startswith("sqlite"):
            self._init_sqlite(self.database_url)
            self._reload_from_sqlite()

    # ------------------------------------------------------------------ sqlite

    def _sqlite_path(self, url: str) -> str:
        # supports sqlite:///rel/path.db and sqlite:////abs/path.db and :memory:
        prefix = "sqlite:///"
        if url.startswith("sqlite:////"):
            return "/" + url[len(prefix):]
        if url.startswith(prefix):
            return url[len(prefix):]
        if url == "sqlite://:memory:":
            return ":memory:"
        return url

    def _init_sqlite(self, url: str) -> None:
        path = self._sqlite_path(url)
        self._sqlite = sqlite3.connect(path, check_same_thread=False)
        self._sqlite.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                trace_id TEXT,
                observation_id TEXT,
                ts TEXT NOT NULL,
                body TEXT NOT NULL
            )
            """
        )
        self._sqlite.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                trace_id TEXT PRIMARY KEY,
                cap_usd REAL NOT NULL,
                on_exceed TEXT NOT NULL,
                breached INTEGER NOT NULL,
                breach_reason TEXT,
                last_spent_usd REAL NOT NULL,
                last_pred_remaining_p50 REAL NOT NULL,
                last_pred_remaining_p90 REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._sqlite.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_trace ON events(trace_id)"
        )
        self._sqlite.commit()

    def _reload_from_sqlite(self) -> None:
        assert self._sqlite is not None
        cur = self._sqlite.execute(
            "SELECT id, type, trace_id, observation_id, ts, body FROM events ORDER BY ts ASC"
        )
        for row in cur.fetchall():
            ev = StoredEvent.from_row(row)
            self._events.append(ev)
            if ev.trace_id:
                self._by_trace.setdefault(ev.trace_id, []).append(len(self._events) - 1)
        cur = self._sqlite.execute("SELECT * FROM budgets")
        for row in cur.fetchall():
            (tid, cap, ox, br, br_r, sp, p50, p90, ca, ua) = row
            self._budgets[tid] = BudgetRecord(
                trace_id=tid,
                cap_usd=cap,
                on_exceed=ox,
                breached=bool(br),
                breach_reason=br_r,
                last_spent_usd=sp,
                last_predicted_remaining_p50=p50,
                last_predicted_remaining_p90=p90,
                created_at=ca,
                updated_at=ua,
            )

    # ------------------------------------------------------------------ events

    def append_event(
        self,
        ev_id: str,
        ev_type: str,
        trace_id: str | None,
        observation_id: str | None,
        ts: str,
        body: dict[str, Any],
    ) -> StoredEvent:
        ev = StoredEvent(
            id=ev_id,
            type=ev_type,
            trace_id=trace_id,
            observation_id=observation_id,
            timestamp=ts,
            body=body,
        )
        with self._lock:
            self._events.append(ev)
            if trace_id:
                self._by_trace.setdefault(trace_id, []).append(len(self._events) - 1)
            if self._sqlite is not None:
                try:
                    self._sqlite.execute(
                        "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?)",
                        ev.to_row(),
                    )
                    self._sqlite.commit()
                except sqlite3.Error:
                    # tolerate duplicates / closed connections
                    pass
        return ev

    def events_for_trace(self, trace_id: str) -> list[StoredEvent]:
        with self._lock:
            idx = self._by_trace.get(trace_id, [])
            return [self._events[i] for i in idx]

    def all_trace_ids(self) -> list[str]:
        with self._lock:
            return list(self._by_trace.keys())

    def trace_count(self) -> int:
        with self._lock:
            return len(self._by_trace)

    # ----------------------------------------------------------------- budgets

    def set_budget(self, b: BudgetRecord) -> None:
        with self._lock:
            self._budgets[b.trace_id] = b
            self._persist_budget(b)

    def get_budget(self, trace_id: str) -> BudgetRecord | None:
        with self._lock:
            return self._budgets.get(trace_id)

    def update_budget(self, b: BudgetRecord) -> None:
        with self._lock:
            b.updated_at = _now_iso()
            self._budgets[b.trace_id] = b
            self._persist_budget(b)

    def _persist_budget(self, b: BudgetRecord) -> None:
        if self._sqlite is None:
            return
        try:
            self._sqlite.execute(
                "INSERT OR REPLACE INTO budgets VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    b.trace_id,
                    b.cap_usd,
                    b.on_exceed,
                    int(b.breached),
                    b.breach_reason,
                    b.last_spent_usd,
                    b.last_predicted_remaining_p50,
                    b.last_predicted_remaining_p90,
                    b.created_at,
                    b.updated_at,
                ),
            )
            self._sqlite.commit()
        except sqlite3.Error:
            pass

    # ----------------------------------------------------------------- helpers

    def reset(self) -> None:
        """Test-only helper."""
        with self._lock:
            self._events.clear()
            self._by_trace.clear()
            self._budgets.clear()
            if self._sqlite is not None:
                self._sqlite.execute("DELETE FROM events")
                self._sqlite.execute("DELETE FROM budgets")
                self._sqlite.commit()


# Module-level singleton — wired into FastAPI dependency in main.py.
_STORE: Store | None = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def reset_store_for_tests() -> Store:
    """Force a fresh in-memory store (sqlite at :memory:) — used by tests."""
    global _STORE
    _STORE = Store(database_url="sqlite://:memory:")
    return _STORE

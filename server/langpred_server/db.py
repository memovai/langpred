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
    project_id: str
    type: str
    trace_id: str | None
    observation_id: str | None
    timestamp: str
    body: dict[str, Any]

    def to_row(self) -> tuple[str, str, str, str | None, str | None, str, str]:
        return (
            self.id,
            self.project_id,
            self.type,
            self.trace_id,
            self.observation_id,
            self.timestamp,
            json.dumps(_to_json_safe(self.body)),
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "StoredEvent":
        # Older SQLite files did not have project_id. Treat those rows as the
        # default local project so existing installs migrate cleanly.
        if len(row) == 6:
            row = (row[0], "default", row[1], row[2], row[3], row[4], row[5])
        return cls(
            id=row[0],
            project_id=row[1],
            type=row[2],
            trace_id=row[3],
            observation_id=row[4],
            timestamp=row[5],
            body=json.loads(row[6]) if row[6] else {},
        )


@dataclass
class BudgetRecord:
    trace_id: str
    project_id: str
    cap_usd: float
    on_exceed: str
    quantile: str = "p50"
    breached: bool = False
    breach_reason: str | None = None
    last_spent_usd: float = 0.0
    last_predicted_remaining_p50: float = 0.0
    last_predicted_remaining_p90: float = 0.0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


@dataclass
class AlertRuleRecord:
    id: str
    trace_id: str
    project_id: str
    condition: str
    webhook_url: str
    min_interval_seconds: float = 30.0
    last_fired_at: str | None = None
    fire_count: int = 0
    last_value: float | None = None
    created_at: str = field(default_factory=_now_iso)


class Store:
    """Thread-safe in-memory + (optional) sqlite event log."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or SETTINGS.database_url
        self._lock = threading.RLock()
        self._events: list[StoredEvent] = []
        self._by_trace: dict[tuple[str, str], list[int]] = {}  # (project_id, trace_id)
        self._budgets: dict[tuple[str, str], BudgetRecord] = {}
        self._alerts: dict[str, AlertRuleRecord] = {}  # id -> rule
        self._alerts_by_trace: dict[tuple[str, str], list[str]] = {}
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
                project_id TEXT NOT NULL DEFAULT 'default',
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
                trace_id TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT 'default',
                cap_usd REAL NOT NULL,
                on_exceed TEXT NOT NULL,
                quantile TEXT NOT NULL DEFAULT 'p50',
                breached INTEGER NOT NULL,
                breach_reason TEXT,
                last_spent_usd REAL NOT NULL,
                last_pred_remaining_p50 REAL NOT NULL,
                last_pred_remaining_p90 REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (project_id, trace_id)
            )
            """
        )
        self._sqlite.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_project_trace ON events(project_id, trace_id)"
        )
        self._ensure_sqlite_column("events", "project_id", "TEXT NOT NULL DEFAULT 'default'")
        self._ensure_sqlite_column("budgets", "project_id", "TEXT NOT NULL DEFAULT 'default'")
        self._ensure_sqlite_column("budgets", "quantile", "TEXT NOT NULL DEFAULT 'p50'")
        self._sqlite.commit()

    def _ensure_sqlite_column(self, table: str, column: str, spec: str) -> None:
        assert self._sqlite is not None
        cols = {row[1] for row in self._sqlite.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self._sqlite.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")

    def _reload_from_sqlite(self) -> None:
        assert self._sqlite is not None
        cur = self._sqlite.execute(
            "SELECT id, project_id, type, trace_id, observation_id, ts, body FROM events ORDER BY ts ASC"
        )
        for row in cur.fetchall():
            ev = StoredEvent.from_row(row)
            self._events.append(ev)
            if ev.trace_id:
                self._by_trace.setdefault((ev.project_id, ev.trace_id), []).append(len(self._events) - 1)
        cur = self._sqlite.execute(
            """
            SELECT trace_id, project_id, cap_usd, on_exceed, quantile, breached,
                   breach_reason, last_spent_usd, last_pred_remaining_p50,
                   last_pred_remaining_p90, created_at, updated_at
            FROM budgets
            """
        )
        for row in cur.fetchall():
            (tid, project_id, cap, ox, quantile, br, br_r, sp, p50, p90, ca, ua) = row
            self._budgets[(project_id, tid)] = BudgetRecord(
                trace_id=tid,
                project_id=project_id,
                cap_usd=cap,
                on_exceed=ox,
                quantile=quantile,
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
        project_id: str,
        ev_type: str,
        trace_id: str | None,
        observation_id: str | None,
        ts: str,
        body: dict[str, Any],
    ) -> StoredEvent:
        ev = StoredEvent(
            id=ev_id,
            project_id=project_id,
            type=ev_type,
            trace_id=trace_id,
            observation_id=observation_id,
            timestamp=ts,
            body=body,
        )
        with self._lock:
            self._events.append(ev)
            if trace_id:
                self._by_trace.setdefault((project_id, trace_id), []).append(len(self._events) - 1)
            if self._sqlite is not None:
                try:
                    self._sqlite.execute(
                        "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?)",
                        ev.to_row(),
                    )
                    self._sqlite.commit()
                except sqlite3.Error:
                    # tolerate duplicates / closed connections
                    pass
        return ev

    def events_for_trace(self, trace_id: str, project_id: str | None = None) -> list[StoredEvent]:
        with self._lock:
            if project_id is None:
                idx = [
                    i
                    for (pid, tid), indices in self._by_trace.items()
                    if tid == trace_id
                    for i in indices
                ]
            else:
                idx = self._by_trace.get((project_id, trace_id), [])
            return [self._events[i] for i in idx]

    def all_trace_ids(self, project_id: str | None = None) -> list[str]:
        with self._lock:
            if project_id is None:
                return [tid for _pid, tid in self._by_trace.keys()]
            return [tid for pid, tid in self._by_trace.keys() if pid == project_id]

    def trace_count(self) -> int:
        with self._lock:
            return len(self._by_trace)

    # ----------------------------------------------------------------- budgets

    def set_budget(self, b: BudgetRecord) -> None:
        with self._lock:
            self._budgets[(b.project_id, b.trace_id)] = b
            self._persist_budget(b)

    def get_budget(self, trace_id: str, project_id: str | None = None) -> BudgetRecord | None:
        with self._lock:
            if project_id is not None:
                return self._budgets.get((project_id, trace_id))
            for (_pid, tid), rec in self._budgets.items():
                if tid == trace_id:
                    return rec
            return None

    def update_budget(self, b: BudgetRecord) -> None:
        with self._lock:
            b.updated_at = _now_iso()
            self._budgets[(b.project_id, b.trace_id)] = b
            self._persist_budget(b)

    # ----------------------------------------------------------------- alerts

    def add_alert(self, rule: AlertRuleRecord) -> None:
        with self._lock:
            self._alerts[rule.id] = rule
            self._alerts_by_trace.setdefault((rule.project_id, rule.trace_id), []).append(rule.id)

    def alerts_for(self, trace_id: str, project_id: str | None = None) -> list[AlertRuleRecord]:
        with self._lock:
            if project_id is None:
                ids = [
                    rule_id
                    for (_pid, tid), rule_ids in self._alerts_by_trace.items()
                    if tid == trace_id
                    for rule_id in rule_ids
                ]
            else:
                ids = self._alerts_by_trace.get((project_id, trace_id), [])
            return [self._alerts[i] for i in ids if i in self._alerts]

    def update_alert(self, rule: AlertRuleRecord) -> None:
        with self._lock:
            self._alerts[rule.id] = rule

    # ----------------------------------------------------------------- helpers

    def _persist_budget(self, b: BudgetRecord) -> None:
        if self._sqlite is None:
            return
        try:
            self._sqlite.execute(
                """
                INSERT OR REPLACE INTO budgets (
                    trace_id, project_id, cap_usd, on_exceed, quantile, breached,
                    breach_reason, last_spent_usd, last_pred_remaining_p50,
                    last_pred_remaining_p90, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    b.trace_id,
                    b.project_id,
                    b.cap_usd,
                    b.on_exceed,
                    b.quantile,
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
            self._alerts.clear()
            self._alerts_by_trace.clear()
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

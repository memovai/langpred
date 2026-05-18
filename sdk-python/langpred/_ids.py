from __future__ import annotations

import os
import uuid


def new_id() -> str:
    """W3C-trace-id-compatible 32-char hex id (no dashes)."""
    return uuid.uuid4().hex


def new_event_id() -> str:
    return str(uuid.uuid4())


def env(name: str, *fallbacks: str, default: str | None = None) -> str | None:
    """Get LANGPRED_X falling back to LANGFUSE_X — the drop-in story."""
    for k in (name, *fallbacks):
        v = os.environ.get(k)
        if v:
            return v
    return default

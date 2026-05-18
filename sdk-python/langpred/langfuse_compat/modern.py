"""
Modern Langfuse (v3+/v4) shape: ``get_client()``, ``start_observation()``,
``start_as_current_observation()``, ``@observe()``, ``propagate_attributes``.

We provide just enough surface area for typical agent instrumentation to
flow into Langpred unchanged. Where Langfuse v4 uses OpenTelemetry, we
approximate via contextvars — good enough for sync + async agent code
without dragging in an OTel SDK.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
from typing import Any, Callable, Iterator

from ..client import Langpred
from ..trace import Generation, Span, Trace


_CLIENT: Langpred | None = None
_CURRENT_TRACE: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "_langpred_current_trace", default=None
)
_CURRENT_OBSERVATION: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_langpred_current_observation", default=None
)
_PROPAGATED: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "_langpred_propagated", default={}
)


def get_client(**kwargs: Any) -> Langpred:
    """Return the singleton client (created lazily). Kwargs are forwarded to
    :class:`langpred.Langpred` on first call."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Langpred(**kwargs)
    return _CLIENT


def _ensure_trace() -> Trace:
    trace = _CURRENT_TRACE.get()
    if trace is None:
        client = get_client()
        attrs = _PROPAGATED.get() or {}
        trace = client.trace(
            name=attrs.get("trace_name"),
            user_id=attrs.get("user_id"),
            session_id=attrs.get("session_id"),
            metadata=attrs.get("metadata"),
            version=attrs.get("version"),
        )
        _CURRENT_TRACE.set(trace)
    return trace


@contextlib.contextmanager
def propagate_attributes(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    version: str | None = None,
    trace_name: str | None = None,
    as_baggage: bool = False,
) -> Iterator[None]:
    """Same shape as ``langfuse.propagate_attributes`` — sets attributes that
    new observations / traces will pick up.
    """
    cur = dict(_PROPAGATED.get() or {})
    if user_id is not None:
        cur["user_id"] = user_id
    if session_id is not None:
        cur["session_id"] = session_id
    if metadata is not None:
        cur["metadata"] = metadata
    if version is not None:
        cur["version"] = version
    if trace_name is not None:
        cur["trace_name"] = trace_name
    token = _PROPAGATED.set(cur)
    try:
        yield
    finally:
        _PROPAGATED.reset(token)


def start_observation(
    *,
    as_type: str = "span",
    name: str | None = None,
    **kwargs: Any,
) -> Span | Generation:
    """Mimics ``langfuse.start_observation``. Returns a Span or Generation
    that the caller must ``.end()``."""
    trace = _ensure_trace()
    parent = _CURRENT_OBSERVATION.get()
    parent_id = getattr(parent, "id", None) if parent else None
    if as_type == "generation":
        return Generation(trace.transport, trace.id, parent_id=parent_id, name=name, **kwargs)
    return Span(trace.transport, trace.id, parent_id=parent_id, name=name, **kwargs)


@contextlib.contextmanager
def start_as_current_observation(
    *,
    as_type: str = "span",
    name: str | None = None,
    **kwargs: Any,
) -> Iterator[Span | Generation]:
    obs = start_observation(as_type=as_type, name=name, **kwargs)
    token = _CURRENT_OBSERVATION.set(obs)
    try:
        yield obs
    finally:
        try:
            obs.end()
        finally:
            _CURRENT_OBSERVATION.reset(token)


def update_current_span(**fields: Any) -> None:
    cur = _CURRENT_OBSERVATION.get()
    if cur is not None and isinstance(cur, Span):
        cur.update(**fields)


def update_current_generation(**fields: Any) -> None:
    cur = _CURRENT_OBSERVATION.get()
    if cur is not None and isinstance(cur, Generation):
        cur.update(**fields)


def observe(
    *,
    as_type: str = "span",
    name: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """``@observe`` decorator — wraps sync or async functions and emits
    spans / generations to Langpred."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        obs_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                payload = (args, kwargs) if capture_input else None
                with start_as_current_observation(
                    as_type=as_type, name=obs_name, input=payload
                ) as obs:
                    result = await fn(*args, **kwargs)
                    if capture_output:
                        obs.update(output=result)
                    return result

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = (args, kwargs) if capture_input else None
            with start_as_current_observation(
                as_type=as_type, name=obs_name, input=payload
            ) as obs:
                result = fn(*args, **kwargs)
                if capture_output:
                    obs.update(output=result)
                return result

        return wrapper

    return decorator


__all__ = [
    "get_client",
    "observe",
    "propagate_attributes",
    "start_observation",
    "start_as_current_observation",
    "update_current_span",
    "update_current_generation",
]

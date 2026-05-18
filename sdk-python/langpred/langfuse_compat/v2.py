"""
``langfuse.Langfuse`` v2 shape: ``langfuse.trace(...).span(...).generation(...)``.

We subclass :class:`langpred.Langpred` and re-export it as ``Langfuse`` so
existing code that does ``from langfuse import Langfuse`` only has to change
the import.
"""
from __future__ import annotations

from ..client import Langpred


class Langfuse(Langpred):
    """Drop-in for ``langfuse.Langfuse``.

    All v2 methods (``trace``, ``flush``, ``shutdown``) are inherited verbatim
    from :class:`langpred.Langpred`. Prediction methods are available on the
    returned :class:`langpred.Trace` object.

    Example:
        >>> from langpred.langfuse_compat import Langfuse
        >>> lf = Langfuse(host="http://localhost:7187")
        >>> trace = lf.trace(name="my_agent")
        >>> gen = trace.generation(model="claude-sonnet-4-6", input="...", output="...")
        >>> gen.end()
        >>> lf.flush()
        >>> cost = trace.predict_cost()
    """

    pass

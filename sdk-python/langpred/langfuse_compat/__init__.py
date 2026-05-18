"""
``from langpred.langfuse_compat import Langfuse`` — a drop-in for
``from langfuse import Langfuse``.

Both v2 (``langfuse.trace(...).generation(...)``) and v3+ (``get_client()``,
``start_observation()``, ``@observe``) entry points are exported here. They
all funnel through the same Langpred transport so prediction + budget calls
work whether the caller uses the legacy or modern Langfuse shape.
"""

from .v2 import Langfuse
from .modern import (
    get_client,
    observe,
    propagate_attributes,
    update_current_span,
    update_current_generation,
)

__all__ = [
    "Langfuse",
    "get_client",
    "observe",
    "propagate_attributes",
    "update_current_span",
    "update_current_generation",
]

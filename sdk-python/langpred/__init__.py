"""
Langpred Python SDK — drop-in superset of the Langfuse client surface.

Two ways to use:

1. **Native**

   .. code-block:: python

      from langpred import Langpred
      lp = Langpred()
      trace = lp.trace(name="my_agent")
      gen = trace.generation(model="claude-sonnet-4-6", input=..., output=...)
      gen.end()
      lp.flush()

      eta = trace.predict_eta()
      cost = trace.predict_cost()
      with trace.set_budget(usd=0.50, on_exceed="kill") as guard:
          agent.run()

2. **Langfuse-compat (drop-in for existing apps)**

   .. code-block:: python

      from langpred.langfuse_compat import Langfuse  # was: from langfuse import Langfuse
      langfuse = Langfuse()
      trace = langfuse.trace(name="my_agent")
      ...
"""

from .client import Langpred
from .trace import Generation, Score, Span, Trace
from .budget import BudgetExceeded, BudgetGuard
from .predict import EtaPrediction, CostPrediction, OffRailsPrediction

__all__ = [
    "Langpred",
    "Trace",
    "Span",
    "Generation",
    "Score",
    "BudgetExceeded",
    "BudgetGuard",
    "EtaPrediction",
    "CostPrediction",
    "OffRailsPrediction",
]

__version__ = "0.1.0"

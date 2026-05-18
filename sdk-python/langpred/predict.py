"""Client-side prediction wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _PredictionBase:
    trace_id: str
    p50: float
    p90: float
    p99: float
    confidence: float
    tier: str
    n_samples: int
    explanation: str = ""

    @classmethod
    def from_response(cls, body: dict[str, Any]) -> "_PredictionBase":
        return cls(
            trace_id=body["trace_id"],
            p50=float(body["p50"]),
            p90=float(body["p90"]),
            p99=float(body["p99"]),
            confidence=float(body["confidence"]),
            tier=body.get("tier", "heuristic"),
            n_samples=int(body.get("n_samples", 0)),
            explanation=body.get("explanation", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}("
            f"p50={self.p50:.4f}, p90={self.p90:.4f}, p99={self.p99:.4f}, "
            f"confidence={self.confidence:.2f}, tier={self.tier}, n={self.n_samples})"
        )


@dataclass
class EtaPrediction(_PredictionBase):
    @property
    def seconds_p50(self) -> float:
        return self.p50

    @property
    def seconds_p90(self) -> float:
        return self.p90

    @property
    def seconds_p99(self) -> float:
        return self.p99


@dataclass
class CostPrediction(_PredictionBase):
    @property
    def usd_p50(self) -> float:
        return self.p50

    @property
    def usd_p90(self) -> float:
        return self.p90

    @property
    def usd_p99(self) -> float:
        return self.p99


@dataclass
class OffRailsPrediction(_PredictionBase):
    @property
    def score(self) -> float:
        return self.p50

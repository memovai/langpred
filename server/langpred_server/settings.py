from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("LANGPRED_HOST", "0.0.0.0")
    port: int = int(os.environ.get("LANGPRED_PORT", "7187"))
    database_url: str = os.environ.get(
        "LANGPRED_DATABASE_URL", "sqlite:///./langpred.db"
    )
    train_interval_seconds: int = int(
        os.environ.get("LANGPRED_TRAIN_INTERVAL_SECONDS", "300")
    )
    train_on_event: bool = os.environ.get("LANGPRED_TRAIN_ON_EVENT", "0") == "1"
    mirror_to: str | None = os.environ.get("LANGPRED_MIRROR_TO") or None
    log_level: str = os.environ.get("LANGPRED_LOG_LEVEL", "info")
    # Knobs for the predictor.
    cold_start_threshold: int = int(
        os.environ.get("LANGPRED_COLD_START_THRESHOLD", "50")
    )
    gbm_promote_threshold: int = int(
        os.environ.get("LANGPRED_GBM_THRESHOLD", "1000")
    )


SETTINGS = Settings()

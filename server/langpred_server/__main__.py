"""`python -m langpred_server` and `langpred-server` entry-point."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("LANGPRED_HOST", "0.0.0.0")
    port = int(os.environ.get("LANGPRED_PORT", "7187"))
    log_level = os.environ.get("LANGPRED_LOG_LEVEL", "info")
    uvicorn.run(
        "langpred_server.main:app",
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()

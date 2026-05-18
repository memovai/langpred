"""
Token -> USD pricing table.

Numbers are the published list prices as of early 2026 (cents per million tokens).
Override via :data:`PRICE_TABLE` or by passing explicit cost in the ingestion
payload. We deliberately fall back to a "median frontier model" price for
unknown models so cost predictions never silently zero out.
"""
from __future__ import annotations

# USD per 1K tokens. {model_substr: (prompt_usd_per_1k, completion_usd_per_1k)}
# Match by substring so "claude-sonnet-4-6-20260101" still hits "claude-sonnet-4-6".
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (0.015, 0.075),
    "claude-opus-4-6": (0.015, 0.075),
    "claude-opus": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-sonnet": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),
    "claude-haiku": (0.0008, 0.004),
    # OpenAI
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "gpt-4.1": (0.005, 0.015),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5": (0.0005, 0.0015),
    "o3-mini": (0.0011, 0.0044),
    "o3": (0.015, 0.06),
    # Google
    "gemini-2.5-pro": (0.00125, 0.005),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini": (0.0005, 0.0015),
    # Open weights — rough hosted prices
    "llama-3": (0.00059, 0.00079),
    "llama": (0.0008, 0.0008),
    "mistral-large": (0.002, 0.006),
    "mistral": (0.0007, 0.0007),
    "deepseek": (0.00027, 0.0011),
    "qwen": (0.0006, 0.0006),
}

_FALLBACK = (0.003, 0.015)  # treat unknown models as a Sonnet-class price


def price_step(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD for a single LLM call given token counts. Robust to None."""
    if not (prompt_tokens or completion_tokens):
        return 0.0
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    if model:
        m = model.lower()
        for prefix, (pp, cp) in PRICE_TABLE.items():
            if prefix in m:
                return (prompt / 1000.0) * pp + (completion / 1000.0) * cp
    pp, cp = _FALLBACK
    return (prompt / 1000.0) * pp + (completion / 1000.0) * cp

"""Lightweight project identity helpers.

Langfuse clients send HTTP Basic auth using the public/secret key pair. We do
not validate keys in the local server, but the public key is still the natural
project boundary for prediction cohorts and budgets.
"""
from __future__ import annotations

import base64


DEFAULT_PROJECT_ID = "default"


def project_id_from_authorization(authorization: str | None) -> str:
    """Return a stable project id derived from the Basic auth public key.

    Missing or malformed auth falls back to ``default`` so local examples and
    existing tests keep working without credentials.
    """
    if not authorization:
        return DEFAULT_PROJECT_ID
    scheme, _, payload = authorization.partition(" ")
    if scheme.lower() != "basic" or not payload:
        return DEFAULT_PROJECT_ID
    try:
        decoded = base64.b64decode(payload).decode("utf-8")
    except Exception:
        return DEFAULT_PROJECT_ID
    public_key, _, _secret = decoded.partition(":")
    public_key = public_key.strip()
    return public_key or DEFAULT_PROJECT_ID

"""Security helpers for the local Trace review service."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping

DEFAULT_TOKEN_ENV = "TRACE_REVIEW_APP_TOKEN"


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is explicitly loopback-only."""

    normalized = str(host).strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    if "%" in normalized:
        normalized = normalized.split("%", 1)[0]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_review_bind(
    host: str,
    *,
    token_env: str = DEFAULT_TOKEN_ENV,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve auth for a bind, rejecting exposed unauthenticated servers."""

    source = os.environ if environ is None else environ
    token = str(source.get(token_env, "")).strip()
    if is_loopback_host(host):
        return token or None
    if not token:
        raise ValueError(
            f"non-loopback review binds require an authentication token in {token_env}"
        )
    return token


__all__ = ["DEFAULT_TOKEN_ENV", "is_loopback_host", "validate_review_bind"]

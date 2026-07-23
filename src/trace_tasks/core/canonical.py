"""Canonical JSON serialization utilities for deterministic identity hashes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any

from . import error_codes

try:
    import rfc8785
except Exception:  # pragma: no cover - dependency errors handled at runtime
    rfc8785 = None


class CanonicalizationError(ValueError):
    """Raised when canonical serialization requirements are violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _normalize(obj: Any) -> Any:
    """Convert supported Python objects to canonical JSON-compatible values."""
    if is_dataclass(obj):
        return _normalize(asdict(obj))
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise CanonicalizationError(
                error_codes.SCHEMA_NON_FINITE_NUMBER,
                "non-finite numbers are not allowed in canonical JSON",
            )
        return obj
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    error_codes.SCHEMA_NON_STRING_KEY,
                    "canonical JSON requires string object keys",
                )
            out[key] = _normalize(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalize(item) for item in obj]
    raise CanonicalizationError(
        error_codes.SCHEMA_UNSUPPORTED_TYPE,
        f"unsupported canonicalization type: {type(obj).__name__}",
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object using deterministic RFC8785 JSON canonicalization."""
    normalized = _normalize(obj)
    try:
        if rfc8785 is not None:
            data = rfc8785.dumps(normalized)
            if isinstance(data, str):
                return data.encode("utf-8")
            return bytes(data)
        # Fallback retains deterministic key ordering and finite-value checks,
        # but callers should prefer environments where rfc8785 is available.
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except CanonicalizationError:
        raise
    except Exception as exc:
        raise CanonicalizationError(
            error_codes.SCHEMA_CANONICALIZATION_FAILED,
            f"canonicalization failed: {exc}",
        ) from exc

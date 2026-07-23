"""Shared range-normalization helpers for visual configuration parsing."""

from __future__ import annotations

from typing import Any, Tuple


def _to_int(value: Any, fallback: int) -> int:
    """Parse an integer with fallback for invalid values."""
    try:
        return int(value)
    except Exception:
        return int(fallback)


def normalize_int_range(value: Any, *, fallback_min: int, fallback_max: int) -> Tuple[int, int]:
    """Normalize range-like input into an ordered integer `(min, max)` tuple."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lo = _to_int(value[0], int(fallback_min))
        hi = _to_int(value[1], int(fallback_max))
    elif value is None:
        lo = int(fallback_min)
        hi = int(fallback_max)
    else:
        scalar = _to_int(value, int(fallback_min))
        lo = int(scalar)
        hi = int(scalar)
    return (int(min(lo, hi)), int(max(lo, hi)))


def normalize_non_negative_int_range(value: Any, *, fallback_min: int, fallback_max: int) -> Tuple[int, int]:
    """Normalize range-like input into ordered non-negative integer bounds."""
    lo, hi = normalize_int_range(
        value,
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    return (max(0, int(lo)), max(0, int(hi)))

"""Shared JSON-ready metadata serialization helpers for geometry tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def geometry_json_ready(value: Any, *, round_floats: bool = True, ndigits: int = 3) -> Any:
    """Return a JSON-native copy of geometry metadata.

    Geometry measurement tasks historically used two policies for float values:
    most trace-facing render metadata rounded floats to three decimals, while a
    few symbolic scene maps preserved full precision. Keep that policy explicit
    at call sites instead of carrying task-local serializer copies.
    """

    if isinstance(value, Mapping):
        return {
            str(key): geometry_json_ready(inner, round_floats=round_floats, ndigits=ndigits)
            for key, inner in value.items()
        }
    if isinstance(value, tuple):
        return [geometry_json_ready(inner, round_floats=round_floats, ndigits=ndigits) for inner in value]
    if isinstance(value, list):
        return [geometry_json_ready(inner, round_floats=round_floats, ndigits=ndigits) for inner in value]
    if bool(round_floats) and isinstance(value, float):
        return round(float(value), int(ndigits))
    return value


__all__ = ["geometry_json_ready"]

"""Annotation helpers for thermal-mixing diagrams."""

from __future__ import annotations

from typing import List, Sequence

def bbox(values: Sequence[float]) -> List[float]:
    """Return a stable rounded pixel bbox."""

    return [round(float(value), 3) for value in values]


__all__ = ["bbox"]

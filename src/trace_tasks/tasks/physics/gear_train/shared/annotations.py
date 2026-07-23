"""Annotation helpers for gear-train witnesses."""

from __future__ import annotations

from typing import Sequence


def bbox(values: Sequence[float]) -> list[float]:
    """Return one JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def clamp_bbox(values: Sequence[float], *, width: int, height: int) -> list[float]:
    """Clamp a bbox to rendered image bounds."""

    x0, y0, x1, y1 = [float(value) for value in values[:4]]
    return bbox(
        (
            max(0.0, min(float(width - 1), min(x0, x1))),
            max(0.0, min(float(height - 1), min(y0, y1))),
            max(1.0, min(float(width), max(x0, x1))),
            max(1.0, min(float(height), max(y0, y1))),
        )
    )


__all__ = ["bbox", "clamp_bbox"]

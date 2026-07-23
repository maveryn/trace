"""Small 2D vector helpers for geometry renderers."""

from __future__ import annotations

import math
from typing import Sequence, Tuple


Point = Tuple[float, float]


def point_to_list(point: Sequence[float], *, ndigits: int = 3) -> list[float]:
    """Return one pixel point as a rounded JSON-ready list."""

    return [round(float(point[0]), int(ndigits)), round(float(point[1]), int(ndigits))]


def add(a: Sequence[float], b: Sequence[float]) -> Point:
    """Return ``a + b``."""

    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]))


def add_scaled(a: Sequence[float], b: Sequence[float], scale: float = 1.0) -> Point:
    """Return ``a + scale * b``."""

    return (
        float(a[0]) + (float(b[0]) * float(scale)),
        float(a[1]) + (float(b[1]) * float(scale)),
    )


def sub(a: Sequence[float], b: Sequence[float]) -> Point:
    """Return ``a - b``."""

    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def mul(a: Sequence[float], factor: float) -> Point:
    """Return one vector multiplied by a scalar."""

    return (float(a[0]) * float(factor), float(a[1]) * float(factor))


def mid(a: Sequence[float], b: Sequence[float]) -> Point:
    """Return the midpoint of two points."""

    return ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product of two vectors."""

    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Euclidean distance between two points."""

    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def unit(a: Sequence[float]) -> Point:
    """Return a unit vector, falling back to the positive x-axis for near-zero input."""

    length = math.hypot(float(a[0]), float(a[1]))
    if length <= 1e-9:
        return (1.0, 0.0)
    return (float(a[0]) / length, float(a[1]) / length)


def perp(a: Sequence[float]) -> Point:
    """Return the left-hand perpendicular vector."""

    return (-float(a[1]), float(a[0]))


__all__ = [
    "Point",
    "add",
    "add_scaled",
    "distance",
    "dot",
    "mid",
    "mul",
    "perp",
    "point_to_list",
    "sub",
    "unit",
]

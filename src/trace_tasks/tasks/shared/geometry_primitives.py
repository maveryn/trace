"""Cross-domain geometric primitives for task generators.

These helpers keep a canonical `Point` type alias and basic low-level geometry
operations in one place so task modules can reuse them instead of duplicating
math/type definitions.
"""

from __future__ import annotations

from typing import Tuple


Point = Tuple[float, float]


def distance_sq(point_a: Point, point_b: Point) -> float:
    """Return squared Euclidean distance between two points."""
    delta_x = float(point_a[0]) - float(point_b[0])
    delta_y = float(point_a[1]) - float(point_b[1])
    return (delta_x * delta_x) + (delta_y * delta_y)


def point_inside_square_canvas(point: Point, *, canvas_size: int, padding: float = 2.0) -> bool:
    """Return whether one point lies inside a square canvas with optional padding."""
    x_value, y_value = point
    return (
        float(padding) <= float(x_value) <= float(canvas_size) - float(padding)
        and float(padding) <= float(y_value) <= float(canvas_size) - float(padding)
    )


__all__ = [
    "Point",
    "distance_sq",
    "point_inside_square_canvas",
]

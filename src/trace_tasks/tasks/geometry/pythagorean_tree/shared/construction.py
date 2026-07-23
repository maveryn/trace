"""Construction primitives for attached-square tree diagrams."""

from __future__ import annotations

import math
from typing import Sequence

from trace_tasks.tasks.geometry.shared.vector2d import add, dot, mul, sub, unit

from .state import Point, Polygon


def square_on_segment(start: Point, end: Point, *, away_from: Point) -> Polygon:
    """Return the square attached to one directed segment, outside the triangle."""

    vector = sub(end, start)
    length = math.hypot(float(vector[0]), float(vector[1]))
    normal = unit((-float(vector[1]), float(vector[0])))
    midpoint = ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
    if dot(sub(away_from, midpoint), normal) > 0.0:
        normal = (-normal[0], -normal[1])
    offset = mul(normal, length)
    return (start, end, add(end, offset), add(start, offset))


def rotate_point(point: Point, angle_radians: float) -> Point:
    """Rotate one local point around the origin."""

    x, y = float(point[0]), float(point[1])
    cos_a = math.cos(float(angle_radians))
    sin_a = math.sin(float(angle_radians))
    return ((x * cos_a) - (y * sin_a), (x * sin_a) + (y * cos_a))


def transform_points(
    points: Sequence[Point],
    *,
    angle_radians: float,
    scale: float,
    offset: Point,
) -> tuple[Point, ...]:
    """Rotate, scale, and translate local points to canvas coordinates."""

    return tuple(add(mul(rotate_point(point, angle_radians), float(scale)), offset) for point in points)


def polygon_center(points: Sequence[Point]) -> Point:
    """Return the centroid of one polygon vertex list."""

    return (
        sum(float(point[0]) for point in points) / float(len(points)),
        sum(float(point[1]) for point in points) / float(len(points)),
    )


__all__ = ["polygon_center", "rotate_point", "square_on_segment", "transform_points"]

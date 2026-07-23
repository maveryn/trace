"""Geometry construction primitives for graph-paper tasks."""

from __future__ import annotations

from math import cos, pi, sin
from random import Random
from typing import Sequence

from .state import Point


def rectangle_points(
    center: Point, width: float, height: float
) -> tuple[Point, Point, Point, Point]:
    """Return graph-unit rectangle vertices around a center."""

    cx, cy = float(center[0]), float(center[1])
    w, h = float(width) / 2.0, float(height) / 2.0
    return ((cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h))


def right_triangle_points(
    center: Point, base: float, height: float
) -> tuple[Point, Point, Point]:
    """Return graph-unit right-triangle vertices around a center."""

    cx, cy = float(center[0]), float(center[1])
    return (
        (cx - base / 2.0, cy - height / 2.0),
        (cx + base / 2.0, cy - height / 2.0),
        (cx - base / 2.0, cy + height / 2.0),
    )


def rotate_points(
    points: Sequence[Point], center: Point, angle_radians: float
) -> tuple[Point, ...]:
    """Rotate graph-unit points around a graph-unit center."""

    cx, cy = float(center[0]), float(center[1])
    cosine = cos(float(angle_radians))
    sine = sin(float(angle_radians))
    rotated: list[Point] = []
    for point in points:
        dx = float(point[0]) - cx
        dy = float(point[1]) - cy
        rotated.append((cx + (dx * cosine - dy * sine), cy + (dx * sine + dy * cosine)))
    return tuple(rotated)


def polygon_area(points: Sequence[Point]) -> float:
    """Shoelace area for graph-unit polygon points."""

    total = 0.0
    pts = list(points)
    for index, point in enumerate(pts):
        nxt = pts[(index + 1) % len(pts)]
        total += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return abs(total) / 2.0


def polygon_perimeter(points: Sequence[Point]) -> float:
    """Perimeter for graph-unit polygon points."""

    total = 0.0
    pts = list(points)
    for index, point in enumerate(pts):
        nxt = pts[(index + 1) % len(pts)]
        total += (
            (float(point[0]) - float(nxt[0])) ** 2
            + (float(point[1]) - float(nxt[1])) ** 2
        ) ** 0.5
    return total


def pi_expression(coefficient: int) -> str:
    """Return a compact kπ expression."""

    value = int(coefficient)
    if value == 1:
        return "π"
    return f"{value}π"


def regular_polygon(
    center: Point, sides: int, radius: float, *, phase: float = 0.0
) -> tuple[Point, ...]:
    """Return graph-unit vertices of a regular polygon."""

    from math import cos, sin

    cx, cy = float(center[0]), float(center[1])
    return tuple(
        (
            cx + float(radius) * cos(float(phase) + 2.0 * pi * index / int(sides)),
            cy + float(radius) * sin(float(phase) + 2.0 * pi * index / int(sides)),
        )
        for index in range(int(sides))
    )


def irregular_convex_polygon(
    center: Point, sides: int, radius: float, rng: Random
) -> tuple[Point, ...]:
    """Return a convex non-regular polygon with a stable side count."""

    resolved_sides = max(5, min(7, int(sides)))
    cx, cy = float(center[0]), float(center[1])
    phase = float(rng.uniform(0.0, 2.0 * pi))
    rotation = float(rng.uniform(0.0, 2.0 * pi))
    scale_x = float(rng.uniform(0.78, 1.16))
    scale_y = float(rng.uniform(0.78, 1.16))
    if abs(scale_x - scale_y) < 0.12:
        scale_y = max(0.72, min(1.22, scale_y + (0.16 if scale_y < 1.0 else -0.16)))
    shear = float(rng.uniform(-0.16, 0.16))
    cosine = cos(rotation)
    sine = sin(rotation)
    points: list[Point] = []
    for index in range(resolved_sides):
        angle = phase + (2.0 * pi * index / resolved_sides)
        local_x = float(radius) * cos(angle)
        local_y = float(radius) * sin(angle)
        aff_x = (scale_x * local_x) + (shear * local_y)
        aff_y = scale_y * local_y
        points.append(
            (
                cx + (aff_x * cosine - aff_y * sine),
                cy + (aff_x * sine + aff_y * cosine),
            )
        )
    return tuple(points)


def concave_polygon(
    center: Point, sides: int, radius: float, rng: Random
) -> tuple[Point, ...]:
    """Return a visually clear inward-notch polygon with a stable side count."""

    resolved_sides = max(5, min(7, int(sides)))
    cx, cy = float(center[0]), float(center[1])
    phase = float(rng.uniform(0.0, 2.0 * pi))
    rotation = float(rng.uniform(0.0, 2.0 * pi))
    scale_x = float(rng.uniform(0.86, 1.12))
    scale_y = float(rng.uniform(0.86, 1.12))
    notch_index = int(rng.randrange(0, resolved_sides))
    notch_factor = float(rng.uniform(0.06, 0.16))
    cosine = cos(rotation)
    sine = sin(rotation)
    points: list[Point] = []
    for index in range(resolved_sides):
        angle = phase + (2.0 * pi * index / resolved_sides)
        radial = float(radius) * (notch_factor if index == notch_index else 1.0)
        local_x = radial * cos(angle)
        local_y = radial * sin(angle)
        aff_x = scale_x * local_x
        aff_y = scale_y * local_y
        points.append(
            (
                cx + (aff_x * cosine - aff_y * sine),
                cy + (aff_x * sine + aff_y * cosine),
            )
        )
    return tuple(points)

"""Low-level geometry helpers for circle-theorem scene builders."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

from .state import Point

def _sample_external_point_side(rng) -> str:
    return "right" if int(rng.randrange(2)) else "left"

def _mirror_point_x(point: Point) -> Point:
    return (-float(point[0]), float(point[1]))

def _apply_external_point_side(
    point_model: Mapping[str, Point],
    circle_center: Point,
    *,
    side: str,
) -> Tuple[Dict[str, Point], Point]:
    if str(side) == "left":
        return dict(point_model), circle_center
    if str(side) == "right":
        return (
            {str(label): _mirror_point_x(point) for label, point in point_model.items()},
            _mirror_point_x(circle_center),
        )
    raise ValueError(f"unsupported external point side: {side!r}")

def _circle_point(radius: float, angle_degrees: float) -> Point:
    angle = math.radians(float(angle_degrees))
    return (float(radius * math.cos(angle)), float(radius * math.sin(angle)))

def _rotated_tangent_unit(angle_degrees: float) -> Point:
    angle = math.radians(float(angle_degrees))
    return (float(-math.sin(angle)), float(math.cos(angle)))

def _add_points(a: Point, b: Point, *, scale: float = 1.0) -> Point:
    return (float(a[0] + (float(scale) * b[0])), float(a[1] + (float(scale) * b[1])))

def _split_cyclic_arc_sum(rng, arc_sum: int) -> Tuple[int, int]:
    candidates = [
        first
        for first in range(35, 156, 5)
        if 35 <= int(arc_sum - first) <= 155
    ]
    if not candidates:
        raise ValueError(f"cannot split cyclic arc sum: {arc_sum}")
    first = int(rng.choice(candidates))
    return first, int(arc_sum - first)

def _extend_ray(start: Point, through: Point, *, distance: float) -> Point:
    sx, sy = float(start[0]), float(start[1])
    tx, ty = float(through[0]), float(through[1])
    dx, dy = tx - sx, ty - sy
    norm = max(1e-6, math.hypot(dx, dy))
    return (
        float(tx + (float(distance) * dx / norm)),
        float(ty + (float(distance) * dy / norm)),
    )

__all__ = [
    '_sample_external_point_side',
    '_mirror_point_x',
    '_apply_external_point_side',
    '_circle_point',
    '_rotated_tangent_unit',
    '_add_points',
    '_split_cyclic_arc_sum',
    '_extend_ray',
]

"""Projection helpers for bearing-route unit geometry to pixels."""

from __future__ import annotations

from collections.abc import Sequence

from .state import BBox, Point


def fit_points_to_box(points: Sequence[Point], bbox: BBox, *, min_scale: float = 6.0) -> tuple[float, Point]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width_units = max(1.0, max_x - min_x)
    height_units = max(1.0, max_y - min_y)
    left, top, right, bottom = (float(v) for v in bbox)
    box_w = max(1.0, right - left)
    box_h = max(1.0, bottom - top)
    scale = max(float(min_scale), min(box_w / width_units, box_h / height_units) * 0.72)
    route_w = width_units * scale
    route_h = height_units * scale
    origin_x = left + ((box_w - route_w) / 2.0) - (min_x * scale)
    origin_y = top + ((box_h - route_h) / 2.0) - (min_y * scale)
    return float(scale), (origin_x, origin_y)


def project_point(point: Point, *, scale: float, origin: Point) -> Point:
    return (float(origin[0]) + (float(point[0]) * float(scale)), float(origin[1]) + (float(point[1]) * float(scale)))


__all__ = ["fit_points_to_box", "project_point"]

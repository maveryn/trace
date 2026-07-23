"""Bearing-route geometry primitives independent of rendering."""

from __future__ import annotations

import math

from .state import Point, RouteCase


def bearing_to_unit_vector(bearing_degrees: int) -> Point:
    theta = math.radians(float(bearing_degrees))
    return (math.sin(theta), -math.cos(theta))


def normalize_bearing(value: int | float) -> int:
    return int(round(float(value))) % 360


def route_unit_points(route_case: RouteCase) -> tuple[Point, Point, Point]:
    u1 = bearing_to_unit_vector(int(route_case.bearing_a))
    u2 = bearing_to_unit_vector(int(route_case.bearing_b))
    p0 = (0.0, 0.0)
    p1 = (u1[0] * float(route_case.leg_a), u1[1] * float(route_case.leg_a))
    p2 = (p1[0] + (u2[0] * float(route_case.leg_b)), p1[1] + (u2[1] * float(route_case.leg_b)))
    return p0, p1, p2


__all__ = ["bearing_to_unit_vector", "normalize_bearing", "route_unit_points"]

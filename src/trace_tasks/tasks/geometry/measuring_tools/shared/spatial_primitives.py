"""Small geometry primitives for visible measuring-tool diagrams."""

from __future__ import annotations

import math

from .state import Point


def add(left: Point, right: Point) -> Point:
    return (float(left[0]) + float(right[0]), float(left[1]) + float(right[1]))


def sub(left: Point, right: Point) -> Point:
    return (float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def scale(point: Point, factor: float) -> Point:
    return (float(point[0]) * float(factor), float(point[1]) * float(factor))


def unit_from_degrees(degree_value: float) -> Point:
    theta = math.radians(float(degree_value))
    return (math.cos(theta), math.sin(theta))


def normal(axis: Point) -> Point:
    return (-float(axis[1]), float(axis[0]))


def protractor_point(center: Point, radius: float, degree_value: float) -> Point:
    theta = math.radians(float(degree_value))
    return (
        float(center[0]) + (float(radius) * math.cos(theta)),
        float(center[1]) - (float(radius) * math.sin(theta)),
    )


def point_to_list(point: Point) -> list[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


__all__ = [
    "add",
    "normal",
    "point_to_list",
    "protractor_point",
    "scale",
    "sub",
    "unit_from_degrees",
]

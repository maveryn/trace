"""Identity-free tangent-packing measurement formulas."""

from __future__ import annotations

import math

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure, round1

from .state import TangentPackingCase


def inscribed_square_side(radius: int) -> float:
    return round1(float(radius) * math.sqrt(2.0))


def square_container_circle_gap(radius: int) -> float:
    r = float(radius)
    return round1((2.0 * r) ** 2 - (math.pi * r * r))


def circle_container_square_gap(radius: int) -> float:
    r = float(radius)
    return round1((math.pi * r * r) - (2.0 * r * r))


def rectangle_equal_circles_gap(radius: int) -> float:
    r = float(radius)
    return round1((4.0 * r) * (2.0 * r) - (2.0 * math.pi * r * r))


def case_trace_values(case: TangentPackingCase) -> dict[str, float | int]:
    return {
        "radius": int(case.radius),
        "square_side": int(case.square_side),
        "container_width": int(case.packed_rectangle_width),
        "container_height": int(case.packed_rectangle_height),
        "inscribed_square_side": float(inscribed_square_side(int(case.radius))),
    }


__all__ = [
    "case_trace_values",
    "circle_container_square_gap",
    "fmt_measure",
    "inscribed_square_side",
    "rectangle_equal_circles_gap",
    "round1",
    "square_container_circle_gap",
]

"""Measurement formatting and formula support for solid-revolution tasks."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure, round1
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map


def format_measure(value: float) -> str:
    """Return a compact geometry measurement label."""

    return fmt_measure(float(value))


def round_volume(value: float) -> float:
    """Round a volume answer to one decimal place."""

    return round1(float(value))


def volume_cylinder(*, diameter: float, height: float) -> float:
    """Return cylinder volume from diameter and height."""

    radius = float(diameter) / 2.0
    return math.pi * radius**2 * float(height)


def volume_cone(*, radius: float, height: float) -> float:
    """Return cone volume from radius and height."""

    return math.pi * float(radius) ** 2 * float(height) / 3.0


def volume_double_cone(*, radius: float, half_height: float) -> float:
    """Return volume of two congruent cones sharing a base."""

    return 2.0 * volume_cone(radius=float(radius), height=float(half_height))


def volume_frustum(*, top_radius: float, bottom_radius: float, height: float) -> float:
    """Return right circular frustum volume."""

    r = float(top_radius)
    radius = float(bottom_radius)
    return math.pi * float(height) * (radius**2 + radius * r + r**2) / 3.0


def unique_answer_support(values: Iterable[float]) -> tuple[float, ...]:
    """Return sorted unique one-decimal answer support values."""

    return tuple(sorted({round_volume(float(value)) for value in values}))


def answer_support_probability_map(support: Sequence[float], answer: float) -> dict[str, float]:
    """Return a one-hot probability map over answer support values."""

    return geometry_selected_probability_map(
        tuple(float(value) for value in support),
        float(answer),
        key_fn=format_measure,
        sort_unique=True,
    )


__all__ = [
    "answer_support_probability_map",
    "format_measure",
    "round_volume",
    "unique_answer_support",
    "volume_cone",
    "volume_cylinder",
    "volume_double_cone",
    "volume_frustum",
]

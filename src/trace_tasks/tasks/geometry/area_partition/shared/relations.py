"""Pure geometric relations for the area-partition scene."""

from __future__ import annotations

from typing import Sequence

from .state import Point


def midpoint(a: Point, b: Point) -> Point:
    """Return the midpoint of two points."""

    return ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0)


def centroid(a: Point, b: Point, c: Point) -> Point:
    """Return the centroid of one triangle."""

    return (
        (float(a[0]) + float(b[0]) + float(c[0])) / 3.0,
        (float(a[1]) + float(b[1]) + float(c[1])) / 3.0,
    )


def total_area_from_unit_partition(*, shaded_area: int, denominator: int) -> int:
    """Infer total area when the shaded region is one equal-area unit."""

    if int(shaded_area) <= 0:
        raise ValueError("shaded area must be positive")
    if int(denominator) <= 1:
        raise ValueError("area denominator must exceed 1")
    return int(shaded_area) * int(denominator)


def selected_probability_map(values: Sequence[float], selected: float) -> dict[str, float]:
    """Return a one-hot probability map for numeric support values."""

    unique = tuple(sorted({float(value) for value in values}))
    return {
        str(int(value) if float(value).is_integer() else value): (
            1.0 if int(value) == int(selected) else 0.0
        )
        for value in unique
    }


__all__ = [
    "centroid",
    "midpoint",
    "selected_probability_map",
    "total_area_from_unit_partition",
]

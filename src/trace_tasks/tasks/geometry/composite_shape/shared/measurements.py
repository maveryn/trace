"""Formula and support helpers for composite-shape measurements."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure, round1
from trace_tasks.tasks.shared.fixed_query import geometry_probability_map, geometry_selected_probability_map

WIDTH_SUPPORT: Tuple[int, ...] = tuple(range(8, 37))
HEIGHT_SUPPORT: Tuple[int, ...] = tuple(range(6, 21))
RADIUS_SUPPORT: Tuple[int, ...] = tuple(range(3, 13))
SECTOR_RADIUS_SUPPORT: Tuple[int, ...] = tuple(range(4, 16))
THETA_SUPPORT: Tuple[int, ...] = tuple(range(35, 156))

SEMICIRCLE_DIMENSION_CANDIDATES: Tuple[Tuple[int, int, int], ...] = tuple(
    (width, height, radius)
    for width in WIDTH_SUPPORT
    for height in HEIGHT_SUPPORT
    for radius in range(3, min(11, height // 2) + 1)
)
QUARTER_CUT_DIMENSION_CANDIDATES: Tuple[Tuple[int, int, int], ...] = tuple(
    (width, height, radius)
    for width in range(9, 31)
    for height in range(7, 21)
    for radius in range(3, min(12, width - 4, height - 3) + 1)
)
SECTOR_DIMENSION_CANDIDATES: Tuple[Tuple[int, int], ...] = tuple(
    (theta, radius)
    for theta in THETA_SUPPORT
    for radius in SECTOR_RADIUS_SUPPORT
)


def dimension_values(index: int) -> tuple[int, int, int]:
    """Return width, height, and radius support values for curved composites."""

    selected_index = int(index)
    if selected_index < 0 or selected_index >= len(SEMICIRCLE_DIMENSION_CANDIDATES):
        raise ValueError("dimension index is outside semicircle candidate support")
    return SEMICIRCLE_DIMENSION_CANDIDATES[selected_index]


def sector_values(index: int) -> tuple[int, int]:
    """Return central-angle and radius support values for sector tasks."""

    selected_index = int(index)
    if selected_index < 0 or selected_index >= len(SECTOR_DIMENSION_CANDIDATES):
        raise ValueError("sector index is outside sector candidate support")
    return SECTOR_DIMENSION_CANDIDATES[selected_index]


def one_hot_support(values: Sequence[Any], selected: Any) -> Dict[str, float]:
    """Return a one-hot support probability map with JSON-stable keys."""

    return geometry_selected_probability_map(
        values,
        selected,
        is_selected=lambda value, target: float(value) == float(target),
    )


def uniform_support(values: Sequence[Any]) -> Dict[str, float]:
    """Return a uniform support probability map for trace metadata."""

    return geometry_probability_map(values, sort_unique=True)


def semicircle_area(radius_units: int) -> float:
    """Return the area of a semicircle with the supplied radius."""

    return 0.5 * math.pi * float(radius_units) ** 2


def semicircle_arc_length(radius_units: int) -> float:
    """Return the arc length of a semicircle with the supplied radius."""

    return math.pi * float(radius_units)


def semicircle_side_remainder_straight_boundary(
    width_units: int,
    height_units: int,
    radius_units: int,
) -> float:
    """Return straight perimeter parts when a semicircle spans part of one side."""

    return (
        (2.0 * float(width_units))
        + (2.0 * float(height_units))
        - (2.0 * float(radius_units))
    )


def semicircle_side_remainder_perimeter(
    width_units: int,
    height_units: int,
    radius_units: int,
) -> float:
    """Return total perimeter with side remainders plus one semicircle arc."""

    return round1(
        semicircle_side_remainder_straight_boundary(
            int(width_units),
            int(height_units),
            int(radius_units),
        )
        + semicircle_arc_length(int(radius_units))
    )


def quarter_sector_values(radius_units: int) -> tuple[float, float]:
    """Return the area and arc length of a quarter-circle cutout."""

    sector_area = 0.25 * math.pi * float(radius_units) ** 2
    arc_length = 0.5 * math.pi * float(radius_units)
    return float(sector_area), float(arc_length)


def sector_arc_length(theta_degrees: int, radius_units: int) -> float:
    """Return a sector arc length for a central angle in degrees."""

    return (float(theta_degrees) / 360.0) * 2.0 * math.pi * float(radius_units)


def sector_area(theta_degrees: int, radius_units: int) -> float:
    """Return a sector area for a central angle in degrees."""

    return (float(theta_degrees) / 360.0) * math.pi * float(radius_units) ** 2


def format_given(value: float) -> str:
    """Format one visible decimal given with one digit."""

    return f"{float(value):.1f}"


def numeric_prompt_slots(values: Mapping[str, Any]) -> Dict[str, str]:
    """Return prompt slots for optional numeric values used by templates."""

    return {
        "total_area": format_given(float(values.get("total_area", 0.0))),
        "arc_length": format_given(float(values.get("arc_length", 0.0))),
        "sector_area": format_given(float(values.get("sector_area", 0.0))),
    }


__all__ = [
    "HEIGHT_SUPPORT",
    "RADIUS_SUPPORT",
    "QUARTER_CUT_DIMENSION_CANDIDATES",
    "SECTOR_DIMENSION_CANDIDATES",
    "SECTOR_RADIUS_SUPPORT",
    "SEMICIRCLE_DIMENSION_CANDIDATES",
    "THETA_SUPPORT",
    "WIDTH_SUPPORT",
    "dimension_values",
    "fmt_measure",
    "format_given",
    "numeric_prompt_slots",
    "one_hot_support",
    "quarter_sector_values",
    "round1",
    "sector_arc_length",
    "sector_area",
    "sector_values",
    "semicircle_arc_length",
    "semicircle_area",
    "semicircle_side_remainder_perimeter",
    "semicircle_side_remainder_straight_boundary",
    "uniform_support",
]

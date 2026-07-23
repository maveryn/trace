"""Neutral survey-traverse measurement helpers."""

from __future__ import annotations

import math
from typing import Sequence

from .state import Point


def normalize_bearing(value: int | float) -> int:
    """Normalize a bearing to integer degrees in the half-open circle."""

    return int(round(float(value))) % 360


def bearing_to_unit_vector(bearing_degrees: int | float) -> Point:
    """Convert clockwise-from-north bearing degrees into image-space unit vector."""

    theta = math.radians(float(bearing_degrees))
    return (math.sin(theta), -math.cos(theta))


def direction_endpoint(start: Point, bearing: int, length: float) -> Point:
    """Project one survey ray endpoint from a start point, bearing, and pixel length."""

    vx, vy = bearing_to_unit_vector(float(bearing))
    return (float(start[0]) + vx * float(length), float(start[1]) + vy * float(length))


def offset_area_from_chainages(chainages: Sequence[int], offsets: Sequence[int]) -> int:
    """Return area from baseline chainages and perpendicular offsets."""

    if len(chainages) != len(offsets):
        raise ValueError("chainage and offset counts must match")
    twice_area = 0
    for left, right in zip(range(len(chainages) - 1), range(1, len(chainages))):
        width = int(chainages[right]) - int(chainages[left])
        if width <= 0:
            raise ValueError("chainages must be strictly increasing")
        twice_area += int(width) * (int(offsets[left]) + int(offsets[right]))
    if twice_area % 2 != 0:
        raise ValueError("survey offset case must have integer area")
    return int(twice_area // 2)


__all__ = [
    "bearing_to_unit_vector",
    "direction_endpoint",
    "normalize_bearing",
    "offset_area_from_chainages",
]

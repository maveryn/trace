"""Shared metadata helpers for non-semantic visual style renderers."""

from __future__ import annotations

from typing import Any, Sequence

from ..color_distance import color_distance, rgb_euclidean_distance
from .palette import Color


def color_separation_metadata(
    *,
    anchor_rgb: Color,
    compared_rgbs: Sequence[Color],
    prefix: str,
) -> dict[str, Any]:
    """Return Lab and RGB distance summaries for one anchor color."""

    compared = [tuple(int(value) for value in color[:3]) for color in compared_rgbs]
    if not compared:
        return {
            f"min_{prefix}_lab_distance": 0.0,
            f"{prefix}_lab_distances": [],
            f"min_{prefix}_rgb_distance": 0.0,
            f"{prefix}_rgb_distances": [],
        }
    lab_distances = [color_distance(anchor_rgb, color, distance_space="lab") for color in compared]
    rgb_distances = [rgb_euclidean_distance(anchor_rgb, color) for color in compared]
    return {
        f"min_{prefix}_lab_distance": round(float(min(lab_distances)), 3),
        f"{prefix}_lab_distances": [round(float(value), 3) for value in lab_distances],
        f"min_{prefix}_rgb_distance": round(float(min(rgb_distances)), 3),
        f"{prefix}_rgb_distances": [round(float(value), 3) for value in rgb_distances],
    }


__all__ = ["color_separation_metadata"]

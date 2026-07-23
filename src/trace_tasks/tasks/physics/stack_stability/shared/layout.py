"""Layout helpers for stack-stability diagrams."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def bbox_from_center(
    center: Tuple[float, float],
    half_w: float,
    half_h: float,
) -> List[float]:
    """Build a rounded pixel bbox centered at one point."""

    return [
        round(float(center[0] - half_w), 3),
        round(float(center[1] - half_h), 3),
        round(float(center[0] + half_w), 3),
        round(float(center[1] + half_h), 3),
    ]


def expand_bbox(bbox: Sequence[float], padding: float) -> List[float]:
    """Expand one bbox by a fixed pixel padding."""

    return [
        round(float(bbox[0]) - float(padding), 3),
        round(float(bbox[1]) - float(padding), 3),
        round(float(bbox[2]) + float(padding), 3),
        round(float(bbox[3]) + float(padding), 3),
    ]


__all__ = ["bbox_from_center", "expand_bbox"]

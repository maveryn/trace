"""Shared vector-arrow geometry helpers for physics diagrams."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from ...shared.drawing import draw_arrow

Point = Tuple[float, float]
BBox = List[float]
DirectionVectorMap = Mapping[str, Tuple[int, int]]

SEMANTIC_DIRECTION_VECTORS: Dict[str, Tuple[int, int]] = {
    "east": (1, 0),
    "northeast": (1, 1),
    "north": (0, 1),
    "northwest": (-1, 1),
    "west": (-1, 0),
    "southwest": (-1, -1),
    "south": (0, -1),
    "southeast": (1, -1),
}


def arrow_bbox(start: Sequence[float], end: Sequence[float], *, padding_px: float) -> BBox:
    """Return a conservative screen-space bbox for one straight arrow."""

    return [
        round(float(min(float(start[0]), float(end[0])) - float(padding_px)), 3),
        round(float(min(float(start[1]), float(end[1])) - float(padding_px)), 3),
        round(float(max(float(start[0]), float(end[0])) + float(padding_px)), 3),
        round(float(max(float(start[1]), float(end[1])) + float(padding_px)), 3),
    ]


def direction_unit_vector(
    direction: str,
    *,
    direction_vectors: DirectionVectorMap = SEMANTIC_DIRECTION_VECTORS,
    screen_y_down: bool = True,
) -> Point:
    """Return a normalized screen-space unit vector for a named physics direction."""

    dx, dy = direction_vectors[str(direction)]
    magnitude = max(1.0, math.hypot(float(dx), float(dy)))
    screen_dy = -float(dy) if bool(screen_y_down) else float(dy)
    return (float(dx) / magnitude, screen_dy / magnitude)


def direction_endpoint(
    center: Sequence[float],
    *,
    direction: str,
    length_px: float,
    direction_vectors: DirectionVectorMap = SEMANTIC_DIRECTION_VECTORS,
    screen_y_down: bool = True,
) -> Point:
    """Return the endpoint after moving from center along a named direction."""

    unit_x, unit_y = direction_unit_vector(
        str(direction),
        direction_vectors=direction_vectors,
        screen_y_down=bool(screen_y_down),
    )
    return (float(center[0]) + unit_x * float(length_px), float(center[1]) + unit_y * float(length_px))


def centered_arrow_endpoints(
    center: Sequence[float],
    *,
    direction: str,
    length_px: float,
    direction_vectors: DirectionVectorMap = SEMANTIC_DIRECTION_VECTORS,
    screen_y_down: bool = True,
    half_fraction: float = 0.5,
) -> Tuple[Point, Point]:
    """Return start/end points for an arrow centered around a point."""

    unit_x, unit_y = direction_unit_vector(
        str(direction),
        direction_vectors=direction_vectors,
        screen_y_down=bool(screen_y_down),
    )
    half_length = float(length_px) * float(half_fraction)
    start = (float(center[0]) - unit_x * half_length, float(center[1]) - unit_y * half_length)
    end = (float(center[0]) + unit_x * half_length, float(center[1]) + unit_y * half_length)
    return start, end


def draw_arrow_with_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    start: Sequence[float],
    end: Sequence[float],
    fill: Tuple[int, int, int],
    width: int,
    head_length_px: float,
    head_width_px: float,
    padding_px: float,
) -> BBox:
    """Draw an arrow and return the conservative bbox used for metadata/annotation."""

    start_point = (float(start[0]), float(start[1]))
    end_point = (float(end[0]), float(end[1]))
    draw_arrow(
        draw,
        start=start_point,
        end=end_point,
        fill=tuple(int(value) for value in fill),
        width=max(1, int(width)),
        head_length_px=float(head_length_px),
        head_width_px=float(head_width_px),
    )
    return arrow_bbox(start_point, end_point, padding_px=float(padding_px))


__all__ = [
    "SEMANTIC_DIRECTION_VECTORS",
    "arrow_bbox",
    "centered_arrow_endpoints",
    "direction_endpoint",
    "direction_unit_vector",
    "draw_arrow_with_bbox",
]

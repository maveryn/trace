"""Canvas-bound helpers for illustration scene renderers."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .object_library import BBox


def bbox_inside_canvas(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    margin: float = 0.0,
) -> bool:
    """Return whether a bbox is fully inside the canvas plus optional margin."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (
        float(margin) <= x0 < x1 <= float(width) - float(margin)
        and float(margin) <= y0 < y1 <= float(height) - float(margin)
    )


def clamp_bbox_to_canvas(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    margin: float = 0.0,
) -> BBox:
    """Shift and shrink one bbox so it remains fully visible on the canvas."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    max_w = max(1.0, float(width) - 2.0 * float(margin))
    max_h = max(1.0, float(height) - 2.0 * float(margin))
    box_w = min(box_w, max_w)
    box_h = min(box_h, max_h)
    max_x0 = max(float(margin), float(width) - float(margin) - box_w)
    max_y0 = max(float(margin), float(height) - float(margin) - box_h)
    clamped_x0 = max(float(margin), min(x0, max_x0))
    clamped_y0 = max(float(margin), min(y0, max_y0))
    return (
        round(clamped_x0, 3),
        round(clamped_y0, 3),
        round(clamped_x0 + box_w, 3),
        round(clamped_y0 + box_h, 3),
    )


def clamp_box_size_to_region(
    *,
    width: float,
    height: float,
    region_bbox: Sequence[float],
    padding_x: float,
    padding_y: float,
    min_width: float,
    min_height: float,
) -> Tuple[float, float]:
    """Return an item size that can fit inside a padded region."""

    x0, y0, x1, y1 = [float(value) for value in region_bbox]
    available_w = max(float(min_width), x1 - x0 - 2.0 * float(padding_x))
    available_h = max(float(min_height), y1 - y0 - 2.0 * float(padding_y))
    return (
        max(float(min_width), min(float(width), available_w)),
        max(float(min_height), min(float(height), available_h)),
    )


def out_of_canvas_bboxes(
    bboxes: Iterable[Sequence[float]],
    *,
    width: int,
    height: int,
    margin: float = 0.0,
) -> list[list[float]]:
    """Return bboxes that fail the canvas-bound check."""

    return [
        [round(float(value), 3) for value in bbox]
        for bbox in bboxes
        if not bbox_inside_canvas(bbox, width=int(width), height=int(height), margin=float(margin))
    ]


__all__ = [
    "bbox_inside_canvas",
    "clamp_bbox_to_canvas",
    "clamp_box_size_to_region",
    "out_of_canvas_bboxes",
]

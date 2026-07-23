"""Geometry primitives for Venn-field icon placement."""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

from ...shared.icon_scene import BBox

from .state import VennSpec


def bbox_from_center(
    center_xy: Sequence[float], sprite_size: Sequence[int]
) -> Tuple[int, int, int, int]:
    """Return an integer bbox for a sprite centered at `center_xy`."""

    cx, cy = float(center_xy[0]), float(center_xy[1])
    w, h = int(sprite_size[0]), int(sprite_size[1])
    x0 = int(round(cx - 0.5 * float(w)))
    y0 = int(round(cy - 0.5 * float(h)))
    return int(x0), int(y0), int(x0 + w), int(y0 + h)


def sample_venn_spec(rng: Any, *, content_bbox: BBox) -> VennSpec:
    """Sample two overlapping circles within one panel content box."""

    x0, y0, x1, y1 = [float(value) for value in content_bbox]
    width = float(x1 - x0)
    height = float(y1 - y0)
    radius = float(min(width * 0.24, height * 0.43))
    radius = float(max(118.0, min(radius, 158.0)))
    separation = float(radius * rng.uniform(1.02, 1.18))
    center_x = 0.5 * float(x0 + x1)
    center_y = float(y0) + (0.50 * height)
    left_center = (center_x - (0.5 * separation), center_y)
    right_center = (center_x + (0.5 * separation), center_y)

    def _circle_bbox(center: Sequence[float]) -> Tuple[int, int, int, int]:
        cx, cy = float(center[0]), float(center[1])
        return (
            int(round(cx - radius)),
            int(round(cy - radius)),
            int(round(cx + radius)),
            int(round(cy + radius)),
        )

    return VennSpec(
        left_center_xy=(float(left_center[0]), float(left_center[1])),
        right_center_xy=(float(right_center[0]), float(right_center[1])),
        radius_px=float(radius),
        left_bbox_xyxy=_circle_bbox(left_center),
        right_bbox_xyxy=_circle_bbox(right_center),
    )


def category_membership(category: str) -> Tuple[bool, bool]:
    """Return left/right circle membership for one Venn category."""

    if str(category) == "both":
        return True, True
    if str(category) == "left_only":
        return True, False
    if str(category) == "right_only":
        return False, True
    if str(category) == "neither":
        return False, False
    raise ValueError(f"unsupported Venn category: {category}")


def bbox_venn_category(
    venn: VennSpec, box: Sequence[int | float], *, margin_px: int
) -> str | None:
    """Return a stable Venn category when a bbox clears circle boundaries."""

    left_inside = _bbox_inside_circle(
        box,
        center_xy=venn.left_center_xy,
        radius_px=venn.radius_px,
        margin_px=int(margin_px),
    )
    right_inside = _bbox_inside_circle(
        box,
        center_xy=venn.right_center_xy,
        radius_px=venn.radius_px,
        margin_px=int(margin_px),
    )
    left_outside = _bbox_outside_circle(
        box,
        center_xy=venn.left_center_xy,
        radius_px=venn.radius_px,
        margin_px=int(margin_px),
    )
    right_outside = _bbox_outside_circle(
        box,
        center_xy=venn.right_center_xy,
        radius_px=venn.radius_px,
        margin_px=int(margin_px),
    )
    if left_inside and right_inside:
        return "both"
    if left_inside and right_outside:
        return "left_only"
    if left_outside and right_inside:
        return "right_only"
    if left_outside and right_outside:
        return "neither"
    return None


def sample_center_for_category(
    rng: Any,
    *,
    content_bbox: BBox,
    sprite_size: Tuple[int, int],
    venn: VennSpec,
    category: str,
    margin_px: int,
) -> Tuple[float, float, Tuple[int, int, int, int]]:
    """Sample a sprite center whose bbox is safely within one Venn category."""

    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    half_w = 0.5 * float(sprite_size[0])
    half_h = 0.5 * float(sprite_size[1])
    min_x = float(x0) + half_w
    max_x = float(x1) - half_w
    min_y = float(y0) + half_h
    max_y = float(y1) - half_h
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("sprite does not fit content bbox")
    for _ in range(1200):
        cx = float(rng.uniform(min_x, max_x))
        cy = float(rng.uniform(min_y, max_y))
        bbox = bbox_from_center((cx, cy), sprite_size)
        if bbox_venn_category(venn, bbox, margin_px=int(margin_px)) == str(category):
            return float(cx), float(cy), tuple(int(value) for value in bbox)
    raise ValueError(f"could not sample icon center in Venn category {category}")


def venn_to_trace(venn: VennSpec) -> dict[str, Any]:
    """Serialize Venn circle geometry for trace metadata."""

    return {
        "left_circle": {
            "center_xy": [float(value) for value in venn.left_center_xy],
            "radius_px": float(venn.radius_px),
            "bbox_xyxy": [int(value) for value in venn.left_bbox_xyxy],
        },
        "right_circle": {
            "center_xy": [float(value) for value in venn.right_center_xy],
            "radius_px": float(venn.radius_px),
            "bbox_xyxy": [int(value) for value in venn.right_bbox_xyxy],
        },
    }


def _bbox_corners(box: Sequence[int | float]) -> Tuple[Tuple[float, float], ...]:
    x0, y0, x1, y1 = [float(value) for value in box]
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _corner_distances_to_circle(
    box: Sequence[int | float], center_xy: Sequence[float]
) -> Tuple[float, ...]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    return tuple(
        math.hypot(float(x) - cx, float(y) - cy) for x, y in _bbox_corners(box)
    )


def _bbox_inside_circle(
    box: Sequence[int | float],
    *,
    center_xy: Sequence[float],
    radius_px: float,
    margin_px: int,
) -> bool:
    threshold = float(radius_px) - float(max(0, int(margin_px)))
    if threshold <= 0.0:
        return False
    return all(
        float(distance) <= threshold
        for distance in _corner_distances_to_circle(box, center_xy)
    )


def _bbox_outside_circle(
    box: Sequence[int | float],
    *,
    center_xy: Sequence[float],
    radius_px: float,
    margin_px: int,
) -> bool:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x0, y0, x1, y1 = [float(value) for value in box]
    nearest_x = min(max(cx, x0), x1)
    nearest_y = min(max(cy, y0), y1)
    return math.hypot(nearest_x - cx, nearest_y - cy) >= float(radius_px) + float(
        max(0, int(margin_px))
    )


__all__ = [
    "bbox_venn_category",
    "category_membership",
    "sample_center_for_category",
    "sample_venn_spec",
    "venn_to_trace",
]

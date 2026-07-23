"""Placement and region-layout helpers for named-field icon scenes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ...shared.icon_scene import BBox
from ...shared.procedural_named_icon_field_scene import (
    bbox_center_float,
    bbox_from_center_and_size,
    label_bbox_for_icon,
    union_bbox,
)

from .spatial_primitives import band_normal, bbox_safely_matches_region
from .state import RegionSpec


def occupancy_bbox_for_icon(
    *,
    icon_bbox: BBox,
    label: str,
    content_bbox: BBox,
    label_font,
    render_params: Mapping[str, Any],
) -> BBox:
    """Return the bbox occupied by an icon plus its optional nearby label."""

    if not str(label):
        return tuple(int(value) for value in icon_bbox)
    label_bbox = label_bbox_for_icon(
        icon_bbox=tuple(int(value) for value in icon_bbox),
        label=str(label),
        content_bbox=tuple(int(value) for value in content_bbox),
        font=label_font,
        padding_px=int(render_params["candidate_label_padding_px"]),
        gap_px=int(render_params["candidate_label_gap_px"]),
    )
    return union_bbox(tuple(int(value) for value in icon_bbox), label_bbox)


def candidate_distances(rng, *, render_params: Mapping[str, Any], option_count: int) -> Tuple[float, ...]:
    """Sample monotonically increasing candidate distances for rank tasks."""

    margin = max(10, int(render_params["distance_rank_margin_px"]))
    min_distance = max(74, int(render_params["center_distance_min_px"]))
    jitter = max(0, int(render_params["center_distance_gap_jitter_px"]))
    values: list[float] = [float(min_distance + int(rng.randint(0, max(1, margin // 2))))]
    for _ in range(1, int(option_count)):
        values.append(float(values[-1] + margin + (int(rng.randint(0, jitter)) if jitter > 0 else 0)))
    return tuple(float(value) for value in values)


def sample_box_region(
    rng,
    *,
    query_key: str,
    counts_inside: bool,
    content_bbox: BBox,
    shape_kind: str,
) -> RegionSpec:
    """Sample a rectangular or elliptical region."""

    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    width = int(x1 - x0)
    height = int(y1 - y0)
    box_w = int(round(float(width) * float(rng.uniform(0.36, 0.56))))
    box_h = int(round(float(height) * float(rng.uniform(0.38, 0.62))))
    box_x0 = int(rng.randint(int(x0 + 18), int(max(x0 + 18, x1 - box_w - 18))))
    box_y0 = int(rng.randint(int(y0 + 16), int(max(y0 + 16, y1 - box_h - 16))))
    bbox = (int(box_x0), int(box_y0), int(box_x0 + box_w), int(box_y0 + box_h))
    center = bbox_center_float(bbox)
    return RegionSpec(
        query_key=str(query_key),
        region_kind="shape",
        counts_inside=bool(counts_inside),
        shape_kind=str(shape_kind),
        bbox_xyxy=bbox,
        ellipse_center_xy=center if str(shape_kind) == "ellipse" else None,
        ellipse_radii_xy=(0.5 * float(box_w), 0.5 * float(box_h)) if str(shape_kind) == "ellipse" else None,
    )


def sample_band_region(
    rng,
    *,
    query_key: str,
    counts_inside: bool,
    content_bbox: BBox,
    band_kind: str,
) -> RegionSpec:
    """Sample a visible band region with clear interior/exterior separation."""

    x0, y0, x1, y1 = [float(value) for value in content_bbox]
    width = float(x1 - x0)
    height = float(y1 - y0)
    nx, ny = band_normal(str(band_kind))
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    values = [(float(x) * float(nx)) + (float(y) * float(ny)) for x, y in corners]
    min_value = min(values)
    max_value = max(values)
    span = max(1.0, float(max_value - min_value))
    half_width = float(rng.uniform(0.13, 0.20)) * min(width, height)
    center_min = min_value + (0.32 * span)
    center_max = max_value - (0.32 * span)
    center_distance = float(rng.uniform(center_min, center_max)) if center_min < center_max else 0.5 * (min_value + max_value)
    tx, ty = -float(ny), float(nx)
    base_x = float(nx) * float(center_distance)
    base_y = float(ny) * float(center_distance)
    line_half_len = 2.0 * math.hypot(width, height)
    polygon = (
        (base_x - float(nx) * half_width - tx * line_half_len, base_y - float(ny) * half_width - ty * line_half_len),
        (base_x - float(nx) * half_width + tx * line_half_len, base_y - float(ny) * half_width + ty * line_half_len),
        (base_x + float(nx) * half_width + tx * line_half_len, base_y + float(ny) * half_width + ty * line_half_len),
        (base_x + float(nx) * half_width - tx * line_half_len, base_y + float(ny) * half_width - ty * line_half_len),
    )
    return RegionSpec(
        query_key=str(query_key),
        region_kind="band",
        counts_inside=bool(counts_inside),
        band_kind=str(band_kind),
        band_normal_xy=(float(nx), float(ny)),
        band_center_distance=float(center_distance),
        band_half_width_px=float(half_width),
        band_polygon_xy=tuple((float(x), float(y)) for x, y in polygon),
    )


def sample_quadrant_region(rng, *, query_key: str, content_bbox: BBox, quadrant_id: str) -> RegionSpec:
    """Sample one quadrant region from the content area."""

    del rng
    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    xm = int(round(0.5 * float(x0 + x1)))
    ym = int(round(0.5 * float(y0 + y1)))
    quadrant_to_bbox = {
        "top_left": (x0, y0, xm, ym),
        "top_right": (xm, y0, x1, ym),
        "bottom_left": (x0, ym, xm, y1),
        "bottom_right": (xm, ym, x1, y1),
    }
    bbox = quadrant_to_bbox[str(quadrant_id)]
    return RegionSpec(
        query_key=str(query_key),
        region_kind="quadrant",
        counts_inside=True,
        shape_kind="rectangle",
        quadrant_id=str(quadrant_id),
        bbox_xyxy=tuple(int(value) for value in bbox),
    )


def sample_shelf_region(
    rng,
    *,
    query_key: str,
    content_bbox: BBox,
    shelf_count_min: int,
    shelf_count_max: int,
) -> RegionSpec:
    """Sample one shelf-row region from the content area."""

    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    shelf_count = int(rng.randint(int(shelf_count_min), int(shelf_count_max)))
    shelf_index = int(rng.randrange(0, int(shelf_count)))
    shelf_h = float(y1 - y0) / float(max(1, int(shelf_count)))
    sy0 = int(round(float(y0) + float(shelf_index) * shelf_h))
    sy1 = int(round(float(y0) + float(shelf_index + 1) * shelf_h))
    return RegionSpec(
        query_key=str(query_key),
        region_kind="shelf",
        counts_inside=True,
        shape_kind="rectangle",
        shelf_index=int(shelf_index),
        shelf_count=int(shelf_count),
        bbox_xyxy=(int(x0), int(sy0), int(x1), int(sy1)),
    )


def sample_region_icon_center(
    rng,
    *,
    content_bbox: BBox,
    sprite_size: Tuple[int, int],
    region: RegionSpec,
    desired_inside: bool,
    margin_px: int,
) -> Tuple[float, float]:
    """Sample an icon center whose full bbox is safely inside/outside a region."""

    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    half_w = 0.5 * float(sprite_size[0])
    half_h = 0.5 * float(sprite_size[1])
    min_x = float(x0) + half_w
    max_x = float(x1) - half_w
    min_y = float(y0) + half_h
    max_y = float(y1) - half_h
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("sprite does not fit content bbox")
    for _ in range(900):
        cx = float(rng.uniform(min_x, max_x))
        cy = float(rng.uniform(min_y, max_y))
        bbox = bbox_from_center_and_size((cx, cy), sprite_size)
        if bbox_safely_matches_region(region, bbox, desired_inside=bool(desired_inside), margin_px=int(margin_px)):
            return (float(cx), float(cy))
    raise ValueError("could not sample center with requested region membership")


__all__ = [
    "candidate_distances",
    "occupancy_bbox_for_icon",
    "sample_band_region",
    "sample_box_region",
    "sample_quadrant_region",
    "sample_region_icon_center",
    "sample_shelf_region",
]

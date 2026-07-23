"""Pure spatial geometry helpers for named-field icon scenes."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from ...shared.icon_scene import BBox

from .state import RegionSpec


def axis_radius(center: Tuple[float, float], axis: Tuple[float, float], content_bbox: BBox) -> float:
    """Return the symmetric radius available from ``center`` along ``axis``."""

    cx, cy = float(center[0]), float(center[1])
    dx, dy = float(axis[0]), float(axis[1])
    x0, y0, x1, y1 = tuple(float(value) for value in content_bbox)

    def forward_limit(sign: float) -> float:
        limits = []
        if abs(dx) > 1e-9:
            limits.append(((x1 if sign * dx > 0 else x0) - cx) / (sign * dx))
        if abs(dy) > 1e-9:
            limits.append(((y1 if sign * dy > 0 else y0) - cy) / (sign * dy))
        positives = [float(value) for value in limits if float(value) > 0.0]
        return min(positives) if positives else 0.0

    return float(min(forward_limit(1.0), forward_limit(-1.0)))


def band_normal(kind: str) -> Tuple[float, float]:
    """Return a normal vector for a visible band orientation."""

    if str(kind) == "vertical":
        return (1.0, 0.0)
    if str(kind) == "horizontal":
        return (0.0, 1.0)
    if str(kind) == "slanted_positive":
        return (math.sqrt(0.5), -math.sqrt(0.5))
    if str(kind) == "slanted_negative":
        return (math.sqrt(0.5), math.sqrt(0.5))
    raise ValueError(f"unsupported band kind: {kind}")


def point_inside_region(region: RegionSpec, center_xy: Sequence[float]) -> bool:
    """Return whether a point lies inside a visible scoped-count region."""

    cx, cy = float(center_xy[0]), float(center_xy[1])
    if region.region_kind in {"shape", "quadrant", "shelf"}:
        if region.shape_kind == "ellipse":
            if region.ellipse_center_xy is None or region.ellipse_radii_xy is None:
                raise ValueError("ellipse region is missing center/radii")
            ex, ey = region.ellipse_center_xy
            rx, ry = region.ellipse_radii_xy
            return ((cx - float(ex)) / max(1e-6, float(rx))) ** 2 + ((cy - float(ey)) / max(1e-6, float(ry))) ** 2 <= 1.0
        if region.bbox_xyxy is None:
            raise ValueError("box-like region is missing bbox")
        x0, y0, x1, y1 = [float(value) for value in region.bbox_xyxy]
        return x0 <= cx <= x1 and y0 <= cy <= y1
    if region.region_kind == "band":
        if region.band_normal_xy is None or region.band_center_distance is None or region.band_half_width_px is None:
            raise ValueError("band region is missing normal/center/width")
        nx, ny = region.band_normal_xy
        distance = abs((float(cx) * float(nx)) + (float(cy) * float(ny)) - float(region.band_center_distance))
        return distance <= float(region.band_half_width_px)
    raise ValueError(f"unsupported region kind: {region.region_kind}")


def _bbox_corners(box: Sequence[int | float]) -> Tuple[Tuple[float, float], ...]:
    x0, y0, x1, y1 = [float(value) for value in box]
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def bbox_safely_matches_region(
    region: RegionSpec,
    bbox_xyxy: Sequence[int | float],
    *,
    desired_inside: bool,
    margin_px: int,
) -> bool:
    """Evaluate region membership with boundary clearance for stable annotations."""

    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    margin = float(max(0, int(margin_px)))
    if region.region_kind in {"shape", "quadrant", "shelf"}:
        if region.shape_kind == "ellipse":
            if region.ellipse_center_xy is None or region.ellipse_radii_xy is None:
                raise ValueError("ellipse region is missing center/radii")
            ex, ey = region.ellipse_center_xy
            rx, ry = region.ellipse_radii_xy
            scale_margin = margin / max(1.0, min(float(rx), float(ry)))
            if bool(desired_inside):
                threshold = max(0.0, (1.0 - scale_margin) ** 2)
                return all(
                    ((float(cx) - float(ex)) / max(1e-6, float(rx))) ** 2
                    + ((float(cy) - float(ey)) / max(1e-6, float(ry))) ** 2
                    <= threshold
                    for cx, cy in _bbox_corners(bbox_xyxy)
                )
            nearest_x = min(max(float(ex), float(x0)), float(x1))
            nearest_y = min(max(float(ey), float(y0)), float(y1))
            value = ((float(nearest_x) - float(ex)) / max(1e-6, float(rx))) ** 2 + (
                (float(nearest_y) - float(ey)) / max(1e-6, float(ry))
            ) ** 2
            return value >= (1.0 + scale_margin) ** 2
        if region.bbox_xyxy is None:
            raise ValueError("box-like region is missing bbox")
        rx0, ry0, rx1, ry1 = [float(value) for value in region.bbox_xyxy]
        if bool(desired_inside):
            return rx0 + margin <= x0 and x1 <= rx1 - margin and ry0 + margin <= y0 and y1 <= ry1 - margin
        return x1 <= rx0 - margin or x0 >= rx1 + margin or y1 <= ry0 - margin or y0 >= ry1 + margin
    if region.region_kind == "band":
        if region.band_normal_xy is None or region.band_center_distance is None or region.band_half_width_px is None:
            raise ValueError("band region is missing normal/center/width")
        nx, ny = region.band_normal_xy
        signed_distances = [
            (float(cx) * float(nx)) + (float(cy) * float(ny)) - float(region.band_center_distance)
            for cx, cy in _bbox_corners(bbox_xyxy)
        ]
        lower = min(float(value) for value in signed_distances)
        upper = max(float(value) for value in signed_distances)
        half_width = float(region.band_half_width_px)
        if bool(desired_inside):
            return lower >= -half_width + margin and upper <= half_width - margin
        return lower >= half_width + margin or upper <= -half_width - margin
    raise ValueError(f"unsupported region kind: {region.region_kind}")


__all__ = [
    "axis_radius",
    "band_normal",
    "bbox_safely_matches_region",
    "point_inside_region",
]

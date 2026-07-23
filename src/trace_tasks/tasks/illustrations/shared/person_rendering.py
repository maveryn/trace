"""Shared visual variation helpers for synthetic people."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from PIL import ImageDraw

from .render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points

BBox = Tuple[float, float, float, float]
RGB = Tuple[int, int, int]

PERSON_GENDER_IDS: Tuple[str, ...] = ("male", "female")
_HAIR_RGB: RGB = (55, 45, 39)


def sample_person_gender(rng) -> str:
    """Sample a render-only person appearance variant with 50/50 probability."""

    return PERSON_GENDER_IDS[int(rng.randint(0, 1))]


def normalize_person_gender(gender_id: str | None) -> str:
    """Return a supported render-only person appearance variant."""

    if gender_id in set(PERSON_GENDER_IDS):
        return str(gender_id)
    return "male"




def _quadratic_points(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float], *, steps: int = 8) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(int(steps) + 1):
        t = float(index) / float(max(1, int(steps)))
        one_minus = 1.0 - t
        x = one_minus * one_minus * float(p0[0]) + 2.0 * one_minus * t * float(p1[0]) + t * t * float(p2[0])
        y = one_minus * one_minus * float(p0[1]) + 2.0 * one_minus * t * float(p1[1]) + t * t * float(p2[1])
        points.append((x, y))
    return points


def _draw_bottom_arced_hair_shape(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    arc_start_fraction: float,
    scale: int,
    outline: RGB | None,
) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    arc_y = y0 + max(0.2, min(0.9, float(arc_start_fraction))) * (y1 - y0)
    cx = 0.5 * (x0 + x1)
    rx = 0.5 * (x1 - x0)
    ry = max(1.0, y1 - arc_y)
    points: list[tuple[float, float]] = [(x0, y0), (x1, y0), (x1, arc_y)]
    for index in range(1, 8):
        theta = math.pi * float(index) / 8.0
        points.append((cx + rx * math.cos(theta), arc_y + ry * math.sin(theta)))
    points.extend([(x0, arc_y), (x0, y0)])
    draw.polygon(_scale_points(points, scale), fill=_HAIR_RGB)
    if outline is not None:
        draw.line(_scale_points(points, scale), fill=outline, width=max(1, int(scale)))


def draw_person_hair_back(
    draw: ImageDraw.ImageDraw,
    *,
    head_bbox: BBox,
    gender_id: str,
    scale: int,
    outline: RGB | None,
) -> None:
    """Draw hair shapes that should sit behind the head."""

    if normalize_person_gender(gender_id) != "female":
        return
    x0, y0, x1, y1 = [float(value) for value in head_bbox]
    w = x1 - x0
    h = y1 - y0
    crown = (x0 - 0.24 * w, y0 - 0.26 * h, x1 + 0.24 * w, y1 + 0.38 * h)
    draw.ellipse(_scale_bbox(crown, scale), fill=_HAIR_RGB, outline=outline, width=max(1, int(scale)))
    left_outer = _quadratic_points(
        (x0 - 0.20 * w, y0 + 0.14 * h),
        (x0 - 0.50 * w, y0 + 0.88 * h),
        (x0 - 0.24 * w, y1 + 0.92 * h),
    )
    left_lock = [
        (x0 + 0.08 * w, y0 + 0.14 * h),
        (x0 - 0.20 * w, y0 + 0.14 * h),
        *left_outer[1:],
        (x0 + 0.04 * w, y1 + 0.78 * h),
        (x0 + 0.12 * w, y0 + 0.24 * h),
    ]
    right_outer = _quadratic_points(
        (x1 + 0.20 * w, y0 + 0.14 * h),
        (x1 + 0.50 * w, y0 + 0.88 * h),
        (x1 + 0.24 * w, y1 + 0.92 * h),
    )
    right_lock = [
        (x1 - 0.08 * w, y0 + 0.14 * h),
        (x1 + 0.20 * w, y0 + 0.14 * h),
        *right_outer[1:],
        (x1 - 0.04 * w, y1 + 0.78 * h),
        (x1 - 0.12 * w, y0 + 0.24 * h),
    ]
    for lock in (left_lock, right_lock):
        draw.polygon(_scale_points(lock, scale), fill=_HAIR_RGB)
        if outline is not None:
            draw.line(_scale_points([*lock, lock[0]], scale), fill=outline, width=max(1, int(scale)))


def draw_person_hair_front(
    draw: ImageDraw.ImageDraw,
    *,
    head_bbox: BBox,
    gender_id: str,
    scale: int,
    outline: RGB | None,
) -> None:
    """Draw hair details that should sit on top of the head."""

    x0, y0, x1, y1 = [float(value) for value in head_bbox]
    w = x1 - x0
    h = y1 - y0
    if normalize_person_gender(gender_id) == "female":
        cap = (x0 - 0.14 * w, y0 - 0.34 * h, x1 + 0.14 * w, y0 + 0.42 * h)
        draw.ellipse(_scale_bbox(cap, scale), fill=_HAIR_RGB, outline=outline, width=max(1, int(scale)))
        fringe = [
            (x0 + 0.04 * w, y0 + 0.10 * h),
            (x0 + 0.44 * w, y0 - 0.12 * h),
            (x0 + 0.76 * w, y0 + 0.16 * h),
            (x0 + 0.34 * w, y0 + 0.34 * h),
        ]
        draw.polygon(_scale_points(fringe, scale), fill=_HAIR_RGB)
        if outline is not None:
            draw.line(_scale_points([*fringe, fringe[0]], scale), fill=outline, width=max(1, int(scale)))
        return
    cap = (x0 + 0.04 * w, y0 - 0.08 * h, x1 - 0.04 * w, y0 + 0.24 * h)
    draw.rounded_rectangle(_scale_bbox(cap, scale), radius=max(2, int(round(0.12 * h * int(scale)))), fill=_HAIR_RGB, outline=outline, width=max(1, int(scale)))
    for sideburn in ((x0 + 0.02 * w, y0 + 0.14 * h, x0 + 0.18 * w, y0 + 0.40 * h), (x1 - 0.18 * w, y0 + 0.14 * h, x1 - 0.02 * w, y0 + 0.40 * h)):
        draw.rounded_rectangle(_scale_bbox(sideburn, scale), radius=max(1, int(round(0.05 * h * int(scale)))), fill=_HAIR_RGB, outline=None)


def draw_person_skirt(
    draw: ImageDraw.ImageDraw,
    *,
    torso_bbox: BBox,
    bottom_y: float,
    fill: RGB,
    outline: RGB | None,
    scale: int,
    width: int,
) -> BBox:
    """Draw a simple restroom-icon style skirt below a torso and return its bbox."""

    x0, y0, x1, y1 = [float(value) for value in torso_bbox]
    w = x1 - x0
    h = y1 - y0
    top_y = y1 - 0.12 * h
    points = [
        (x0 + 0.08 * w, top_y),
        (x1 - 0.08 * w, top_y),
        (x1 + 0.34 * w, float(bottom_y)),
        (x0 - 0.34 * w, float(bottom_y)),
    ]
    draw.polygon(_scale_points(points, scale), fill=fill)
    if outline is not None:
        draw.line(_scale_points([*points, points[0]], scale), fill=outline, width=max(1, int(width)))
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )

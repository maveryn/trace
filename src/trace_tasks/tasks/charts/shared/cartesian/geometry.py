"""Pure geometry and projection helpers for cartesian chart scenes."""

from __future__ import annotations

from typing import Sequence


BBox = list[float]
Point = list[float]


def round_bbox(values: Sequence[float]) -> BBox:
    """Return a JSON-stable bbox-like float list."""

    return [round(float(value), 3) for value in values]


def round_point(x: float, y: float) -> Point:
    """Return a JSON-stable pixel point."""

    return [round(float(x), 3), round(float(y), 3)]


def union_bboxes(boxes: Sequence[Sequence[float]]) -> BBox:
    """Return the union of non-empty bbox lists."""

    valid = [tuple(float(value) for value in box[:4]) for box in boxes if len(box) >= 4]
    if not valid:
        return []
    return round_bbox(
        [
            min(box[0] for box in valid),
            min(box[1] for box in valid),
            max(box[2] for box in valid),
            max(box[3] for box in valid),
        ]
    )


def clip_bbox_to_container(bbox_values: Sequence[float], container: Sequence[float]) -> BBox:
    """Clip one bbox to a containing bbox."""

    x0, y0, x1, y1 = [float(value) for value in bbox_values[:4]]
    cx0, cy0, cx1, cy1 = [float(value) for value in container[:4]]
    return round_bbox([max(cx0, x0), max(cy0, y0), min(cx1, x1), min(cy1, y1)])


def project_linear(
    value: float,
    *,
    domain_min: float,
    domain_max: float,
    pixel_min: float,
    pixel_max: float,
    min_span: float = 1.0,
    clamp: bool = False,
) -> float:
    """Project one scalar from data space to pixel space."""

    span = max(float(min_span), float(domain_max) - float(domain_min))
    fraction = (float(value) - float(domain_min)) / float(span)
    if bool(clamp):
        fraction = max(0.0, min(1.0, float(fraction)))
    return float(pixel_min) + float(fraction) * (float(pixel_max) - float(pixel_min))


def project_linear_inverted(
    value: float,
    *,
    domain_min: float,
    domain_max: float,
    pixel_top: float,
    pixel_bottom: float,
    min_span: float = 1.0,
    clamp: bool = False,
) -> float:
    """Project one scalar to a y-axis pixel coordinate."""

    span = max(float(min_span), float(domain_max) - float(domain_min))
    fraction = (float(value) - float(domain_min)) / float(span)
    if bool(clamp):
        fraction = max(0.0, min(1.0, float(fraction)))
    return float(pixel_bottom) - float(fraction) * (float(pixel_bottom) - float(pixel_top))


def project_index(index: int, *, pixel_min: float, pixel_max: float, count: int) -> float:
    """Project one zero-based ordered index into evenly spaced pixel positions."""

    if int(count) <= 1:
        return 0.5 * float(float(pixel_min) + float(pixel_max))
    return float(pixel_min) + (float(index) / float(int(count) - 1)) * (float(pixel_max) - float(pixel_min))


def project_xy(
    *,
    x_value: float,
    y_value: float,
    plot_bbox: Sequence[float],
    x_min: float = 0.0,
    x_max: float = 100.0,
    y_min: float = 0.0,
    y_max: float = 100.0,
    min_span: float = 1.0,
    clamp: bool = False,
) -> tuple[float, float]:
    """Project one x/y data point into a cartesian plot bbox."""

    x0, y0, x1, y1 = [float(value) for value in plot_bbox[:4]]
    return (
        project_linear(
            float(x_value),
            domain_min=float(x_min),
            domain_max=float(x_max),
            pixel_min=float(x0),
            pixel_max=float(x1),
            min_span=float(min_span),
            clamp=bool(clamp),
        ),
        project_linear_inverted(
            float(y_value),
            domain_min=float(y_min),
            domain_max=float(y_max),
            pixel_top=float(y0),
            pixel_bottom=float(y1),
            min_span=float(min_span),
            clamp=bool(clamp),
        ),
    )


__all__ = [
    "BBox",
    "Point",
    "clip_bbox_to_container",
    "project_index",
    "project_linear",
    "project_linear_inverted",
    "project_xy",
    "round_bbox",
    "round_point",
    "union_bboxes",
]

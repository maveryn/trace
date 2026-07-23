"""Marker drawing primitives for cartesian chart renderers."""

from __future__ import annotations

import math
from typing import Literal, Sequence

from PIL import ImageDraw

from .geometry import round_bbox


RGB = Sequence[int]
TriangleStyle = Literal["up_wide", "down"]
RingStyle = Literal["outline", "inner_fill"]


def draw_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    shape: str,
    fill: RGB,
    outline: RGB,
    width: int = 2,
    marker_fill: str = "filled",
    triangle_style: TriangleStyle = "down",
    ring_style: RingStyle = "outline",
    ring_inner_scale: float = 0.58,
    cross_width: int | None = None,
    polygon_outline_width: int | None = None,
) -> list[float]:
    """Draw one common cartesian marker and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    marker_bbox = [cx - r, cy - r, cx + r, cy + r]
    resolved_outline = tuple(int(value) for value in outline)
    resolved_fill = tuple(int(value) for value in fill) if str(marker_fill) == "filled" else (255, 255, 255)
    line_width = max(1, int(width))
    shape_name = str(shape)

    if shape_name == "square":
        draw.rectangle(marker_bbox, fill=resolved_fill, outline=resolved_outline, width=line_width)
    elif shape_name == "diamond":
        points = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(points, fill=resolved_fill, outline=resolved_outline)
        if polygon_outline_width is not None:
            draw.line([*points, points[0]], fill=resolved_outline, width=max(1, int(polygon_outline_width)))
    elif shape_name == "triangle":
        if str(triangle_style) == "up_wide":
            points = [(cx, cy - r), (cx + r * 0.9, cy + r * 0.8), (cx - r * 0.9, cy + r * 0.8)]
        else:
            points = [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        draw.polygon(points, fill=resolved_fill, outline=resolved_outline)
        if polygon_outline_width is not None:
            draw.line([*points, points[0]], fill=resolved_outline, width=max(1, int(polygon_outline_width)))
    elif shape_name == "ring":
        draw.ellipse(marker_bbox, fill=(255, 255, 255), outline=resolved_outline, width=max(1, int(width)))
        if str(ring_style) == "inner_fill":
            inner_radius = float(r) * float(ring_inner_scale)
            inner = (
                cx - inner_radius,
                cy - inner_radius,
                cx + inner_radius,
                cy + inner_radius,
            )
            draw.ellipse(inner, fill=tuple(int(value) for value in fill), outline=tuple(int(value) for value in fill))
    elif shape_name == "cross":
        resolved_width = max(2, int(cross_width if cross_width is not None else width))
        draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=resolved_outline, width=resolved_width)
        draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=resolved_outline, width=resolved_width)
    elif shape_name == "pentagon":
        points = [
            (
                cx + math.cos((-90.0 + 72.0 * index) * math.pi / 180.0) * r,
                cy + math.sin((-90.0 + 72.0 * index) * math.pi / 180.0) * r,
            )
            for index in range(5)
        ]
        draw.polygon(points, fill=resolved_fill, outline=resolved_outline)
    else:
        draw.ellipse(marker_bbox, fill=resolved_fill, outline=resolved_outline, width=line_width)
    return round_bbox(marker_bbox)


__all__ = ["draw_marker"]

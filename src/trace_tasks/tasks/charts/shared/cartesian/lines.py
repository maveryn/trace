"""Line drawing primitives for cartesian chart renderers."""

from __future__ import annotations

import math
from typing import Sequence

from PIL import ImageDraw


RGB = Sequence[int]
Point = tuple[float, float]


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    *,
    fill: RGB,
    width: int,
    dash_px: float = 10.0,
    gap_px: float = 7.0,
) -> None:
    """Draw one deterministic dashed line between two points."""

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    length = math.hypot(float(x1 - x0), float(y1 - y0))
    if length <= 0.0:
        return
    dx = float(x1 - x0) / float(length)
    dy = float(y1 - y0) / float(length)
    position = 0.0
    dash = max(1.0, float(dash_px))
    gap = max(0.0, float(gap_px))
    while float(position) < float(length):
        segment_end = min(float(length), float(position + dash))
        draw.line(
            (
                float(x0 + dx * position),
                float(y0 + dy * position),
                float(x0 + dx * segment_end),
                float(y0 + dy * segment_end),
            ),
            fill=tuple(int(value) for value in fill),
            width=max(1, int(width)),
        )
        position += float(dash + gap)


def line_segments_for_style(style: str, *, width: int) -> tuple[tuple[float, float], ...]:
    """Return the dash pattern used by scientific/style line charts."""

    resolved = str(style)
    if resolved == "dotted":
        return ((0.0, max(5.0, float(width) * 4.5)),)
    if resolved == "dashdot":
        return (
            (max(12.0, float(width) * 7.0), max(5.0, float(width) * 3.5)),
            (0.0, max(5.0, float(width) * 3.5)),
        )
    if resolved == "long_dash":
        return ((max(20.0, float(width) * 11.0), max(8.0, float(width) * 5.0)),)
    if resolved == "short_dash":
        return ((max(8.0, float(width) * 4.5), max(5.0, float(width) * 3.0)),)
    return ((max(14.0, float(width) * 8.0), max(6.0, float(width) * 4.0)),)


def draw_styled_segment(
    draw: ImageDraw.ImageDraw,
    p0: Point,
    p1: Point,
    *,
    fill: RGB,
    width: int,
    style: str,
) -> None:
    """Draw one solid, dotted, dashed, or dash-dot segment."""

    if str(style) == "solid":
        draw.line([p0, p1], fill=tuple(int(value) for value in fill), width=max(1, int(width)))
        return
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return
    ux = dx / length
    uy = dy / length
    cursor = 0.0
    pattern = line_segments_for_style(str(style), width=max(1, int(width)))
    pattern_index = 0
    while cursor <= length:
        draw_len, gap_len = pattern[int(pattern_index) % len(pattern)]
        if draw_len <= 0.0:
            radius = max(1.0, float(width) * 0.8)
            cx = x0 + ux * cursor
            cy = y0 + uy * cursor
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=tuple(int(value) for value in fill))
            cursor += float(gap_len)
        else:
            end = min(length, cursor + draw_len)
            draw.line(
                [(x0 + ux * cursor, y0 + uy * cursor), (x0 + ux * end, y0 + uy * end)],
                fill=tuple(int(value) for value in fill),
                width=max(1, int(width)),
            )
            cursor = end + float(gap_len)
        pattern_index += 1


def draw_styled_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Point],
    *,
    fill: RGB,
    width: int,
    style: str,
) -> None:
    """Draw a polyline by applying the same style to each segment."""

    for p0, p1 in zip(points, points[1:]):
        draw_styled_segment(draw, p0, p1, fill=fill, width=int(width), style=str(style))


__all__ = [
    "draw_dashed_line",
    "draw_styled_polyline",
    "draw_styled_segment",
    "line_segments_for_style",
]

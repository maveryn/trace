"""Small deterministic drawing primitives reused across task families."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from PIL import ImageDraw

from .text_legibility import draw_traced_text
from .text_rendering import resolve_text_stroke_fill


def draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    *,
    radius: int,
    fill: Sequence[int],
    outline: Sequence[int],
    width: int,
) -> None:
    """Draw one rounded rectangle with deterministic styling."""

    draw.rounded_rectangle(
        bbox,
        radius=int(radius),
        fill=tuple(int(value) for value in fill),
        outline=tuple(int(value) for value in outline),
        width=int(width),
    )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    fill: Sequence[int],
    stroke_fill: Sequence[int],
    stroke_width: int = 1,
) -> List[float]:
    """Draw centered text and return the final text bbox."""

    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
    left, top, right, bottom = [float(value) for value in bbox]
    cx, cy = float(center[0]), float(center[1])
    tx = float(cx - (0.5 * (left + right)))
    ty = float(cy - (0.5 * (top + bottom)))
    draw_traced_text(
        draw,
        xy=(tx, ty),
        text=str(text),
        fill_rgb=tuple(int(v) for v in fill),
        font=font,
        stroke_width=max(0, int(stroke_width)),
        stroke_rgb=tuple(int(v) for v in stroke_fill),
        role="visible_text",
        required=False,
    )
    return [
        round(float(tx + left), 3),
        round(float(ty + top), 3),
        round(float(tx + right), 3),
        round(float(ty + bottom), 3),
    ]


def draw_centered_text_with_auto_stroke(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center_xy: Tuple[float, float],
    font,
    fill: Sequence[int],
    stroke_width_px: int,
) -> List[float]:
    """Draw centered text using the standard contrast stroke for the fill color."""

    return draw_centered_text(
        draw,
        text=str(text),
        center=(float(center_xy[0]), float(center_xy[1])),
        font=font,
        fill=fill,
        stroke_fill=resolve_text_stroke_fill(fill),
        stroke_width=max(0, int(stroke_width_px)),
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Sequence[int],
    width: int,
    dash_px: float,
    gap_px: float,
) -> None:
    """Draw one deterministic dashed line between two points."""

    start_x, start_y = float(start[0]), float(start[1])
    end_x, end_y = float(end[0]), float(end[1])
    dx = float(end_x - start_x)
    dy = float(end_y - start_y)
    length = math.hypot(float(dx), float(dy))
    if float(length) <= 1e-6:
        return
    dash_len = max(1.0, float(dash_px))
    gap_len = max(0.0, float(gap_px))
    unit_x = float(dx / length)
    unit_y = float(dy / length)
    cursor = 0.0
    while float(cursor) < float(length):
        dash_end = min(float(length), float(cursor + dash_len))
        seg_start = (
            float(start_x + (unit_x * cursor)),
            float(start_y + (unit_y * cursor)),
        )
        seg_end = (
            float(start_x + (unit_x * dash_end)),
            float(start_y + (unit_y * dash_end)),
        )
        draw.line(
            [seg_start, seg_end],
            fill=tuple(int(value) for value in fill),
            width=max(1, int(width)),
        )
        cursor = float(dash_end + gap_len)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Sequence[int],
    width: int,
    head_length_px: float,
    head_width_px: float,
) -> None:
    """Draw one simple arrow with a triangular head."""

    start_x, start_y = float(start[0]), float(start[1])
    end_x, end_y = float(end[0]), float(end[1])
    dx = float(end_x - start_x)
    dy = float(end_y - start_y)
    length = math.hypot(float(dx), float(dy))
    if float(length) <= 1e-6:
        return
    unit_x = float(dx / length)
    unit_y = float(dy / length)
    head_length = min(float(head_length_px), float(length) * 0.45)
    shaft_end = (
        float(end_x - (unit_x * head_length)),
        float(end_y - (unit_y * head_length)),
    )
    draw.line(
        [(start_x, start_y), shaft_end],
        fill=tuple(int(value) for value in fill),
        width=max(1, int(width)),
    )
    perp_x = float(-unit_y)
    perp_y = float(unit_x)
    half_head = float(0.5 * float(head_width_px))
    head_points = [
        (float(end_x), float(end_y)),
        (
            float(shaft_end[0] + (perp_x * half_head)),
            float(shaft_end[1] + (perp_y * half_head)),
        ),
        (
            float(shaft_end[0] - (perp_x * half_head)),
            float(shaft_end[1] - (perp_y * half_head)),
        ),
    ]
    draw.polygon(
        head_points,
        fill=tuple(int(value) for value in fill),
        outline=tuple(int(value) for value in fill),
    )


__all__ = [
    "draw_arrow",
    "draw_centered_text",
    "draw_centered_text_with_auto_stroke",
    "draw_dashed_line",
    "draw_rounded_rect",
]

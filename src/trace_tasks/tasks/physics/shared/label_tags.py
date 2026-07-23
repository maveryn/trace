"""Shared rounded text-label tag rendering for physics diagrams."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from PIL import ImageDraw

from ...shared.bbox_projection import bbox_union_many
from ...shared.drawing import draw_centered_text
from ...shared.text_rendering import resolve_text_stroke_fill

RGB = Tuple[int, int, int]
BBox = List[float]


def _bbox(values: Sequence[float]) -> BBox:
    return [round(float(value), 3) for value in values]


def text_tag_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Sequence[float],
    font: object,
    stroke_width_px: int = 0,
    pad_x_px: float = 9.0,
    pad_y_px: float = 6.0,
) -> BBox:
    """Return the rounded-tag outer bbox for centered text before drawing."""

    cx, cy = float(center[0]), float(center[1])
    text_bbox_value = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width_px)))
    text_width = float(text_bbox_value[2] - text_bbox_value[0])
    text_height = float(text_bbox_value[3] - text_bbox_value[1])
    return _bbox(
        (
            cx - text_width * 0.5 - float(pad_x_px),
            cy - text_height * 0.5 - float(pad_y_px),
            cx + text_width * 0.5 + float(pad_x_px),
            cy + text_height * 0.5 + float(pad_y_px),
        )
    )


def draw_text_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Sequence[float],
    font: object,
    fill_rgb: RGB,
    outline_rgb: RGB,
    text_rgb: RGB,
    stroke_width_px: int,
    pad_x_px: float = 9.0,
    pad_y_px: float = 6.0,
    radius_px: float = 7.0,
    outline_width_px: int = 2,
    text_stroke_width_px: int | None = None,
) -> BBox:
    """Draw a rounded text tag and return the union of backing and glyph bboxes."""

    cx, cy = float(center[0]), float(center[1])
    tag_bbox = text_tag_bbox(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        stroke_width_px=0,
        pad_x_px=float(pad_x_px),
        pad_y_px=float(pad_y_px),
    )
    draw.rounded_rectangle(
        tuple(float(value) for value in tag_bbox),
        radius=float(radius_px),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(1, int(outline_width_px)),
    )
    resolved_text_stroke_width = max(0, int(stroke_width_px) - 1) if text_stroke_width_px is None else max(0, int(text_stroke_width_px))
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=resolve_text_stroke_fill(tuple(int(value) for value in text_rgb)),
        stroke_width=resolved_text_stroke_width,
    )
    return _bbox(bbox_union_many(tag_bbox, text_draw_bbox))


__all__ = ["draw_text_tag", "text_tag_bbox"]

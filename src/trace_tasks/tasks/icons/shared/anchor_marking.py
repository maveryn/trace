"""Reusable anchor-label drawing helpers for icon relation scenes."""

from __future__ import annotations

from typing import Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.text_legibility import draw_centered_traced_text
from ...shared.text_rendering import load_font


BBox = Tuple[int, int, int, int]


def expand_bbox(box: Sequence[int | float], pad_px: int) -> BBox:
    """Expand one `xyxy` box by a symmetric pixel padding."""

    return (
        int(round(float(box[0]) - float(pad_px))),
        int(round(float(box[1]) - float(pad_px))),
        int(round(float(box[2]) + float(pad_px))),
        int(round(float(box[3]) + float(pad_px))),
    )


def draw_anchor_marker(
    *,
    image: Image.Image,
    anchor_bbox: Sequence[int | float],
    content_bbox: Sequence[int | float],
    highlight_padding_px: int,
    highlight_radius_px: int,
    outline_rgb: Tuple[int, int, int],
    label_color_rgb: Tuple[int, int, int],
    panel_fill_rgb: Tuple[int, int, int],
    label_font_size_px: int,
    label_text: str,
    label_role: str = "icon_anchor_label_text",
) -> BBox:
    """Draw one highlighted anchor box + label and return the expanded highlight bbox."""

    anchor_box = tuple(int(round(float(value))) for value in anchor_bbox)
    highlight_box = expand_bbox(anchor_box, int(highlight_padding_px))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        highlight_box,
        radius=max(0, int(highlight_radius_px)),
        outline=tuple(int(v) for v in outline_rgb),
        width=3,
    )
    font_size = int(label_font_size_px)
    label_x = 0.5 * float(anchor_box[0] + anchor_box[2])
    label_y = float(anchor_box[1]) - max(14.0, float(font_size) * 0.75)
    min_y = float(content_bbox[1]) + (0.75 * float(font_size))
    if label_y < min_y:
        label_y = float(anchor_box[3]) + max(14.0, float(font_size) * 0.75)
    draw_centered_traced_text(
        draw,
        text=str(label_text),
        center=(float(label_x), float(label_y)),
        font=load_font(font_size, bold=True),
        fill_rgb=tuple(int(v) for v in label_color_rgb),
        stroke_rgb=tuple(int(v) for v in panel_fill_rgb),
        stroke_width=2,
        role=str(label_role),
        required=False,
    )
    return tuple(int(value) for value in highlight_box)


__all__ = ["draw_anchor_marker", "expand_bbox"]

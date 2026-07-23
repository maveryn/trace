"""Reusable labeled option-panel chrome for puzzle tasks with image choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from PIL import ImageDraw

from .drawing import draw_centered_text, draw_rounded_rect


@dataclass(frozen=True)
class RenderedPuzzleOptionPanel:
    """Geometry returned after drawing one labeled option panel."""

    panel_bbox: list[float]
    label_bbox: list[float]
    content_bbox: list[float]


def render_puzzle_option_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Tuple[float, float, float, float],
    option_label: str,
    label_font,
    label_center_y_px: float,
    content_box_size_px: float,
    content_gap_px: float,
    panel_fill_rgb: Sequence[int],
    content_fill_rgb: Sequence[int],
    border_color_rgb: Sequence[int],
    text_color_rgb: Sequence[int],
    text_stroke_rgb: Sequence[int],
    panel_corner_radius_px: int,
    content_corner_radius_px: int,
    border_width_px: int,
) -> RenderedPuzzleOptionPanel:
    """Draw one option panel and return its traced panel/label/content geometry."""

    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(panel_corner_radius_px),
        fill=panel_fill_rgb,
        outline=border_color_rgb,
        width=int(border_width_px),
    )

    panel_left, panel_top, panel_right, _ = [float(value) for value in panel_bbox]
    panel_width = float(panel_right - panel_left)
    label_bbox = draw_centered_text(
        draw,
        text=str(option_label),
        center=(float(panel_left + 0.5 * panel_width), float(label_center_y_px)),
        font=label_font,
        fill=text_color_rgb,
        stroke_fill=text_stroke_rgb,
        stroke_width=1,
    )

    content_left = float(panel_left + 0.5 * (panel_width - float(content_box_size_px)))
    content_top = float(label_bbox[3] + float(content_gap_px))
    content_bbox = (
        float(content_left),
        float(content_top),
        float(content_left + float(content_box_size_px)),
        float(content_top + float(content_box_size_px)),
    )
    draw_rounded_rect(
        draw,
        content_bbox,
        radius=int(content_corner_radius_px),
        fill=content_fill_rgb,
        outline=border_color_rgb,
        width=int(border_width_px),
    )
    return RenderedPuzzleOptionPanel(
        panel_bbox=[round(float(value), 3) for value in panel_bbox],
        label_bbox=list(label_bbox),
        content_bbox=[round(float(value), 3) for value in content_bbox],
    )


__all__ = [
    "RenderedPuzzleOptionPanel",
    "render_puzzle_option_panel",
]

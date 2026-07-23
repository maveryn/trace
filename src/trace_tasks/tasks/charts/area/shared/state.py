"""State containers for the area chart scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class AreaRenderParams:
    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    tick_length_px: int
    label_font_size_px: int
    tick_font_size_px: int
    value_font_size_px: int
    legend_font_size_px: int
    label_stroke_width_px: int
    area_outline_width_px: int
    point_radius_px: int
    axis_color_rgb: tuple[int, int, int]
    grid_color_rgb: tuple[int, int, int]
    plot_fill_rgb: tuple[int, int, int]
    text_color_rgb: tuple[int, int, int]
    text_stroke_rgb: tuple[int, int, int]
    legend_border_rgb: tuple[int, int, int]
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedAreaPanel:
    image: Image.Image
    plot_bbox_px: tuple[int, int, int, int]
    y_axis_max: int
    y_ticks: tuple[int, ...]
    entities: tuple[dict[str, Any], ...]
    point_traces: tuple[dict[str, Any], ...]
    legend_traces: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AreaRenderResult:
    image: Image.Image
    panel: RenderedAreaPanel
    render_params: AreaRenderParams
    information_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str


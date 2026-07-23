"""Candlestick scene state and render dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


RGB = tuple[int, int, int]
BBox = list[float]
Point = list[float]


@dataclass(frozen=True)
class Candle:
    candle_id: str
    label: str
    open_value: int
    high_value: int
    low_value: int
    close_value: int

    @property
    def direction(self) -> str:
        return "up" if int(self.close_value) > int(self.open_value) else "down"

    @property
    def body_size(self) -> int:
        return abs(int(self.close_value) - int(self.open_value))

    @property
    def wick_range(self) -> int:
        return int(self.high_value) - int(self.low_value)


@dataclass(frozen=True)
class Selection:
    answer: int | str
    answer_type: str
    annotation_candle_ids: tuple[str, ...]
    annotation_label_ids: tuple[str, ...]
    annotation_roles: tuple[str, ...]
    trace: dict[str, Any]


@dataclass(frozen=True)
class Dataset:
    candles: tuple[Candle, ...]
    selection: Selection


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    wick_line_width_px: int
    body_outline_width_px: int
    tick_length_px: int
    title_font_size_px: int
    tick_font_size_px: int
    label_font_size_px: int
    value_font_size_px: int
    candle_width_fraction: float
    y_axis_min: int
    y_axis_max: int
    y_tick_step: int
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    plot_fill_rgb: RGB
    text_color_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    up_fill_rgb: RGB
    down_fill_rgb: RGB
    wick_rgb: RGB
    body_outline_rgb: RGB
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class Rendered:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    candle_bboxes_px: dict[str, BBox]
    body_bboxes_px: dict[str, BBox]
    wick_bboxes_px: dict[str, BBox]
    value_label_bboxes_px: dict[str, BBox]
    x_label_bboxes_px: dict[str, BBox]


@dataclass(frozen=True)
class RenderArtifacts:
    rendered: Rendered
    render_params: RenderParams
    background_style: dict[str, Any]
    font_assets: dict[str, Any]
    post_image_noise: dict[str, Any]


__all__ = [
    "BBox",
    "Candle",
    "Dataset",
    "Point",
    "RGB",
    "RenderArtifacts",
    "RenderParams",
    "Rendered",
    "Selection",
]

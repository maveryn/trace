"""State containers for waterfall chart scene primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image


RGB = tuple[int, int, int]
BBox = list[float]


@dataclass(frozen=True)
class WaterfallStep:
    """One signed contribution in a waterfall chart."""

    step_id: str
    label: str
    delta: int
    running_before: int
    running_after: int


@dataclass(frozen=True)
class WaterfallDataset:
    """Sampled start value plus ordered waterfall contribution steps."""

    start_value: int
    final_value: int
    steps: tuple[WaterfallStep, ...]


@dataclass(frozen=True)
class WaterfallRenderParams:
    """Resolved visual parameters for one waterfall render."""

    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    connector_width_px: int
    bar_outline_width_px: int
    tick_length_px: int
    title_font_size_px: int
    tick_font_size_px: int
    label_font_size_px: int
    value_font_size_px: int
    threshold_font_size_px: int
    bar_width_fraction: float
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    plot_fill_rgb: RGB
    text_color_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    start_fill_rgb: RGB
    final_fill_rgb: RGB
    positive_fill_rgb: RGB
    negative_fill_rgb: RGB
    connector_rgb: RGB
    threshold_rgb: RGB
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedWaterfall:
    """Rendered waterfall chart with projected primitive boxes."""

    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    bar_bboxes_px: dict[str, BBox]
    value_label_bboxes_px: dict[str, BBox]
    x_label_bboxes_px: dict[str, BBox]
    connector_bboxes_px: dict[str, BBox]
    extra_bboxes_px: dict[str, BBox]
    threshold_value: int | None
    y_axis_max: int


@dataclass(frozen=True)
class WaterfallRenderArtifacts:
    """Final image and render metadata after background/noise processing."""

    image: Image.Image
    rendered_scene: RenderedWaterfall
    render_params: WaterfallRenderParams
    background_style: Mapping[str, Any]
    post_image_noise: Mapping[str, Any]
    chart_font_family: str


__all__ = [
    "BBox",
    "RGB",
    "RenderedWaterfall",
    "WaterfallDataset",
    "WaterfallRenderArtifacts",
    "WaterfallRenderParams",
    "WaterfallStep",
]

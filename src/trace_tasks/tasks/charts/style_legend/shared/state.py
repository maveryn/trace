"""State containers for the style-legend chart scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.tasks.charts.shared.cartesian.geometry import round_bbox, round_point


DOMAIN = "charts"
SCENE_ID = "style_legend"
SCENE_NAMESPACE = "charts.style_legend"

RGB = tuple[int, int, int]
Point = list[float]
BBox = list[float]

LINE_STYLES: tuple[str, ...] = ("solid", "dashed", "dotted", "dashdot", "long_dash", "short_dash")
MARKER_SHAPES: tuple[str, ...] = ("circle", "square", "diamond", "triangle", "ring", "cross")
MARKER_FILLS: tuple[str, ...] = ("filled", "open")
SUPPORTED_STYLE_PALETTE_MODES: tuple[str, ...] = ("grayscale", "muted_color", "colorblind_safe")
SUPPORTED_LEGEND_POSITIONS: tuple[str, ...] = ("right", "inside_top_right", "top")


@dataclass(frozen=True)
class SeriesStyle:
    color_rgb: RGB
    line_style: str
    marker_shape: str
    marker_fill: str
    line_width_px: int


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    label: str
    values: tuple[int, ...]
    style: SeriesStyle


@dataclass(frozen=True)
class StyleLegendDataset:
    x_labels: tuple[str, ...]
    x_label_meta: dict[str, Any]
    series: tuple[SeriesSpec, ...]
    series_label_meta: dict[str, Any]
    target_x_index: int
    threshold_value: int | None
    pair_series_ids: tuple[str, str]
    palette_mode: str
    palette_mode_probabilities: dict[str, float]
    legend_position: str
    legend_position_probabilities: dict[str, float]


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    margin_left_px: int
    margin_right_px: int
    margin_top_px: int
    margin_bottom_px: int
    title_font_size_px: int
    tick_font_size_px: int
    label_font_size_px: int
    legend_font_size_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    point_radius_px: int
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    panel_fill_rgb: RGB
    panel_outline_rgb: RGB
    threshold_rgb: RGB
    font_family: str
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedStyleLegend:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    legend_bbox_px: BBox
    legend_item_bboxes_px: dict[str, BBox]
    point_map_px: dict[str, dict[str, Point]]
    point_bboxes_px: dict[str, BBox]
    threshold_bbox_px: BBox | None
    render_meta: dict[str, Any]


def point_id(series_id: str, x_index: int) -> str:
    return f"{str(series_id)}|x{int(x_index)}"


def bbox(values: tuple[float, float, float, float] | list[float]) -> BBox:
    return round_bbox(values)


def point(x: float, y: float) -> Point:
    return round_point(float(x), float(y))

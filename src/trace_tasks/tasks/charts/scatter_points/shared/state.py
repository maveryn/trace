"""State records for scatter-point chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image

from trace_tasks.tasks.charts.shared.label_assets import ResolvedChartLabels


DOMAIN = "charts"
SCENE_ID = "scatter_points"
SCENE_NAMESPACE = "charts.scatter_points"

SUPPORTED_THRESHOLD_AXES: Tuple[str, ...] = ("x", "y")
SUPPORTED_THRESHOLD_DIRECTIONS: Tuple[str, ...] = ("above", "below")
SUPPORTED_MEAN_AXES: Tuple[str, ...] = ("x", "y")
SUPPORTED_MEAN_EXTREMA: Tuple[str, ...] = ("largest", "smallest")
MARKER_SHAPES: Tuple[str, ...] = ("circle", "square", "diamond", "triangle", "ring", "pentagon")

RGB = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Point:
    point_id: str
    x_value: float
    y_value: float
    category_label: str
    color_rgb: RGB
    marker_shape: str


@dataclass(frozen=True)
class Category:
    label: str
    color_rgb: RGB
    marker_shape: str
    point_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Query:
    answer: int | str
    answer_type: str
    annotation_point_ids: Tuple[str, ...]
    trace: Dict[str, Any]


@dataclass(frozen=True)
class Dataset:
    scene_variant: str
    points: Tuple[Point, ...]
    categories: Tuple[Category, ...]
    query: Query
    label_resolution: ResolvedChartLabels | None = None


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
    tick_length_px: int
    point_radius_px: int
    tick_font_size_px: int
    label_font_size_px: int
    legend_font_size_px: int
    title_font_size_px: int
    legend_gap_px: int
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    text_color_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    plot_fill_rgb: RGB
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    threshold_line_rgb: RGB
    threshold_label_rgb: RGB
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedScene:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: list[float]
    panel_bbox_px: list[float]
    point_bboxes: dict[str, list[float]]
    point_centers: dict[str, list[float]]
    legend_bboxes: dict[str, list[float]]
    threshold_guide_bbox_px: list[float]
    title_bbox_px: list[float]
    x_axis_label_bbox_px: list[float]
    y_axis_label_bbox_px: list[float]
    title_text: str


@dataclass(frozen=True)
class ScatterPointsRenderResult:
    image: Image.Image
    rendered_scene: RenderedScene
    render_params: RenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str


__all__ = [
    "BBox",
    "Category",
    "DOMAIN",
    "Dataset",
    "MARKER_SHAPES",
    "Point",
    "Query",
    "RGB",
    "RenderParams",
    "RenderedScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_MEAN_AXES",
    "SUPPORTED_MEAN_EXTREMA",
    "SUPPORTED_THRESHOLD_AXES",
    "SUPPORTED_THRESHOLD_DIRECTIONS",
    "ScatterPointsRenderResult",
]

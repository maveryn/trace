"""State records for scatter-readout chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "scatter_readout"
SCENE_NAMESPACE = "charts.scatter_readout"
SCENE_VARIANT = "marker_scatter"

MONTH_LABELS: Tuple[str, ...] = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct")
MARKER_SHAPES: Tuple[str, ...] = ("circle", "square", "diamond", "triangle", "ring")

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Point:
    point_id: str
    series_label: str
    x_label: str
    x_index: int
    y_value: int


@dataclass(frozen=True)
class Series:
    label: str
    color_rgb: RGB
    marker_shape: str
    points: Tuple[Point, ...]


@dataclass(frozen=True)
class SceneDataset:
    scene_variant: str
    x_axis_title: str
    y_axis_title: str
    x_labels: Tuple[str, ...]
    series: Tuple[Series, ...]


@dataclass(frozen=True)
class QueryBinding:
    answer: int | str
    answer_type: str
    target_series_label: str
    target_point_id: str
    annotation_point_ids: Tuple[str, ...]
    annotation_x_label: str
    trace: Dict[str, Any]


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
    value_font_size_px: int
    legend_font_size_px: int
    title_font_size_px: int
    legend_gap_px: int
    show_title: bool
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    text_color_rgb: RGB
    text_stroke_rgb: RGB
    plot_fill_rgb: RGB
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedScene:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: list[float]
    point_bboxes: dict[str, list[float]]
    value_label_bboxes: dict[str, list[float]]
    point_annotation_bboxes: dict[str, list[float]]
    x_label_bboxes: dict[str, list[float]]
    legend_bboxes: dict[str, list[float]]
    title_bbox_px: list[float]
    title_text: str


@dataclass(frozen=True)
class ScatterReadoutRenderResult:
    image: Image.Image
    rendered_scene: RenderedScene
    render_params: RenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str


__all__ = [
    "DOMAIN",
    "MARKER_SHAPES",
    "MONTH_LABELS",
    "Point",
    "QueryBinding",
    "RGB",
    "RenderParams",
    "RenderedScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "ScatterReadoutRenderResult",
    "SceneDataset",
    "Series",
]

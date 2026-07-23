"""State records for error-bar series chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "errorbar_series"
SCENE_NAMESPACE = "charts_errorbar_series_base"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "marker_errorbar",
    "line_marker_errorbar",
    "grouped_errorbar",
)

RGB = Tuple[int, int, int]
BBox = List[float]
Point = List[float]


@dataclass(frozen=True)
class ErrorbarSeries:
    series_id: str
    label: str
    color_rgb: RGB
    lower_values: Tuple[int, ...]
    mid_values: Tuple[int, ...]
    upper_values: Tuple[int, ...]


@dataclass(frozen=True)
class ErrorbarQuery:
    prompt_key: str
    answer: int | str
    answer_type: str
    annotation_kind: str
    annotation_item_keys: Tuple[str, ...]
    params: Dict[str, Any]


@dataclass(frozen=True)
class ErrorbarDataset:
    x_labels: Tuple[str, ...]
    x_label_meta: Dict[str, Any]
    series: Tuple[ErrorbarSeries, ...]
    series_label_meta: Dict[str, Any]
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    prompt_key: str
    prompt_key_probabilities: Dict[str, float]
    threshold_value: int | None
    target_series_id: str
    target_x_index: int | None
    title: str
    query: ErrorbarQuery


@dataclass(frozen=True)
class ErrorbarRenderParams:
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
    axis_min: int
    axis_max: int
    tick_step: int
    axis_line_width_px: int
    grid_line_width_px: int
    errorbar_line_width_px: int
    center_line_width_px: int
    cap_width_px: int
    point_radius_px: int
    series_offset_px: int
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    panel_fill_rgb: RGB
    panel_outline_rgb: RGB
    threshold_rgb: RGB
    font_family: str
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class ErrorbarRendered:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: BBox
    errorbar_bboxes_px: Dict[str, BBox]
    point_map_px: Dict[str, Dict[str, Dict[str, Point]]]
    threshold_bbox_px: BBox | None
    render_meta: Dict[str, Any]


__all__ = [
    "BBox",
    "DOMAIN",
    "ErrorbarDataset",
    "ErrorbarQuery",
    "ErrorbarRenderParams",
    "ErrorbarRendered",
    "ErrorbarSeries",
    "Point",
    "RGB",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
]

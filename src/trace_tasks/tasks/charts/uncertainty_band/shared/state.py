"""State objects for uncertainty-band chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "uncertainty_band"
SCENE_NAMESPACE = "charts.uncertainty_band"
PROMPT_BUNDLE_ID = "charts_uncertainty_band_v1"

RGB = Tuple[int, int, int]
Point = List[float]


@dataclass(frozen=True)
class BandSeries:
    series_id: str
    label: str
    color_rgb: RGB
    lower_values: Tuple[int, ...]
    mid_values: Tuple[int, ...]
    upper_values: Tuple[int, ...]


@dataclass(frozen=True)
class Dataset:
    x_labels: Tuple[str, ...]
    x_label_meta: Dict[str, Any]
    series: Tuple[BandSeries, BandSeries]
    series_label_meta: Dict[str, Any]
    title: str


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    margin_left_px: int
    margin_right_px: int
    margin_top_px: int
    margin_bottom_px: int
    title_band_height_px: int
    legend_width_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    band_outline_width_px: int
    center_line_width_px: int
    point_radius_px: int
    title_font_size_px: int
    label_font_size_px: int
    tick_font_size_px: int
    legend_font_size_px: int
    axis_min: int
    axis_max: int
    tick_step: int
    band_alpha: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    font_family: str
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class Rendered:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: List[float]
    series_band_bboxes_px: Dict[str, List[float]]
    point_map_px: Dict[str, Dict[str, Dict[str, Point]]]
    overlap_points_px: Dict[str, Point]
    render_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderArtifacts:
    rendered: Rendered
    post_image_noise: Dict[str, Any]


__all__ = [
    "BandSeries",
    "DOMAIN",
    "Dataset",
    "PROMPT_BUNDLE_ID",
    "Point",
    "RGB",
    "RenderArtifacts",
    "RenderParams",
    "Rendered",
    "SCENE_ID",
    "SCENE_NAMESPACE",
]

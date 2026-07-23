"""Passive state containers for the 3D bar-grid chart scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "bar_3d"
SCENE_NAMESPACE = "charts.bar_3d"

RGB = Tuple[int, int, int]
BBox = List[float]


@dataclass(frozen=True)
class _BarCell:
    bar_id: str
    x_label: str
    series_label: str
    x_index: int
    series_index: int
    value: int
    color_rgb: RGB


@dataclass(frozen=True)
class _Selection:
    answer: int
    annotation_bar_ids: Tuple[str, ...]
    trace: Dict[str, Any]
    annotation_kind: str = "point_set"
    annotation_bar_id_pairs: Tuple[Tuple[str, str], ...] = ()
    annotation_bar_id_groups: Dict[str, Tuple[str, ...]] | None = None


@dataclass(frozen=True)
class _Dataset:
    x_labels: Tuple[str, ...]
    series_labels: Tuple[str, ...]
    bars: Tuple[_BarCell, ...]
    selection: _Selection


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    bar_edge_width_px: int
    tick_length_px: int
    tick_font_size_px: int
    label_font_size_px: int
    value_font_size_px: int
    legend_font_size_px: int
    label_stroke_width_px: int
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    plot_fill_rgb: RGB
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    text_color_rgb: RGB
    text_stroke_rgb: RGB
    legend_border_rgb: RGB
    bar_edge_rgb: RGB
    depth_axis_dx_px: int
    depth_axis_dy_px: int
    bar_face_dx_px: int
    bar_face_dy_px: int
    bar_width_px: int
    bar_style_variant: str
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class _RenderedBarGrid:
    image: Image.Image
    plot_bbox_px: BBox
    y_axis_max: int
    y_ticks: Tuple[int, ...]
    entities: Tuple[Dict[str, Any], ...]
    bar_traces: Tuple[Dict[str, Any], ...]
    legend_traces: Tuple[Dict[str, Any], ...]
    layout_jitter_meta: Dict[str, Any]
    bar_style_meta: Dict[str, Any]


@dataclass(frozen=True)
class BarGridRenderArtifacts:
    rendered: _RenderedBarGrid
    background_style: Dict[str, Any]
    font_assets: Dict[str, Any]
    post_image_noise: Dict[str, Any]

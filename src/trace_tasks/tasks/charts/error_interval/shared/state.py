"""Scene state dataclasses for error-interval charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


RGB = Tuple[int, int, int]
BBox = List[float]
Point = List[float]
Segment = List[Point]


@dataclass(frozen=True)
class _IntervalItem:
    item_id: str
    label: str
    lower: int
    midpoint: int
    upper: int
    color_rgb: RGB


@dataclass(frozen=True)
class _Query:
    prompt_key: str
    answer: int | str
    answer_type: str
    annotation_type: str
    annotation_item_ids: Tuple[str, ...]
    params: Dict[str, Any]


@dataclass(frozen=True)
class _Dataset:
    items: Tuple[_IntervalItem, ...]
    prompt_key: str
    query_probabilities: Dict[str, float]
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    reference_value: int | None
    title: str
    query: _Query


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    outer_margin_left_px: int
    outer_margin_right_px: int
    outer_margin_top_px: int
    outer_margin_bottom_px: int
    title_band_height_px: int
    label_band_px: int
    plot_padding_px: int
    panel_corner_radius_px: int
    panel_outline_width_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    interval_line_width_px: int
    cap_length_px: int
    point_radius_px: int
    bar_width_fraction: float
    title_font_size_px: int
    label_font_size_px: int
    tick_font_size_px: int
    value_font_size_px: int
    axis_min: int
    axis_max: int
    tick_step: int
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    panel_fill_rgb: RGB
    panel_outline_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    reference_rgb: RGB
    interval_outline_rgb: RGB
    font_family: str
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class _Rendered:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: BBox
    item_bboxes_px: Dict[str, BBox]
    interval_bboxes_px: Dict[str, BBox]
    interval_center_points_px: Dict[str, List[float]]
    interval_segments_px: Dict[str, Segment]
    render_meta: Dict[str, Any]


__all__ = [
    "BBox",
    "Point",
    "RGB",
    "Segment",
    "_Dataset",
    "_IntervalItem",
    "_Query",
    "_Rendered",
    "_RenderParams",
]

"""State dataclasses and scene constants for dashboard chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "dashboard"
SCENE_NAMESPACE = "charts.dashboard"
SCENE_VARIANT = "mixed_dashboard"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (SCENE_VARIANT,)
SUPPORTED_RANK_DIRECTIONS: Tuple[str, ...] = ("largest", "smallest")
SUPPORTED_CONDITION_COMPARISONS: Tuple[str, ...] = ("greater_than", "less_than")
SUPPORTED_REQUESTED_TRUTHS: Tuple[str, ...] = ("true", "false")
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SUPPORTED_PANEL_KINDS: Tuple[str, ...] = ("bar", "line", "donut", "radar")
PANEL_KIND_NAMES: Dict[str, str] = {
    "bar": "Bars",
    "line": "Line",
    "donut": "Donut",
    "radar": "Radar",
}

RGB = Tuple[int, int, int]
BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]
AnnotationRef = Tuple[str, str]


@dataclass(frozen=True)
class Category:
    category_id: str
    label: str
    color_rgb: RGB


@dataclass(frozen=True)
class Panel:
    panel_id: str
    kind: str
    name: str
    values_by_category_id: Dict[str, int]


@dataclass(frozen=True)
class DashboardQuery:
    answer: int | str
    answer_type: str
    annotation_refs: Tuple[AnnotationRef, ...]
    params: Dict[str, Any]


@dataclass(frozen=True)
class DashboardDataset:
    scene_variant: str
    categories: Tuple[Category, ...]
    panels: Tuple[Panel, ...]
    query: DashboardQuery


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    panel_gap_px: int
    dashboard_margin_px: int
    title_height_px: int
    panel_padding_px: int
    panel_border_width_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    bar_min_height_px: int
    point_radius_px: int
    line_width_px: int
    title_font_size_px: int
    panel_title_font_size_px: int
    label_font_size_px: int
    value_font_size_px: int
    tick_font_size_px: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    text_color_rgb: RGB
    muted_text_color_rgb: RGB
    connector_color_rgb: RGB
    donut_hole_fill_rgb: RGB
    category_palette_rgb: Tuple[RGB, ...]
    font_family: str
    layout_offset_x_px: int
    layout_offset_y_px: int
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedDashboard:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    panel_bboxes_px: Dict[str, BBox]
    support_bboxes_px: Dict[str, Dict[str, BBox]]
    support_points_px: Dict[str, Dict[str, Point]]
    value_label_bboxes_px: Dict[str, Dict[str, BBox]]
    option_statement_bboxes_px: Dict[str, BBox]
    context_text_elements: Tuple[Dict[str, Any], ...]
    context_text_layout: Dict[str, Any]


@dataclass(frozen=True)
class DashboardBaseSample:
    categories: Tuple[Category, ...]
    panels: Tuple[Panel, ...]
    common_params: Dict[str, Any]


__all__ = [
    "AnnotationRef",
    "BBox",
    "Category",
    "DOMAIN",
    "DashboardBaseSample",
    "DashboardDataset",
    "DashboardQuery",
    "OPTION_LETTERS",
    "PANEL_KIND_NAMES",
    "Panel",
    "Point",
    "RGB",
    "RenderParams",
    "RenderedDashboard",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "SUPPORTED_CONDITION_COMPARISONS",
    "SUPPORTED_PANEL_KINDS",
    "SUPPORTED_RANK_DIRECTIONS",
    "SUPPORTED_REQUESTED_TRUTHS",
    "SUPPORTED_SCENE_VARIANTS",
]

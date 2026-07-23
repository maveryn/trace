"""State containers for the sunburst chart scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "sunburst"
SCENE_KIND = "chart_sunburst_hierarchy"
SCENE_VARIANT = "sunburst_hierarchy"
SCENE_NAMESPACE = "charts.sunburst"

RGB = tuple[int, int, int]
BBox = list[float]
Point = list[float]


@dataclass(frozen=True)
class SunburstNode:
    node_id: str
    label: str
    level: str
    parent_id: str | None
    value: int
    child_ids: tuple[str, ...]
    color_rgb: RGB


@dataclass(frozen=True)
class SunburstTree:
    nodes: tuple[SunburstNode, ...]
    root_id: str
    parent_ids: tuple[str, ...]
    subgroup_ids: tuple[str, ...]
    leaf_ids: tuple[str, ...]
    generation_ranges: dict[str, Any]


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    center_x_px: int
    center_y_px: int
    inner_radius_px: int
    parent_outer_radius_px: int
    subgroup_outer_radius_px: int
    leaf_outer_radius_px: int
    parent_font_size_px: int
    subgroup_font_size_px: int
    leaf_font_size_px: int
    value_font_size_px: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    plot_fill_rgb: RGB
    text_color_rgb: RGB
    separator_rgb: RGB
    text_stroke_rgb: RGB
    label_stroke_width_px: int


@dataclass(frozen=True)
class RenderedSunburst:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    node_traces: tuple[dict[str, Any], ...]
    leaf_value_bbox_by_node_id: dict[str, BBox]
    chart_bbox_px: BBox
    render_meta: dict[str, Any]
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_assets: dict[str, Any]


__all__ = [
    "BBox",
    "DOMAIN",
    "Point",
    "RGB",
    "RenderedSunburst",
    "RenderParams",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "SunburstNode",
    "SunburstTree",
]

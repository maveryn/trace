"""State containers for treemap chart scene primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "treemap"
SCENE_NAMESPACE = "charts.treemap"
PROMPT_BUNDLE_ID = "charts_treemap_v1"

RGB = tuple[int, int, int]
BBox = list[float]


@dataclass(frozen=True)
class TreemapLeaf:
    leaf_id: str
    parent_id: str
    parent_label: str
    label: str
    value: int
    color_rgb: RGB


@dataclass(frozen=True)
class TreemapParent:
    parent_id: str
    label: str
    leaf_ids: tuple[str, ...]
    value: int
    color_rgb: RGB


@dataclass(frozen=True)
class TreemapDataset:
    title: str
    parent_axis: str
    leaf_axis: str
    parents: tuple[TreemapParent, ...]
    leaves: tuple[TreemapLeaf, ...]
    generation_ranges: dict[str, Any]


@dataclass(frozen=True)
class TreemapRenderParams:
    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    title_font_size_px: int
    parent_font_size_px: int
    leaf_font_size_px: int
    value_font_size_px: int
    note_font_size_px: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    text_color_rgb: RGB
    muted_text_rgb: RGB
    separator_rgb: RGB
    text_stroke_rgb: RGB
    label_stroke_width_px: int


@dataclass(frozen=True)
class RenderedTreemap:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    leaf_traces: tuple[dict[str, Any], ...]
    parent_traces: tuple[dict[str, Any], ...]
    annotation_bbox_by_leaf_id: dict[str, BBox]
    chart_bbox_px: BBox
    render_meta: dict[str, Any]


__all__ = [
    "BBox",
    "DOMAIN",
    "PROMPT_BUNDLE_ID",
    "RGB",
    "RenderedTreemap",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "TreemapDataset",
    "TreemapLeaf",
    "TreemapParent",
    "TreemapRenderParams",
]

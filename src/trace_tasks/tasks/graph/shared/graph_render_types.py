"""Shared node-link render constants and trace dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]
SUPPORTED_NODE_SHAPE_VARIANTS: Tuple[str, ...] = ("circle", "rounded_square", "hexagon")
SUPPORTED_EDGE_ROUTING_VARIANTS: Tuple[str, ...] = ("straight", "mixed_arc")
SUPPORTED_LAYOUT_TRANSFORM_VARIANTS: Tuple[str, ...] = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "mirror_left_right",
    "mirror_up_down",
)


@dataclass(frozen=True)
class GraphRenderParams:
    """Resolved render-time parameters for one graph scene."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_title_font_size_px: int
    node_shape_variant: str
    edge_routing_variant: str
    node_radius_px: int
    edge_width_px: int
    arrow_length_px: int
    arrow_width_px: int
    node_border_width_px: int
    label_font_size_px: int
    theme_tone: str
    panel_style_variant: str
    background_color_rgb: Tuple[int, int, int]
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    title_color_rgb: Tuple[int, int, int]
    edge_color_rgb: Tuple[int, int, int]
    node_fill_rgb: Tuple[int, int, int]
    node_border_rgb: Tuple[int, int, int]
    label_text_rgb: Tuple[int, int, int]
    label_stroke_rgb: Tuple[int, int, int]
    information_scene_style: Dict[str, Any] | None = None
    text_legibility: Dict[str, Any] | None = None
    font_family: str = ""
    font_asset: Dict[str, Any] | None = None
    font_asset_version: str = ""
    font_exclusion_reason: str = "readout font pool; no scene-local exclusion"
    content_jitter_max_px: int = 0
    context_text_probability: float = 0.0
    context_text_max_elements: int = 0
    context_block_probability: float = 0.0
    context_block_max_elements: int = 0
    context_block_position_weights: Dict[str, float] | None = None
    context_block_clutter_level_weights: Dict[str, float] | None = None


@dataclass(frozen=True)
class RenderedGraphNode:
    """Trace-ready node placement for one rendered graph scene."""

    label: str
    degree: int
    center_xy: Point
    bbox_xyxy: BBox
    neighbors: Tuple[str, ...]
    successors: Tuple[str, ...] = ()
    predecessors: Tuple[str, ...] = ()
    color_name: str | None = None
    fill_rgb: Tuple[int, int, int] | None = None
    border_rgb: Tuple[int, int, int] | None = None
    label_text_rgb: Tuple[int, int, int] | None = None
    label_stroke_rgb: Tuple[int, int, int] | None = None


@dataclass(frozen=True)
class RenderedGraphEdge:
    """Trace-ready edge segment for one rendered graph scene."""

    edge_id: str
    node_u_label: str
    node_v_label: str
    directed: bool
    segment_px: Tuple[Point, Point]
    route_variant: str = "straight"
    control_px: Point | None = None
    weight: int | None = None
    weight_label_bbox_xyxy: BBox | None = None
    edge_label: str | None = None
    edge_label_bbox_xyxy: BBox | None = None
    color_name: str | None = None
    edge_color_rgb: Tuple[int, int, int] | None = None


@dataclass(frozen=True)
class RenderedGraphScene:
    """Full render output for one graph scene."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    nodes: Tuple[RenderedGraphNode, ...]
    edges: Tuple[RenderedGraphEdge, ...]
    layout_variant: str
    layout_transform_variant: str
    edge_routing_variant: str
    crossing_count: int
    resolved_label_font_size_px: int
    resolved_label_stroke_width_px: int


__all__ = [
    "BBox",
    "GraphRenderParams",
    "Point",
    "RenderedGraphEdge",
    "RenderedGraphNode",
    "RenderedGraphScene",
    "SUPPORTED_EDGE_ROUTING_VARIANTS",
    "SUPPORTED_LAYOUT_TRANSFORM_VARIANTS",
    "SUPPORTED_NODE_SHAPE_VARIANTS",
]

"""State objects for radial Sankey chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "radial_sankey"
SCENE_NAMESPACE = "charts.radial_sankey"
PROMPT_BUNDLE_ID = "charts_radial_sankey_v1"

RADIAL_CHORD_SANKEY = "radial_chord_sankey"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (RADIAL_CHORD_SANKEY,)

RGB = tuple[int, int, int]
Point = tuple[float, float]
BBox = list[float]


@dataclass(frozen=True)
class FlowNode:
    node_id: str
    label: str
    role: str
    index: int


@dataclass(frozen=True)
class FlowLink:
    link_id: str
    source_id: str
    source_label: str
    target_id: str
    target_label: str
    value: int


@dataclass(frozen=True)
class RadialSankeyFrame:
    scene_variant: str
    scene_probabilities: dict[str, float]
    scene_title: str
    sources: tuple[FlowNode, ...]
    targets: tuple[FlowNode, ...]
    links: tuple[FlowLink, ...]
    source_count_bounds: tuple[int, int]
    target_count_bounds: tuple[int, int]
    link_count_bounds: tuple[int, int]
    max_links_per_node_side: int
    link_side_counts: dict[str, dict[str, int]]
    value_min: int
    value_max: int


@dataclass(frozen=True)
class RadialSankeyQuestion:
    branch_id: str
    branch_probabilities: dict[str, float]
    answer: int | str
    answer_type: str
    annotation_type: str
    annotation_link_ids: tuple[str, ...]
    annotation_node_ids: tuple[str, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class RadialSankeyDataset:
    frame: RadialSankeyFrame
    question: RadialSankeyQuestion


@dataclass(frozen=True)
class RadialRenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    title_band_height_px: int
    ring_radius_px: int
    chord_radius_inset_px: int
    node_width_px: int
    node_height_px: int
    node_border_width_px: int
    min_flow_width_px: int
    max_flow_width_px: int
    value_label_font_size_px: int
    value_label_gap_px: int
    node_label_font_size_px: int
    title_font_size_px: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    plot_fill_rgb: RGB
    ring_line_rgb: RGB
    source_node_fill_rgb: RGB
    target_node_fill_rgb: RGB
    node_border_rgb: RGB
    node_text_rgb: RGB
    value_label_fill_rgb: RGB
    value_label_border_rgb: RGB
    value_label_text_rgb: RGB
    title_color_rgb: RGB
    color_scheme_name: str
    flow_palette_rgb: tuple[RGB, ...]
    flow_alpha: int
    layout_offset_x_px: int
    layout_offset_y_px: int
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedRadialSankey:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    panel_bbox_px: BBox
    title_bbox_px: BBox
    plot_bbox_px: BBox
    node_bbox_map: dict[str, BBox]
    node_label_bbox_map: dict[str, BBox]
    link_bbox_map: dict[str, BBox]
    link_label_bbox_map: dict[str, BBox]
    link_center_map: dict[str, list[float]]


@dataclass(frozen=True)
class RadialSankeyRenderResult:
    image: Image.Image
    rendered_scene: RenderedRadialSankey
    render_params: RadialRenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str

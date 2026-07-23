"""State records for standard Sankey chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "sankey"
SCENE_NAMESPACE = "charts.sankey"
PROMPT_BUNDLE_ID = "charts_sankey_v1"

THREE_COLUMN_SANKEY = "three_column_sankey"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (THREE_COLUMN_SANKEY,)

RGB = tuple[int, int, int]
Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class FlowNode:
    node_id: str
    label: str
    column: str
    index: int


@dataclass(frozen=True)
class FlowPath:
    path_id: str
    source_id: str
    source_label: str
    middle_id: str
    middle_label: str
    target_id: str
    target_label: str
    first_value: int
    second_value: int
    bottleneck_value: int
    absolute_difference: int


@dataclass(frozen=True)
class SankeyFrame:
    scene_variant: str
    scene_probabilities: dict[str, float]
    scene_title: str
    sources: tuple[FlowNode, ...]
    middles: tuple[FlowNode, ...]
    targets: tuple[FlowNode, ...]
    paths: tuple[FlowPath, ...]
    source_count_bounds: tuple[int, int]
    middle_count_bounds: tuple[int, int]
    target_count_bounds: tuple[int, int]
    path_count_bounds: tuple[int, int]
    max_paths_per_node_side: int
    path_side_counts: dict[str, dict[str, int]]
    value_min: int
    value_max: int
    route_focus: dict[str, Any]


@dataclass(frozen=True)
class SankeyQuestion:
    branch_id: str
    branch_probabilities: dict[str, float]
    answer: int
    answer_type: str
    annotation_type: str
    annotation_segment_ids: tuple[str, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class SankeyDataset:
    frame: SankeyFrame
    question: SankeyQuestion


@dataclass(frozen=True)
class FlowRenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    title_band_height_px: int
    node_width_px: int
    node_height_px: int
    node_border_width_px: int
    port_separation_px: int
    shared_pair_lane_gap_px: int
    min_flow_width_px: int
    max_flow_width_px: int
    value_label_font_size_px: int
    value_label_gap_px: int
    source_middle_label_t: float
    middle_target_label_t: float
    node_label_font_size_px: int
    title_font_size_px: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    plot_fill_rgb: RGB
    node_fill_rgb: RGB
    node_border_rgb: RGB
    node_text_rgb: RGB
    value_label_fill_rgb: RGB
    value_label_border_rgb: RGB
    value_label_text_rgb: RGB
    title_color_rgb: RGB
    flow_alpha: int
    layout_offset_x_px: int
    layout_offset_y_px: int
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedSankey:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    panel_bbox_px: list[float]
    title_bbox_px: list[float]
    plot_bbox_px: list[float]
    node_bbox_map: dict[str, list[float]]
    node_label_bbox_map: dict[str, list[float]]
    segment_bbox_map: dict[str, list[float]]
    segment_label_bbox_map: dict[str, list[float]]
    segment_center_map: dict[str, list[float]]


@dataclass(frozen=True)
class SankeyRenderResult:
    image: Image.Image
    rendered_scene: RenderedSankey
    render_params: FlowRenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str

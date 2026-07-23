"""State contracts for the graph flow-network scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image

from ...shared.graph_sample_types import GraphTopologySample
from ...shared.graph_scene import RenderedGraphScene


SCENE_ID = "flow_network"
SUPPORTED_FLOW_LAYOUT_VARIANTS: Tuple[str, ...] = ("layered",)
FLOW_INTERNAL_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class FlowNetworkDefaults:
    """Stable fallback defaults for flow-network scenes."""

    node_count_min: int = 5
    node_count_max: int = 6
    max_flow_value_min: int = 2
    max_flow_value_max: int = 6
    max_flow_cut_edge_count_min: int = 1
    max_flow_cut_edge_count_max: int = 2
    min_cut_edge_count_min: int = 1
    min_cut_edge_count_max: int = 5
    cut_capacity_part_max: int = 9
    distractor_edge_min: int = 1
    distractor_edge_max: int = 2
    max_flow_distractor_edge_min: int = 0
    max_flow_distractor_edge_max: int = 1
    max_crossing_count: int = 999
    canvas_width: int = 864
    canvas_height: int = 640
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 20
    panel_title_font_size_px: int = 24
    node_shape_variant: str = "circle"
    node_radius_min_px: int = 20
    node_radius_max_px: int = 25
    edge_width_px: int = 4
    arrow_length_px: int = 14
    arrow_width_px: int = 9
    node_border_width_px: int = 2
    label_font_size_px: int = 20
    capacity_label_font_size_px: int = 20
    capacity_label_offset_px: int = 22
    capacity_label_padding_px: int = 6
    node_color_name: str = "blue"


@dataclass(frozen=True)
class FlowNetworkAxes:
    """Semantic sampling axes for one capacitated source-sink network."""

    node_count: int
    target_cut_edge_count: int
    target_flow_value: int
    distractor_edge_count: int


@dataclass(frozen=True)
class ResolvedFlowNetworkAxes:
    """Resolved semantic and visual axes for one rendered network."""

    node_count: int
    target_cut_edge_count: int
    target_flow_value: int
    distractor_edge_count: int
    layout_variant: str
    layout_transform_variant: str
    edge_routing_variant: str
    node_color_name: str
    max_crossing_count: int
    node_count_probabilities: Dict[str, float]
    target_cut_edge_count_probabilities: Dict[str, float]
    target_flow_value_probabilities: Dict[str, float]
    distractor_edge_count_probabilities: Dict[str, float]
    layout_variant_probabilities: Dict[str, float]
    layout_transform_variant_probabilities: Dict[str, float]
    edge_routing_variant_probabilities: Dict[str, float]
    node_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class CutResult:
    """One verified source-sink cut."""

    value: int
    source_side: Tuple[int, ...]
    sink_side: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class FlowNetworkSample:
    """Trace-ready capacitated flow network."""

    graph_sample: GraphTopologySample
    source_label: str
    sink_label: str
    capacity_by_edge_label: Dict[Tuple[str, str], int]
    original_max_flow_value: int
    original_min_cut_edges: Tuple[Tuple[str, str], ...]
    original_min_cut_partition: Tuple[Tuple[str, ...], Tuple[str, ...]]


@dataclass(frozen=True)
class FlowNetworkRender:
    """Rendered flow-network scene plus image-level metadata."""

    image: Image.Image
    rendered_scene: RenderedGraphScene
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]


@dataclass(frozen=True)
class FlowNetworkSceneBundle:
    """Scene-level render bundle before public task answer binding."""

    axes: ResolvedFlowNetworkAxes
    render_params: Any
    flow_sample: FlowNetworkSample
    render: FlowNetworkRender
    annotation_edges: Tuple[Tuple[str, str], ...]
    annotation_projection: Mapping[str, Any]
    annotation_segments: Tuple[Any, ...]
    capacity_label_font_size_px: int
    capacity_label_offset_px: int
    capacity_label_padding_px: int


__all__ = [
    "SCENE_ID",
    "SUPPORTED_FLOW_LAYOUT_VARIANTS",
    "FLOW_INTERNAL_LABELS",
    "CutResult",
    "FlowNetworkAxes",
    "FlowNetworkDefaults",
    "FlowNetworkRender",
    "FlowNetworkSample",
    "FlowNetworkSceneBundle",
    "ResolvedFlowNetworkAxes",
]

"""State records and fixed supports for the pipe-network graph scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import networkx as nx
from PIL import Image


SCENE_ID = "pipe_network"
GridCell = Tuple[int, int]
NodeEdge = Tuple[int, int]
LabelEdge = Tuple[str, str]

SUPPORTED_PIPE_GRID_SHAPE_VARIANTS: Tuple[str, ...] = ("3x4", "3x5", "4x4", "4x5")
SUPPORTED_PIPE_LABEL_VARIANTS: Tuple[str, ...] = ("letters", "numbers")
PIPE_VISUAL_STYLE_IDS: Tuple[str, ...] = (
    "industrial_steel",
    "copper_plumbing",
    "teal_plant",
    "blueprint_tubes",
)


@dataclass(frozen=True)
class PipeJunctionNetworkSample:
    """Trace-ready sampled pipe-junction network."""

    graph: nx.Graph
    node_labels: Tuple[str, ...]
    label_by_node: Dict[int, str]
    node_by_label: Dict[str, int]
    node_grid_cells: Dict[int, GridCell]
    open_edges: Tuple[NodeEdge, ...]
    blocked_edges: Tuple[NodeEdge, ...]
    open_edge_labels: Tuple[LabelEdge, ...]
    blocked_edge_labels: Tuple[LabelEdge, ...]
    adjacency_by_label: Dict[str, Tuple[str, ...]]
    degrees_by_label: Dict[str, int]
    query_label: str = ""
    source_label: str = ""
    goal_label: str = ""
    target_labels: Tuple[str, ...] = ()
    target_edges: Tuple[LabelEdge, ...] = ()
    target_shortest_path_length: int = 0
    target_reachable_count: int = 0
    target_bridge_count: int = 0
    query_distance: int = 0
    target_exact_distance_count: int = 0
    grid_shape_variant: str = ""
    label_variant: str = ""
    label_source_kind: str = ""
    label_bucket: str = ""
    label_manifest: str = ""
    label_filter: Mapping[str, Any] | None = None
    label_bucket_probabilities: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RenderedPipeJunctionNode:
    """One rendered pipe junction."""

    label: str
    open_degree: int
    grid_cell: GridCell
    center_xy: Tuple[int, int]
    bbox_xyxy: Tuple[int, int, int, int]
    open_neighbors: Tuple[str, ...]


@dataclass(frozen=True)
class RenderedPipeJunctionEdge:
    """One rendered open or blocked pipe segment."""

    edge_id: str
    node_u_label: str
    node_v_label: str
    pipe_state: str
    segment_px: Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass(frozen=True)
class RenderedPipeJunctionScene:
    """Full pipe-junction render output."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    nodes: Tuple[RenderedPipeJunctionNode, ...]
    edges: Tuple[RenderedPipeJunctionEdge, ...]
    grid_shape_variant: str
    resolved_label_font_size_px: int
    resolved_label_stroke_width_px: int
    open_pipe_width_px: int
    blocked_pipe_width_px: int


__all__ = [
    "GridCell",
    "LabelEdge",
    "NodeEdge",
    "PIPE_VISUAL_STYLE_IDS",
    "PipeJunctionNetworkSample",
    "RenderedPipeJunctionEdge",
    "RenderedPipeJunctionNode",
    "RenderedPipeJunctionScene",
    "SCENE_ID",
    "SUPPORTED_PIPE_GRID_SHAPE_VARIANTS",
    "SUPPORTED_PIPE_LABEL_VARIANTS",
]

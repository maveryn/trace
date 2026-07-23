"""Shared graph-domain sample types and small taxonomy helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import networkx as nx

from .label_assets import SUPPORTED_GRAPH_LABEL_VARIANTS


SUPPORTED_LAYOUT_VARIANTS: Tuple[str, ...] = (
    "circular",
    "shell",
    "spring",
    "grid_jitter",
    "layered",
    "component_clustered",
    "path_spine",
    "radial_tree",
)
SUPPORTED_TOPOLOGY_PROFILES: Tuple[str, ...] = ("balanced", "low_degree", "hub_heavy")
SUPPORTED_LABEL_VARIANTS: Tuple[str, ...] = SUPPORTED_GRAPH_LABEL_VARIANTS
SUPPORTED_NODE_LINK_LABEL_VARIANTS: Tuple[str, ...] = SUPPORTED_GRAPH_LABEL_VARIANTS
SUPPORTED_DEGREE_QUERY_IDS: Tuple[str, ...] = ("degree_count", "directed_degree_count")
SUPPORTED_DIRECTED_DEGREE_MODES: Tuple[str, ...] = ("in_degree", "out_degree")
SUPPORTED_NODE_COLOR_COUNT_DIRECTIONS: Tuple[str, ...] = ("undirected", "directed")
SUPPORTED_EDGE_COLOR_COUNT_DIRECTIONS: Tuple[str, ...] = ("undirected", "directed")
SUPPORTED_EDGE_ATTRIBUTE_LABEL_DIRECTIONS: Tuple[str, ...] = ("undirected", "directed")
SUPPORTED_CROSS_COLOR_EDGE_COUNT_DIRECTIONS: Tuple[str, ...] = ("undirected", "directed")
SUPPORTED_ISOLATED_AFTER_NODE_REMOVAL_DIRECTIONS: Tuple[str, ...] = ("undirected", "directed")
SUPPORTED_COMMON_NEIGHBOR_MODES: Tuple[str, ...] = (
    "undirected_common_neighbor",
    "directed_common_successor",
    "directed_common_predecessor",
)
SUPPORTED_UNIQUE_NODE_LABEL_RELATION_MODES: Tuple[str, ...] = (
    "undirected_unique_neighbor",
    "directed_unique_successor",
    "directed_unique_predecessor",
)
SUPPORTED_NAMED_NODE_DEGREE_DIRECTIONS: Tuple[str, ...] = ("undirected", "directed")
SUPPORTED_NAMED_NODE_DIRECTED_DEGREE_MODES: Tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "total_degree",
)
SUPPORTED_EXTREME_DEGREE_DIRECTIONS: Tuple[str, ...] = ("undirected", "directed")
SUPPORTED_EXTREME_DEGREE_EXTREMA: Tuple[str, ...] = ("max", "min")
SUPPORTED_EXTREME_DEGREE_DIRECTED_MODES: Tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "total_degree",
)
SUPPORTED_ARTICULATION_QUERY_IDS: Tuple[str, ...] = ("articulation_point_count",)
SUPPORTED_BRIDGE_QUERY_IDS: Tuple[str, ...] = ("bridge_count",)
SUPPORTED_OPTIMIZATION_QUERY_IDS: Tuple[str, ...] = ("minimum_spanning_tree_weight",)
SUPPORTED_COMPONENT_QUERY_IDS: Tuple[str, ...] = ("same_component_count",)
SUPPORTED_COMPONENT_EDGE_EDIT_MODES: Tuple[str, ...] = ("edge_removal", "edge_addition")
SUPPORTED_REACHABLE_QUERY_IDS: Tuple[str, ...] = ("reachable_count",)
SUPPORTED_CYCLE_QUERY_IDS: Tuple[str, ...] = ("unique_cycle_size",)
SUPPORTED_CHORDLESS_CYCLE_QUERY_IDS: Tuple[str, ...] = ("largest_chordless_cycle_size",)
SUPPORTED_HAMILTONIAN_CYCLE_QUERY_IDS: Tuple[str, ...] = (
    "next_in_hamiltonian_cycle_label",
    "previous_in_hamiltonian_cycle_label",
)
SUPPORTED_COMPONENT_COMPARISON_QUERY_IDS: Tuple[str, ...] = ("largest_component_size",)
SUPPORTED_PATH_QUERY_IDS: Tuple[str, ...] = ("shortest_path_length", "directed_shortest_path_length")
SUPPORTED_LONGEST_PATH_QUERY_IDS: Tuple[str, ...] = ("directed_longest_path_length",)
SUPPORTED_ORDER_QUERY_IDS: Tuple[str, ...] = (
    "first_in_topological_order_label",
    "last_in_topological_order_label",
)
SUPPORTED_REACHABLE_EDGE_EDIT_MODES: Tuple[str, ...] = ("edge_removal", "edge_addition")


@dataclass(frozen=True)
class GraphTopologySample:
    """Trace-ready labeled graph topology shared across graph tasks."""

    graph: nx.Graph | nx.DiGraph
    directed: bool
    node_labels: Tuple[str, ...]
    edge_labels: Tuple[Tuple[str, str], ...]
    degrees_by_label: Dict[str, int]
    in_degrees_by_label: Dict[str, int]
    out_degrees_by_label: Dict[str, int]
    adjacency_by_label: Dict[str, Tuple[str, ...]]
    successors_by_label: Dict[str, Tuple[str, ...]]
    predecessors_by_label: Dict[str, Tuple[str, ...]]
    edge_count: int
    topology_profile: str
    label_variant: str
    label_source_kind: str = field(default="", kw_only=True)
    label_bucket: str = field(default="", kw_only=True)
    label_manifest: str = field(default="", kw_only=True)
    label_filter: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    label_bucket_probabilities: Dict[str, float] = field(default_factory=dict, kw_only=True)


@dataclass(frozen=True)
class GraphCountSample(GraphTopologySample):
    """Trace-ready simple graph sample for graph counting tasks."""

    target_labels: Tuple[str, ...]
    degree_sequence: Tuple[int, ...]
    in_degree_sequence: Tuple[int, ...]
    out_degree_sequence: Tuple[int, ...]
    query_degree: int
    degree_mode: str


@dataclass(frozen=True)
class GraphCommonNeighborSample(GraphTopologySample):
    """Trace-ready graph sample for common-neighbor relation count tasks."""

    query_label_a: str
    query_label_b: str
    target_labels: Tuple[str, ...]
    target_count: int
    common_neighbor_mode: str
    graph_directionality: str


@dataclass(frozen=True)
class GraphUniqueNodeLabelRelationSample(GraphTopologySample):
    """Trace-ready graph sample for unique neighbor/successor/predecessor lookup tasks."""

    query_label: str
    answer_label: str
    target_labels: Tuple[str, ...]
    relation_mode: str
    graph_directionality: str
    supporting_edge: Tuple[str, str]


@dataclass(frozen=True)
class GraphNodeColorCountSample(GraphTopologySample):
    """Trace-ready graph sample for semantic node-color counting tasks."""

    target_labels: Tuple[str, ...]
    target_count: int
    target_color_name: str
    node_color_names_by_label: Dict[str, str]
    color_counts_by_name: Dict[str, int]
    graph_directionality: str


@dataclass(frozen=True)
class GraphEdgeColorCountSample(GraphTopologySample):
    """Trace-ready graph sample for semantic edge-color counting tasks."""

    target_edges: Tuple[Tuple[str, str], ...]
    target_count: int
    target_color_name: str
    edge_color_names_by_label: Dict[Tuple[str, str], str]
    color_counts_by_name: Dict[str, int]
    graph_directionality: str


@dataclass(frozen=True)
class GraphEdgeTextLabelCountSample(GraphTopologySample):
    """Trace-ready graph sample for visible edge-text label counting tasks."""

    target_edges: Tuple[Tuple[str, str], ...]
    target_count: int
    target_edge_label: str
    edge_label_support: Tuple[str, ...]
    edge_attribute_labels_by_label: Dict[Tuple[str, str], str]
    edge_label_counts_by_value: Dict[str, int]
    graph_directionality: str
    edge_label_source_kind: str = field(default="", kw_only=True)
    edge_label_bucket: str = field(default="", kw_only=True)
    edge_label_manifest: str = field(default="", kw_only=True)
    edge_label_filter: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    edge_label_bucket_probabilities: Dict[str, float] = field(default_factory=dict, kw_only=True)


@dataclass(frozen=True)
class GraphEdgeAttributeLabelSample(GraphTopologySample):
    """Trace-ready graph sample for visible labeled-edge lookup tasks."""

    query_edge: Tuple[str, str]
    target_edge_label: str
    edge_label_support: Tuple[str, ...]
    edge_attribute_labels_by_label: Dict[Tuple[str, str], str]
    edge_label_counts_by_value: Dict[str, int]
    graph_directionality: str
    query_path_labels: Tuple[str, ...] = ()
    query_path_edge_index: int | None = None
    query_path_edge_position: str | None = None
    edge_label_source_kind: str = field(default="", kw_only=True)
    edge_label_bucket: str = field(default="", kw_only=True)
    edge_label_manifest: str = field(default="", kw_only=True)
    edge_label_filter: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    edge_label_bucket_probabilities: Dict[str, float] = field(default_factory=dict, kw_only=True)


@dataclass(frozen=True)
class GraphCrossColorEdgeCountSample(GraphTopologySample):
    """Trace-ready graph sample for edges connecting queried node colors."""

    target_edges: Tuple[Tuple[str, str], ...]
    target_count: int
    source_color_name: str
    target_color_name: str
    node_color_names_by_label: Dict[str, str]
    color_counts_by_name: Dict[str, int]
    graph_directionality: str


@dataclass(frozen=True)
class GraphIsolatedAfterNodeRemovalSample(GraphTopologySample):
    """Trace-ready graph sample for node-removal isolated-node counting tasks."""

    query_label: str
    removed_node_label: str
    target_labels: Tuple[str, ...]
    target_count: int
    graph_directionality: str
    pre_removal_adjacency_by_label: Dict[str, Tuple[str, ...]]
    post_removal_adjacency_by_label: Dict[str, Tuple[str, ...]]
    pre_removal_successors_by_label: Dict[str, Tuple[str, ...]]
    post_removal_successors_by_label: Dict[str, Tuple[str, ...]]
    pre_removal_predecessors_by_label: Dict[str, Tuple[str, ...]]
    post_removal_predecessors_by_label: Dict[str, Tuple[str, ...]]
    pre_removal_degrees_by_label: Dict[str, int]
    post_removal_degrees_by_label: Dict[str, int]
    pre_removal_in_degrees_by_label: Dict[str, int]
    post_removal_in_degrees_by_label: Dict[str, int]
    pre_removal_out_degrees_by_label: Dict[str, int]
    post_removal_out_degrees_by_label: Dict[str, int]
    post_removal_edge_labels: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class GraphNamedNodeDegreeSample(GraphTopologySample):
    """Trace-ready graph sample for one queried node's degree value."""

    query_label: str
    target_edges: Tuple[Tuple[str, str], ...]
    target_degree: int
    degree_mode: str
    degree_sequence: Tuple[int, ...]
    in_degree_sequence: Tuple[int, ...]
    out_degree_sequence: Tuple[int, ...]
    total_degrees_by_label: Dict[str, int]


@dataclass(frozen=True)
class GraphExtremeDegreeSample(GraphTopologySample):
    """Trace-ready graph sample for an extreme degree-value comparison."""

    target_labels: Tuple[str, ...]
    target_degree: int
    extremum_mode: str
    degree_mode: str
    degree_sequence: Tuple[int, ...]
    in_degree_sequence: Tuple[int, ...]
    out_degree_sequence: Tuple[int, ...]
    queried_degrees_by_label: Dict[str, int]
    total_degrees_by_label: Dict[str, int]


@dataclass(frozen=True)
class GraphComponentSample(GraphTopologySample):
    """Trace-ready disconnected graph sample for component-relation tasks."""

    query_label: str
    target_labels: Tuple[str, ...]
    components_by_label: Tuple[Tuple[str, ...], ...]
    component_sizes: Tuple[int, ...]
    component_count: int
    target_component_size: int


@dataclass(frozen=True)
class GraphComponentAfterEdgeEditSample(GraphTopologySample):
    """Trace-ready graph sample for undirected edge-edit component-size tasks."""

    query_label: str
    edit_edge: Tuple[str, str]
    edit_operation: str
    target_labels: Tuple[str, ...]
    target_component_size: int
    pre_edit_adjacency_by_label: Dict[str, Tuple[str, ...]]
    post_edit_adjacency_by_label: Dict[str, Tuple[str, ...]]
    pre_edit_components_by_label: Tuple[Tuple[str, ...], ...]
    post_edit_components_by_label: Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class GraphReachableSample(GraphTopologySample):
    """Trace-ready directed graph sample for reachable-count tasks."""

    query_label: str
    target_labels: Tuple[str, ...]
    target_reachable_count: int
    unreachable_labels: Tuple[str, ...]
    reachable_edge_count: int
    unreachable_edge_count: int
    annotation_labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphReachableAfterEdgeEditSample(GraphTopologySample):
    """Trace-ready directed graph sample for post-edit reachable-count tasks."""

    query_label: str
    edit_edge: Tuple[str, str]
    edit_operation: str
    target_labels: Tuple[str, ...]
    target_reachable_count: int
    unreachable_labels: Tuple[str, ...]
    pre_edit_reachable_labels: Tuple[str, ...]
    post_edit_reachable_labels: Tuple[str, ...]
    pre_edit_successors_by_label: Dict[str, Tuple[str, ...]]
    post_edit_successors_by_label: Dict[str, Tuple[str, ...]]
    pre_edit_predecessors_by_label: Dict[str, Tuple[str, ...]]
    post_edit_predecessors_by_label: Dict[str, Tuple[str, ...]]
    pre_edit_edge_labels: Tuple[Tuple[str, str], ...]
    post_edit_edge_labels: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class GraphLargestComponentSample(GraphTopologySample):
    """Trace-ready disconnected graph sample for largest-component tasks."""

    target_labels: Tuple[str, ...]
    components_by_label: Tuple[Tuple[str, ...], ...]
    component_sizes: Tuple[int, ...]
    component_count: int
    target_largest_component_size: int


@dataclass(frozen=True)
class GraphUniqueCycleSample(GraphTopologySample):
    """Trace-ready unicyclic graph sample for unique-cycle tasks."""

    target_labels: Tuple[str, ...]
    target_cycle_size: int
    attachment_count: int


@dataclass(frozen=True)
class GraphLargestChordlessCycleSample(GraphTopologySample):
    """Trace-ready graph sample for largest chordless-cycle tasks."""

    target_labels: Tuple[str, ...]
    target_cycle_size: int
    chordless_cycle_sizes: Tuple[int, ...]
    chordless_cycle_labels: Tuple[Tuple[str, ...], ...]
    attachment_count: int
    extra_edge_count: int


@dataclass(frozen=True)
class GraphHamiltonianCycleNeighborSample(GraphTopologySample):
    """Trace-ready graph sample for Hamiltonian-cycle neighbor lookup tasks."""

    target_labels: Tuple[str, ...]
    query_label: str
    answer_label: str
    relation_mode: str
    orientation_start_label: str
    orientation_next_label: str
    orientation_final_label: str
    hamiltonian_cycle_count: int
    extra_edge_count: int


@dataclass(frozen=True)
class GraphShortestPathSample(GraphTopologySample):
    """Trace-ready graph sample for unique shortest-path tasks."""

    source_label: str
    goal_label: str
    target_labels: Tuple[str, ...]
    target_shortest_path_length: int
    attachment_count: int
    extra_edge_count: int
    annotation_labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphLongestPathSample(GraphTopologySample):
    """Trace-ready directed DAG sample for unique longest-path tasks."""

    source_label: str
    goal_label: str
    target_labels: Tuple[str, ...]
    target_longest_path_length: int
    attachment_count: int
    extra_edge_count: int


@dataclass(frozen=True)
class GraphArticulationPointSample(GraphTopologySample):
    """Trace-ready graph sample for articulation-point counting tasks."""

    target_labels: Tuple[str, ...]
    target_count: int


@dataclass(frozen=True)
class GraphBridgeSample(GraphTopologySample):
    """Trace-ready graph sample for bridge-edge counting tasks."""

    target_edges: Tuple[Tuple[str, str], ...]
    target_count: int


@dataclass(frozen=True)
class GraphMinimumSpanningTreeSample(GraphTopologySample):
    """Trace-ready weighted graph sample for unique MST tasks."""

    edge_weights_by_label: Dict[Tuple[str, str], int]
    target_edges: Tuple[Tuple[str, str], ...]
    target_total_weight: int
    extra_edge_count: int


@dataclass(frozen=True)
class GraphTopologicalOrderSample(GraphTopologySample):
    """Trace-ready directed graph sample for unique topological-order tasks."""

    query_label: str
    target_labels: Tuple[str, ...]
    target_position: int
    extra_edge_count: int
    answer_label: str = ""
    annotation_labels: Tuple[str, ...] = ()


def graph_label_sort_key(label: str) -> Tuple[int, int | str]:
    """Return one natural sort key for graph node labels."""

    text = str(label)
    if str(text).isdigit():
        return (0, int(text))
    return (1, str(text))


def canonicalize_graph_edge_label(
    left_label: str,
    right_label: str,
    *,
    directed: bool = False,
) -> Tuple[str, str]:
    """Return one canonical label pair for a graph edge."""

    left = str(left_label)
    right = str(right_label)
    if bool(directed):
        return (left, right)
    return tuple(sorted((left, right), key=graph_label_sort_key))


def sort_graph_edge_labels(
    edges: Sequence[Tuple[str, str]],
    *,
    directed: bool = False,
) -> Tuple[Tuple[str, str], ...]:
    """Return graph edge labels in deterministic canonical order."""

    canonical = [
        canonicalize_graph_edge_label(str(left), str(right), directed=bool(directed))
        for left, right in edges
    ]
    return tuple(
        sorted(
            canonical,
            key=lambda pair: (graph_label_sort_key(str(pair[0])), graph_label_sort_key(str(pair[1]))),
        )
    )


def graph_directionality_for_query_id(query_id: str) -> str:
    """Return the graph directionality implied by one query id."""

    variant = str(query_id)
    if variant in {
        "directed_degree_count",
        "directed_shortest_path_length",
        "first_in_topological_order_label",
        "last_in_topological_order_label",
        "topological_position",
    }:
        return "directed"
    return "undirected"


def graph_degree_mode_for_query_id(query_id: str, *, degree_mode: str | None = None) -> str:
    """Return the query-degree mode implied by one query id."""

    variant = str(query_id)
    if variant == "directed_degree_count":
        if degree_mode is None:
            return "in_degree"
        mode = str(degree_mode)
        if mode not in SUPPORTED_DIRECTED_DEGREE_MODES:
            raise ValueError(f"unsupported directed degree mode: {degree_mode}")
        return str(mode)
    if degree_mode is not None and str(degree_mode) != "degree":
        raise ValueError(f"unsupported undirected degree mode: {degree_mode}")
    return "degree"


__all__ = [
    "SUPPORTED_ARTICULATION_QUERY_IDS",
    "SUPPORTED_BRIDGE_QUERY_IDS",
    "SUPPORTED_CHORDLESS_CYCLE_QUERY_IDS",
    "SUPPORTED_COMMON_NEIGHBOR_MODES",
    "SUPPORTED_COMPONENT_EDGE_EDIT_MODES",
    "SUPPORTED_COMPONENT_QUERY_IDS",
    "SUPPORTED_COMPONENT_COMPARISON_QUERY_IDS",
    "SUPPORTED_CROSS_COLOR_EDGE_COUNT_DIRECTIONS",
    "SUPPORTED_CYCLE_QUERY_IDS",
    "SUPPORTED_DEGREE_QUERY_IDS",
    "SUPPORTED_DIRECTED_DEGREE_MODES",
    "SUPPORTED_EDGE_ATTRIBUTE_LABEL_DIRECTIONS",
    "SUPPORTED_EDGE_COLOR_COUNT_DIRECTIONS",
    "SUPPORTED_EXTREME_DEGREE_DIRECTIONS",
    "SUPPORTED_EXTREME_DEGREE_DIRECTED_MODES",
    "SUPPORTED_EXTREME_DEGREE_EXTREMA",
    "SUPPORTED_HAMILTONIAN_CYCLE_QUERY_IDS",
    "SUPPORTED_ISOLATED_AFTER_NODE_REMOVAL_DIRECTIONS",
    "SUPPORTED_LAYOUT_VARIANTS",
    "SUPPORTED_LABEL_VARIANTS",
    "SUPPORTED_NAMED_NODE_DEGREE_DIRECTIONS",
    "SUPPORTED_NAMED_NODE_DIRECTED_DEGREE_MODES",
    "SUPPORTED_NODE_LINK_LABEL_VARIANTS",
    "SUPPORTED_NODE_COLOR_COUNT_DIRECTIONS",
    "SUPPORTED_LONGEST_PATH_QUERY_IDS",
    "SUPPORTED_OPTIMIZATION_QUERY_IDS",
    "SUPPORTED_ORDER_QUERY_IDS",
    "SUPPORTED_REACHABLE_EDGE_EDIT_MODES",
    "SUPPORTED_REACHABLE_QUERY_IDS",
    "SUPPORTED_PATH_QUERY_IDS",
    "SUPPORTED_TOPOLOGY_PROFILES",
    "SUPPORTED_UNIQUE_NODE_LABEL_RELATION_MODES",
    "GraphArticulationPointSample",
    "GraphBridgeSample",
    "GraphCommonNeighborSample",
    "GraphComponentAfterEdgeEditSample",
    "GraphComponentSample",
    "GraphCountSample",
    "GraphCrossColorEdgeCountSample",
    "GraphEdgeAttributeLabelSample",
    "GraphEdgeColorCountSample",
    "GraphEdgeTextLabelCountSample",
    "GraphExtremeDegreeSample",
    "GraphHamiltonianCycleNeighborSample",
    "GraphIsolatedAfterNodeRemovalSample",
    "GraphLargestComponentSample",
    "GraphLargestChordlessCycleSample",
    "GraphLongestPathSample",
    "GraphMinimumSpanningTreeSample",
    "GraphNamedNodeDegreeSample",
    "GraphNodeColorCountSample",
    "GraphReachableAfterEdgeEditSample",
    "GraphReachableSample",
    "GraphShortestPathSample",
    "GraphTopologicalOrderSample",
    "GraphTopologySample",
    "GraphUniqueCycleSample",
    "GraphUniqueNodeLabelRelationSample",
    "canonicalize_graph_edge_label",
    "graph_degree_mode_for_query_id",
    "graph_directionality_for_query_id",
    "graph_label_sort_key",
    "sort_graph_edge_labels",
]

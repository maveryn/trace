"""Shared graph-domain minimum-spanning-tree samplers."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Dict, Mapping, Sequence, Tuple

import networkx as nx

from ...shared.graph_algorithms import bfs_dist_count_by_adjacency, reconstruct_unique_shortest_path_by_adjacency
from .graph_edge_sampling import (
    _add_directed_edge_without_reciprocal,
    _add_random_directed_distractor_edges,
    _add_random_undirected_distractor_edges,
    _profile_extra_edge_budget,
)
from .graph_feasibility import *  # noqa: F403
from .graph_profile_sampling import (
    _choose_attachment_parent,
    _edge_weight_for_profile,
    _sample_profile_extra_edges,
    _sample_profile_tree_graph,
)
from .graph_sample_types import *  # noqa: F403
from .graph_topology_helpers import (
    _build_labeled_graph_topology_sample,
    _components_by_label_for_graph,
    _digraph_predecessor_adjacency_by_node,
    _digraph_successor_adjacency_by_node,
    _directed_adjacency_by_label_for_graph,
    _directed_predecessors_by_label_for_graph,
    _directed_successors_by_label_for_graph,
    _graph_adjacency_by_node,
    _has_reciprocal_edges,
    _post_removal_degree_maps_by_label,
    _undirected_adjacency_by_label_for_graph,
)

def _assign_unique_mst_weights(
    rng: random.Random,
    *,
    tree_edges: Sequence[Tuple[int, int]],
    extra_edges: Sequence[Tuple[int, int]],
    edge_weight_min: int,
    edge_weight_max: int,
) -> Dict[Tuple[int, int], int]:
    """Assign distinct edge weights that make the tree the unique MST.

    We sample a distinct subset of the allowed integer weights and reserve the
    heaviest sampled weights for the non-tree edges. This guarantees every
    non-tree edge is heavier than every tree edge, so the spanning tree is the
    unique minimum spanning tree by the cycle property.
    """

    tree = [tuple(sorted((int(left), int(right)))) for left, right in tree_edges]
    extras = [tuple(sorted((int(left), int(right)))) for left, right in extra_edges]
    total_edge_count = int(len(tree) + len(extras))
    available_weights = tuple(range(int(edge_weight_min), int(edge_weight_max) + 1))
    if int(total_edge_count) > len(available_weights):
        raise ValueError("not enough distinct edge weights available for weighted MST construction")

    selected_weights = sorted(int(value) for value in rng.sample(available_weights, int(total_edge_count)))
    tree_weights = selected_weights[: len(tree)]
    extra_weights = selected_weights[len(tree) :]
    rng.shuffle(tree_weights)
    rng.shuffle(extra_weights)

    weight_by_edge: Dict[Tuple[int, int], int] = {}
    for edge, weight in zip(tree, tree_weights):
        weight_by_edge[tuple(edge)] = int(weight)
    for edge, weight in zip(extras, extra_weights):
        weight_by_edge[tuple(edge)] = int(weight)
    return weight_by_edge

def sample_minimum_spanning_tree_weight_graph(
    rng: random.Random,
    *,
    node_count: int,
    extra_edge_count: int,
    topology_profile: str,
    label_variant: str,
    edge_weight_min: int,
    edge_weight_max: int,
) -> GraphMinimumSpanningTreeSample:
    """Construct one connected weighted graph with a unique minimum spanning tree."""

    feasible_extra_support = feasible_extra_edge_counts_for_minimum_spanning_tree(
        node_count=int(node_count),
        extra_edge_count_min=int(extra_edge_count),
        extra_edge_count_max=int(extra_edge_count),
        edge_weight_min=int(edge_weight_min),
        edge_weight_max=int(edge_weight_max),
    )
    if int(extra_edge_count) not in feasible_extra_support:
        raise ValueError("extra_edge_count is outside feasible support for the requested MST query")

    tree_graph = _sample_profile_tree_graph(
        rng,
        node_count=int(node_count),
        topology_profile=str(topology_profile),
    )
    tree_edges = [tuple(sorted((int(left), int(right)))) for left, right in tree_graph.edges()]
    graph = tree_graph.copy()
    extra_edges = _sample_profile_extra_edges(
        rng,
        graph=graph,
        extra_edge_count=int(extra_edge_count),
        topology_profile=str(topology_profile),
    )
    weight_by_edge = _assign_unique_mst_weights(
        rng,
        tree_edges=tuple(tree_edges),
        extra_edges=tuple(extra_edges),
        edge_weight_min=int(edge_weight_min),
        edge_weight_max=int(edge_weight_max),
    )
    for left, right in graph.edges():
        graph[int(left)][int(right)]["weight"] = int(weight_by_edge[tuple(sorted((int(left), int(right))))])

    mst_graph = nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")
    mst_edges = sort_graph_edge_labels(
        tuple((str(left), str(right)) for left, right in mst_graph.edges()),
        directed=False,
    )
    tree_edges_canonical = sort_graph_edge_labels(
        tuple((str(left), str(right)) for left, right in tree_edges),
        directed=False,
    )
    if mst_edges != tree_edges_canonical:
        raise ValueError("weighted MST sampler failed to preserve the intended unique spanning tree")

    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    labeled_weight_by_edge: Dict[Tuple[str, str], int] = {}
    for left, right in graph.edges():
        left_label = str(label_by_node[int(left)])
        right_label = str(label_by_node[int(right)])
        edge_label = canonicalize_graph_edge_label(left_label, right_label, directed=False)
        labeled_weight_by_edge[tuple(edge_label)] = int(graph[int(left)][int(right)]["weight"])

    mst_edge_labels = sort_graph_edge_labels(
        tuple(
            (
                str(label_by_node[int(left)]),
                str(label_by_node[int(right)]),
            )
            for left, right in mst_graph.edges()
        ),
        directed=False,
    )
    target_total_weight = sum(int(labeled_weight_by_edge[tuple(edge)]) for edge in mst_edge_labels)
    return GraphMinimumSpanningTreeSample(
        graph=topology_sample.graph,
        directed=False,
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in topology_sample.degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        edge_weights_by_label=dict(labeled_weight_by_edge),
        target_edges=tuple((str(left), str(right)) for left, right in mst_edge_labels),
        target_total_weight=int(target_total_weight),
        extra_edge_count=int(extra_edge_count),
    )


__all__ = [
    'sample_minimum_spanning_tree_weight_graph',
]

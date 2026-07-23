"""Shared graph-domain connected-component samplers."""

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
from .graph_path_order_sampling import (
    _add_random_non_edges,
    _random_bounded_positive_composition,
    _random_positive_composition,
    _random_tree_graph,
)


def _sample_connected_component_graph(
    rng: random.Random,
    *,
    size: int,
    topology_profile: str,
) -> nx.Graph:
    """Return one connected component graph matching the requested profile."""

    node_count = int(size)
    if int(node_count) <= 0:
        raise ValueError("component size must be positive")
    if int(node_count) == 1:
        graph = nx.Graph()
        graph.add_node(0)
        return graph

    profile = str(topology_profile)
    if profile == "hub_heavy":
        graph = nx.star_graph(int(node_count) - 1)
        max_extra = max(0, min(int(node_count - 2), ((int(node_count) - 1) * (int(node_count) - 2)) // 2))
        _add_random_non_edges(graph, rng, extra_edges=int(rng.randint(0, max_extra if max_extra > 0 else 0)))
        return graph

    graph = _random_tree_graph(rng, size=int(node_count))
    if profile == "low_degree":
        max_extra = 1 if int(node_count) >= 4 else 0
    else:
        complete_edges = (int(node_count) * (int(node_count) - 1)) // 2
        max_extra = max(0, min(complete_edges - (int(node_count) - 1), int(node_count // 2) + 1))
    _add_random_non_edges(graph, rng, extra_edges=int(rng.randint(0, max_extra if max_extra > 0 else 0)))
    return graph

def _build_unlabeled_disconnected_component_graph(
    rng: random.Random,
    *,
    component_sizes: Sequence[int],
    topology_profile: str,
) -> Tuple[nx.Graph, Tuple[Tuple[int, ...], ...]]:
    """Build one unlabeled disconnected graph from explicit component sizes."""

    graph = nx.Graph()
    component_nodes: list[Tuple[int, ...]] = []
    node_offset = 0
    for size in component_sizes:
        component_graph = _sample_connected_component_graph(
            rng,
            size=int(size),
            topology_profile=str(topology_profile),
        )
        mapping = {int(node): int(node + node_offset) for node in component_graph.nodes()}
        component_graph = nx.relabel_nodes(component_graph, mapping, copy=True)
        graph.add_nodes_from(component_graph.nodes())
        graph.add_edges_from(component_graph.edges())
        ordered_nodes = tuple(sorted((int(node) for node in component_graph.nodes())))
        component_nodes.append(ordered_nodes)
        node_offset += int(size)
    return graph, tuple(component_nodes)

def _build_disconnected_component_graph(
    rng: random.Random,
    *,
    component_sizes: Sequence[int],
    topology_profile: str,
    label_variant: str,
) -> Tuple[GraphTopologySample, Dict[int, str], Tuple[Tuple[int, ...], ...]]:
    """Build one labeled disconnected graph from explicit component sizes."""

    graph, component_nodes = _build_unlabeled_disconnected_component_graph(
        rng,
        component_sizes=component_sizes,
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    return topology_sample, label_by_node, tuple(component_nodes)

def sample_component_count_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_component_size: int,
    component_count: int,
    topology_profile: str,
    label_variant: str,
) -> GraphComponentSample:
    """Construct one disconnected undirected graph for same-component queries."""

    node_count_int = int(node_count)
    target_size_int = int(target_component_size)
    component_count_int = int(component_count)
    feasible_node_support = feasible_node_counts_for_component_query(
        target_component_size=int(target_size_int),
        component_count=int(component_count_int),
        node_count_min=int(target_size_int + component_count_int - 1),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested component query")

    target_component_index = int(rng.randrange(int(component_count_int)))
    remaining_nodes = int(node_count_int - target_size_int)
    other_sizes = list(
        _random_positive_composition(
            rng,
            total=int(remaining_nodes),
            parts=int(component_count_int - 1),
        )
    )
    component_sizes = []
    for component_index in range(int(component_count_int)):
        if int(component_index) == int(target_component_index):
            component_sizes.append(int(target_size_int))
        else:
            component_sizes.append(int(other_sizes.pop()))

    topology_sample, label_by_node, component_nodes = _build_disconnected_component_graph(
        rng,
        component_sizes=tuple(int(size) for size in component_sizes),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    target_nodes = tuple(int(node) for node in component_nodes[int(target_component_index)])
    query_node = int(rng.choice(target_nodes))
    component_labels = [
        tuple(sorted((str(label_by_node[int(node)]) for node in nodes), key=graph_label_sort_key))
        for nodes in component_nodes
    ]
    component_labels = sorted(component_labels, key=lambda labels: graph_label_sort_key(labels[0]) if labels else (0, ""))
    target_labels = tuple(sorted((str(label_by_node[int(node)]) for node in target_nodes), key=graph_label_sort_key))
    return GraphComponentSample(
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
        query_label=str(label_by_node[int(query_node)]),
        target_labels=tuple(str(label) for label in target_labels),
        components_by_label=tuple(tuple(str(label) for label in labels) for labels in component_labels),
        component_sizes=tuple(int(len(labels)) for labels in component_labels),
        component_count=int(component_count_int),
        target_component_size=int(target_size_int),
    )

def _sample_component_size_after_edge_removal_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_component_size: int,
    topology_profile: str,
) -> Tuple[nx.Graph, nx.Graph, Tuple[int, int], int, Tuple[int, ...]]:
    """Build a pre/post graph pair where removing one bridge yields the target component."""

    node_count_int = int(node_count)
    target_size_int = int(target_component_size)
    if int(target_size_int) < 1 or int(target_size_int) >= int(node_count_int):
        raise ValueError("edge-removal target_component_size must be in [1, node_count - 1]")
    other_size = int(node_count_int - target_size_int)
    target_component_index = int(rng.randrange(2))
    component_sizes = [int(other_size), int(other_size)]
    component_sizes[int(target_component_index)] = int(target_size_int)
    component_sizes[1 - int(target_component_index)] = int(other_size)
    base_graph, component_nodes = _build_unlabeled_disconnected_component_graph(
        rng,
        component_sizes=tuple(int(size) for size in component_sizes),
        topology_profile=str(topology_profile),
    )
    target_nodes = tuple(int(node) for node in component_nodes[int(target_component_index)])
    other_nodes = tuple(int(node) for node in component_nodes[1 - int(target_component_index)])
    bridge_left = int(rng.choice(target_nodes))
    bridge_right = int(rng.choice(other_nodes))
    pre_graph = base_graph.copy()
    pre_graph.add_edge(int(bridge_left), int(bridge_right))
    post_graph = base_graph.copy()
    query_node = int(rng.choice(target_nodes))
    return pre_graph, post_graph, (int(bridge_left), int(bridge_right)), int(query_node), tuple(int(node) for node in target_nodes)

def _sample_component_size_after_edge_addition_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_component_size: int,
    topology_profile: str,
) -> Tuple[nx.Graph, nx.Graph, Tuple[int, int], int, Tuple[int, ...]]:
    """Build a pre/post graph pair where adding one edge yields the target component."""

    node_count_int = int(node_count)
    target_size_int = int(target_component_size)
    if int(target_size_int) < 2 or int(target_size_int) > int(node_count_int):
        raise ValueError("edge-addition target_component_size must be in [2, node_count]")
    left_size = int(rng.randint(1, int(target_size_int) - 1))
    right_size = int(target_size_int - left_size)
    component_sizes = [int(left_size), int(right_size)]
    remaining = int(node_count_int - target_size_int)
    if int(remaining) > 0:
        component_sizes.append(int(remaining))
    base_graph, component_nodes = _build_unlabeled_disconnected_component_graph(
        rng,
        component_sizes=tuple(int(size) for size in component_sizes),
        topology_profile=str(topology_profile),
    )
    left_nodes = tuple(int(node) for node in component_nodes[0])
    right_nodes = tuple(int(node) for node in component_nodes[1])
    edit_left = int(rng.choice(left_nodes))
    edit_right = int(rng.choice(right_nodes))
    pre_graph = base_graph.copy()
    post_graph = base_graph.copy()
    post_graph.add_edge(int(edit_left), int(edit_right))
    query_node = int(rng.choice(left_nodes))
    target_nodes = tuple(int(node) for node in (*left_nodes, *right_nodes))
    return pre_graph, post_graph, (int(edit_left), int(edit_right)), int(query_node), tuple(int(node) for node in target_nodes)

def sample_component_size_after_edge_edit_graph(
    rng: random.Random,
    *,
    edit_operation: str,
    node_count: int,
    target_component_size: int,
    topology_profile: str,
    label_variant: str,
) -> GraphComponentAfterEdgeEditSample:
    """Construct one undirected graph with a post-edit component-size witness."""

    operation = str(edit_operation)
    if operation not in SUPPORTED_COMPONENT_EDGE_EDIT_MODES:
        raise ValueError(f"unsupported edit_operation: {edit_operation}")
    if operation == "edge_removal":
        pre_graph, post_graph, edit_edge_nodes, query_node, expected_target_nodes = _sample_component_size_after_edge_removal_graph(
            rng,
            node_count=int(node_count),
            target_component_size=int(target_component_size),
            topology_profile=str(topology_profile),
        )
    else:
        pre_graph, post_graph, edit_edge_nodes, query_node, expected_target_nodes = _sample_component_size_after_edge_addition_graph(
            rng,
            node_count=int(node_count),
            target_component_size=int(target_component_size),
            topology_profile=str(topology_profile),
        )

    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=pre_graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    query_label = str(label_by_node[int(query_node)])
    edit_edge = canonicalize_graph_edge_label(
        str(label_by_node[int(edit_edge_nodes[0])]),
        str(label_by_node[int(edit_edge_nodes[1])]),
        directed=False,
    )
    pre_edit_adjacency = _undirected_adjacency_by_label_for_graph(pre_graph, label_by_node=label_by_node)
    post_edit_adjacency = _undirected_adjacency_by_label_for_graph(post_graph, label_by_node=label_by_node)
    pre_components = _components_by_label_for_graph(pre_graph, label_by_node=label_by_node)
    post_components = _components_by_label_for_graph(post_graph, label_by_node=label_by_node)
    expected_target_label_set = {str(label_by_node[int(node)]) for node in expected_target_nodes}
    target_labels = tuple(sorted(expected_target_label_set, key=graph_label_sort_key))
    matching_component = next(
        tuple(str(label) for label in component)
        for component in post_components
        if str(query_label) in {str(label) for label in component}
    )
    if set(str(label) for label in matching_component) != set(str(label) for label in target_labels):
        raise ValueError("edge-edit sampler produced inconsistent post-edit component metadata")
    if int(len(target_labels)) != int(target_component_size):
        raise ValueError("edge-edit sampler produced an unexpected target component size")

    return GraphComponentAfterEdgeEditSample(
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
        query_label=str(query_label),
        edit_edge=(str(edit_edge[0]), str(edit_edge[1])),
        edit_operation=str(operation),
        target_labels=tuple(str(label) for label in target_labels),
        target_component_size=int(target_component_size),
        pre_edit_adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in pre_edit_adjacency.items()},
        post_edit_adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in post_edit_adjacency.items()},
        pre_edit_components_by_label=tuple(tuple(str(label) for label in component) for component in pre_components),
        post_edit_components_by_label=tuple(tuple(str(label) for label in component) for component in post_components),
    )

def sample_largest_component_size_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_largest_component_size: int,
    component_count: int,
    topology_profile: str,
    label_variant: str,
) -> GraphLargestComponentSample:
    """Construct one disconnected undirected graph with a unique largest component."""

    node_count_int = int(node_count)
    target_size_int = int(target_largest_component_size)
    component_count_int = int(component_count)
    feasible_node_support = feasible_node_counts_for_unique_largest_component(
        target_largest_component_size=int(target_size_int),
        component_count=int(component_count_int),
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested largest-component query")

    target_component_index = int(rng.randrange(int(component_count_int)))
    other_sizes = _random_bounded_positive_composition(
        rng,
        total=int(node_count_int - target_size_int),
        parts=int(component_count_int - 1),
        max_value=int(target_size_int - 1),
    )
    if other_sizes is None:
        raise ValueError("failed to sample a bounded positive component-size partition")

    component_sizes = []
    other_sizes_list = list(int(size) for size in other_sizes)
    for component_index in range(int(component_count_int)):
        if int(component_index) == int(target_component_index):
            component_sizes.append(int(target_size_int))
        else:
            component_sizes.append(int(other_sizes_list.pop()))

    topology_sample, label_by_node, component_nodes = _build_disconnected_component_graph(
        rng,
        component_sizes=tuple(int(size) for size in component_sizes),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    component_labels = [
        tuple(sorted((str(label_by_node[int(node)]) for node in nodes), key=graph_label_sort_key))
        for nodes in component_nodes
    ]
    component_labels = sorted(component_labels, key=lambda labels: graph_label_sort_key(labels[0]) if labels else (0, ""))
    target_nodes = tuple(int(node) for node in component_nodes[int(target_component_index)])
    target_labels = tuple(sorted((str(label_by_node[int(node)]) for node in target_nodes), key=graph_label_sort_key))
    return GraphLargestComponentSample(
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
        target_labels=tuple(str(label) for label in target_labels),
        components_by_label=tuple(tuple(str(label) for label in labels) for labels in component_labels),
        component_sizes=tuple(int(len(labels)) for labels in component_labels),
        component_count=int(component_count_int),
        target_largest_component_size=int(target_size_int),
    )


__all__ = [
    '_build_disconnected_component_graph',
    '_build_unlabeled_disconnected_component_graph',
    'sample_component_count_graph',
    'sample_component_size_after_edge_edit_graph',
    'sample_largest_component_size_graph',
]

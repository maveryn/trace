"""Shared graph-domain bridge and articulation samplers."""

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
    _random_tree_graph,
)


def _sample_zero_articulation_graph(
    rng: random.Random,
    *,
    node_count: int,
    topology_profile: str,
) -> nx.Graph:
    """Return one simple graph with zero articulation points."""

    node_count_int = int(node_count)
    if int(node_count_int) < 3:
        raise ValueError("zero-articulation sampling requires at least three nodes")
    graph = nx.cycle_graph(int(node_count_int))
    profile = str(topology_profile)
    if profile != "low_degree":
        max_extra = max(0, min(int(node_count_int // 2), ((node_count_int * (node_count_int - 1)) // 2) - int(node_count_int)))
        extra_edges = int(rng.randint(0, max_extra))
        _add_random_non_edges(graph, rng, extra_edges=int(extra_edges))
    if any(True for _ in nx.articulation_points(graph)):
        raise ValueError("zero-articulation sampler produced an articulation point")
    return graph

def _sample_zero_bridge_graph(
    rng: random.Random,
    *,
    node_count: int,
    topology_profile: str,
) -> nx.Graph:
    """Return one connected graph with zero bridges."""

    node_count_int = int(node_count)
    if int(node_count_int) < 3:
        raise ValueError("zero-bridge sampling requires at least three nodes")
    graph = nx.cycle_graph(int(node_count_int))
    profile = str(topology_profile)
    if profile != "low_degree":
        complete_edges = (int(node_count_int) * (int(node_count_int) - 1)) // 2
        cycle_edges = int(node_count_int)
        max_extra = max(0, min(int(node_count_int // 2), int(complete_edges - cycle_edges)))
        _add_random_non_edges(graph, rng, extra_edges=int(rng.randint(0, max_extra if max_extra > 0 else 0)))
    if any(True for _ in nx.bridges(graph)):
        raise ValueError("zero-bridge sampler produced a bridge edge")
    return graph

def _sample_bridge_block_sizes(
    rng: random.Random,
    *,
    node_count: int,
    target_count: int,
    topology_profile: str,
) -> Tuple[int, ...]:
    """Return block sizes for one connected exact-bridge construction.

    Blocks are either size `1` or bridgeless size `>= 3`. Connecting the blocks
    with a tree therefore yields exactly `target_count` bridge edges.
    """

    node_count_int = int(node_count)
    target_count_int = int(target_count)
    block_count = int(target_count_int) + 1
    if int(block_count) > int(node_count_int):
        raise ValueError("bridge block count exceeds node count")
    extra_nodes = int(node_count_int - block_count)
    if int(extra_nodes) == 1:
        raise ValueError("bridge construction cannot realize exactly one extra node beyond the bridge skeleton")
    sizes = [1] * int(block_count)
    if int(extra_nodes) <= 0:
        return tuple(int(value) for value in sizes)

    max_expanded = min(int(block_count), int(extra_nodes // 2))
    profile = str(topology_profile)
    if profile == "hub_heavy":
        expanded_count = 1
    elif profile == "low_degree":
        expanded_count = max_expanded
    else:
        expanded_count = int(rng.randint(1, max_expanded))
    recipient_indices = list(range(int(block_count)))
    rng.shuffle(recipient_indices)
    recipients = recipient_indices[: int(expanded_count)]
    for index in recipients:
        sizes[int(index)] += 2
    remaining = int(extra_nodes - (2 * expanded_count))
    while int(remaining) > 0:
        recipient = int(rng.choice(recipients))
        sizes[int(recipient)] += 1
        remaining -= 1
    return tuple(int(value) for value in sizes)

def _block_anchor_node(
    block_nodes: Sequence[int],
    *,
    topology_profile: str,
) -> int:
    """Return one anchor node inside a bridgeless block for bridge attachments."""

    nodes = tuple(int(node) for node in block_nodes)
    if not nodes:
        raise ValueError("bridge block anchor selection requires at least one node")
    profile = str(topology_profile)
    if profile == "hub_heavy":
        return int(nodes[0])
    if profile == "low_degree":
        return int(nodes[-1])
    return int(nodes[0])

def _sample_bridge_count_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_count: int,
    topology_profile: str,
) -> nx.Graph:
    """Return one connected graph with the requested number of bridges."""

    node_count_int = int(node_count)
    target_count_int = int(target_count)
    if int(target_count_int) < 0 or int(target_count_int) > max(0, int(node_count_int) - 1):
        raise ValueError("target bridge count is outside feasible bounds")
    if int(target_count_int) == 0:
        return _sample_zero_bridge_graph(
            rng,
            node_count=int(node_count_int),
            topology_profile=str(topology_profile),
        )

    block_sizes = _sample_bridge_block_sizes(
        rng,
        node_count=int(node_count_int),
        target_count=int(target_count_int),
        topology_profile=str(topology_profile),
    )
    block_count = len(block_sizes)
    profile = str(topology_profile)
    if profile == "hub_heavy":
        skeleton = nx.star_graph(int(block_count) - 1)
    elif profile == "low_degree":
        skeleton = nx.path_graph(int(block_count))
    else:
        skeleton = _random_tree_graph(rng, size=int(block_count))

    if profile == "hub_heavy":
        ordered_block_indices = tuple(
            index
            for index, _ in sorted(
                enumerate(block_sizes),
                key=lambda item: (-int(item[1]), int(item[0])),
            )
        )
        skeleton_to_size_index = {int(skeleton_index): int(size_index) for skeleton_index, size_index in enumerate(ordered_block_indices)}
    else:
        skeleton_to_size_index = {int(index): int(index) for index in range(int(block_count))}

    graph = nx.Graph()
    block_nodes_by_skeleton: Dict[int, Tuple[int, ...]] = {}
    anchor_by_skeleton: Dict[int, int] = {}
    next_node = 0
    for skeleton_index in range(int(block_count)):
        size_index = int(skeleton_to_size_index[int(skeleton_index)])
        block_size = int(block_sizes[int(size_index)])
        block_nodes = tuple(range(int(next_node), int(next_node + block_size)))
        next_node += int(block_size)
        graph.add_nodes_from(block_nodes)
        if int(block_size) >= 3:
            cycle_edges = list(zip(block_nodes, block_nodes[1:] + block_nodes[:1]))
            graph.add_edges_from((int(left), int(right)) for left, right in cycle_edges)
            if profile != "low_degree":
                block_subgraph = graph.subgraph(block_nodes).copy()
                complete_edges = (int(block_size) * (int(block_size) - 1)) // 2
                cycle_edge_count = int(block_size)
                max_extra = max(0, min(int(block_size // 2), int(complete_edges - cycle_edge_count)))
                if int(max_extra) > 0:
                    extra_edges = 1 if profile == "hub_heavy" else int(rng.randint(0, max_extra))
                    _add_random_non_edges(block_subgraph, rng, extra_edges=int(min(max_extra, extra_edges)))
                    graph.add_edges_from((int(left), int(right)) for left, right in block_subgraph.edges())
        block_nodes_by_skeleton[int(skeleton_index)] = tuple(int(node) for node in block_nodes)
        anchor_by_skeleton[int(skeleton_index)] = _block_anchor_node(block_nodes, topology_profile=profile)

    for left_block, right_block in skeleton.edges():
        graph.add_edge(int(anchor_by_skeleton[int(left_block)]), int(anchor_by_skeleton[int(right_block)]))

    bridge_edges = tuple(
        sort_graph_edge_labels(
            tuple((str(left), str(right)) for left, right in nx.bridges(graph)),
            directed=False,
        )
    )
    if int(len(bridge_edges)) != int(target_count_int):
        raise ValueError("bridge sampler failed to preserve the requested bridge count")
    return graph

def _sample_articulation_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_count: int,
    topology_profile: str,
    attempts: int = 200,
) -> nx.Graph:
    """Return one graph with the requested articulation-point count."""

    node_count_int = int(node_count)
    target_count_int = int(target_count)
    if int(target_count_int) < 0 or int(target_count_int) > max(0, int(node_count_int) - 2):
        raise ValueError("target articulation count is outside feasible bounds")
    if int(target_count_int) == 0:
        return _sample_zero_articulation_graph(
            rng,
            node_count=int(node_count_int),
            topology_profile=str(topology_profile),
        )

    for _ in range(max(1, int(attempts))):
        graph = nx.path_graph(int(target_count_int) + 2)
        next_node = int(target_count_int) + 2
        while int(next_node) < int(node_count_int):
            edges = [(int(left), int(right)) for left, right in graph.edges()]
            if not edges:
                break
            weights = [
                _edge_weight_for_profile(
                    graph,
                    edge=(int(left), int(right)),
                    topology_profile=str(topology_profile),
                )
                for left, right in edges
            ]
            left, right = rng.choices(edges, weights=weights, k=1)[0]
            graph.add_node(int(next_node))
            graph.add_edge(int(next_node), int(left))
            graph.add_edge(int(next_node), int(right))
            next_node += 1
        articulation_nodes = tuple(sorted((int(node) for node in nx.articulation_points(graph))))
        if len(articulation_nodes) == int(target_count_int):
            return graph
    raise ValueError("failed to sample the requested articulation-point support")

def sample_articulation_point_count_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_count: int,
    topology_profile: str,
    label_variant: str,
    attempts: int = 200,
) -> GraphArticulationPointSample:
    """Construct one graph with the requested articulation-point count."""

    feasible_node_support = feasible_node_counts_for_articulation_point_count(
        target_count=int(target_count),
        node_count_min=int(node_count),
        node_count_max=int(node_count),
    )
    if int(node_count) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested articulation query")

    graph = _sample_articulation_graph(
        rng,
        node_count=int(node_count),
        target_count=int(target_count),
        topology_profile=str(topology_profile),
        attempts=int(attempts),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    articulation_nodes = tuple(int(node) for node in nx.articulation_points(graph))
    target_labels = tuple(sorted((str(label_by_node[int(node)]) for node in articulation_nodes), key=graph_label_sort_key))
    return GraphArticulationPointSample(
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
        target_count=int(target_count),
    )

def sample_bridge_count_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_count: int,
    topology_profile: str,
    label_variant: str,
) -> GraphBridgeSample:
    """Construct one graph with the requested number of bridge edges."""

    feasible_node_support = feasible_node_counts_for_bridge_count(
        target_count=int(target_count),
        node_count_min=int(node_count),
        node_count_max=int(node_count),
    )
    if int(node_count) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested bridge query")

    graph = _sample_bridge_count_graph(
        rng,
        node_count=int(node_count),
        target_count=int(target_count),
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    bridge_edges = sort_graph_edge_labels(
        tuple(
            (
                str(label_by_node[int(left)]),
                str(label_by_node[int(right)]),
            )
            for left, right in nx.bridges(graph)
        ),
        directed=False,
    )
    return GraphBridgeSample(
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
        target_edges=tuple((str(left), str(right)) for left, right in bridge_edges),
        target_count=int(target_count),
    )


__all__ = [
    'sample_articulation_point_count_graph',
    'sample_bridge_count_graph',
]

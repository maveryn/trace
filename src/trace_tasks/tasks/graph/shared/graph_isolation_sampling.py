"""Shared graph-domain node-removal isolation samplers."""

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

@lru_cache(maxsize=128)
def feasible_node_counts_for_isolated_node_count_after_node_removal(
    *,
    graph_directionality: str,
    target_count: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize an exact post-removal isolated-node count."""

    directionality = str(graph_directionality)
    target_count_int = int(target_count)
    if directionality not in SUPPORTED_ISOLATED_AFTER_NODE_REMOVAL_DIRECTIONS:
        return ()
    if int(target_count_int) < 0:
        return ()

    feasible = []
    for node_count in range(int(node_count_min), int(node_count_max) + 1):
        node_count_int = int(node_count)
        if int(node_count_int) < 2:
            continue
        remaining_after_removal = int(node_count_int) - 1
        non_target_remaining = int(remaining_after_removal) - int(target_count_int)
        if int(target_count_int) > int(remaining_after_removal):
            continue
        if int(non_target_remaining) == 1:
            continue
        feasible.append(int(node_count_int))
    return tuple(int(value) for value in feasible)

def _post_removal_isolated_nodes(
    graph: nx.Graph | nx.DiGraph,
    *,
    removed_node: int,
    directed: bool,
) -> Tuple[int, ...]:
    """Return node ids isolated after removing one query node."""

    post_graph = graph.copy()
    post_graph.remove_node(int(removed_node))
    if bool(directed):
        return tuple(
            sorted(
                (
                    int(node)
                    for node in post_graph.nodes()
                    if int(post_graph.in_degree(int(node))) + int(post_graph.out_degree(int(node))) == 0
                )
            )
        )
    return tuple(sorted((int(node) for node in post_graph.nodes() if int(post_graph.degree(int(node))) == 0)))

def _isolated_after_node_removal_is_valid(
    graph: nx.Graph | nx.DiGraph,
    *,
    removed_node: int,
    target_nodes: Sequence[int],
    directed: bool,
) -> bool:
    """Return whether removing one node yields exactly the requested isolated nodes."""

    if bool(directed) and _has_reciprocal_edges(graph):  # type: ignore[arg-type]
        return False
    isolated_nodes = set(
        int(node)
        for node in _post_removal_isolated_nodes(
            graph,
            removed_node=int(removed_node),
            directed=bool(directed),
        )
    )
    return isolated_nodes == {int(node) for node in target_nodes}

def _add_isolated_removal_distractor_edges(
    graph: nx.Graph | nx.DiGraph,
    rng: random.Random,
    *,
    removed_node: int,
    target_nodes: Sequence[int],
    topology_profile: str,
    directed: bool,
) -> nx.Graph | nx.DiGraph:
    """Add safe distractor edges that preserve the post-removal isolated-node witness."""

    nodes = tuple(sorted((int(node) for node in graph.nodes())))
    target_node_set = {int(node) for node in target_nodes}
    extra_budget = _profile_extra_edge_budget(
        node_count=len(nodes),
        topology_profile=str(topology_profile),
        directed=bool(directed),
    )
    added = 0
    if bool(directed):
        candidates = [(int(left), int(right)) for left in nodes for right in nodes if int(left) != int(right)]
    else:
        candidates = [(int(left), int(right)) for left, right in nx.non_edges(graph)]
    rng.shuffle(candidates)

    for left, right in candidates:
        if int(added) >= int(extra_budget):
            break
        source = int(left)
        target = int(right)
        if int(source) in target_node_set and int(target) != int(removed_node):
            continue
        if int(target) in target_node_set and int(source) != int(removed_node):
            continue
        if bool(directed):
            digraph = graph  # type: ignore[assignment]
            if digraph.has_edge(int(source), int(target)) or digraph.has_edge(int(target), int(source)):
                continue
            digraph.add_edge(int(source), int(target))
            if not _isolated_after_node_removal_is_valid(
                digraph,
                removed_node=int(removed_node),
                target_nodes=tuple(int(node) for node in target_nodes),
                directed=True,
            ):
                digraph.remove_edge(int(source), int(target))
                continue
        else:
            undirected_graph = graph  # type: ignore[assignment]
            if undirected_graph.has_edge(int(source), int(target)):
                continue
            undirected_graph.add_edge(int(source), int(target))
            if not _isolated_after_node_removal_is_valid(
                undirected_graph,
                removed_node=int(removed_node),
                target_nodes=tuple(int(node) for node in target_nodes),
                directed=False,
            ):
                undirected_graph.remove_edge(int(source), int(target))
                continue
        added += 1
    return graph

def _sample_isolated_after_node_removal_base_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    node_count: int,
    target_count: int,
    topology_profile: str,
) -> nx.Graph | nx.DiGraph:
    """Build a graph where removing node 0 makes exactly target_count nodes isolated."""

    directionality = str(graph_directionality)
    directed = bool(directionality == "directed")
    node_count_int = int(node_count)
    target_count_int = int(target_count)
    feasible_support = feasible_node_counts_for_isolated_node_count_after_node_removal(
        graph_directionality=str(directionality),
        target_count=int(target_count_int),
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_support:
        raise ValueError("node_count is outside feasible support for isolated-node-after-removal count")

    graph: nx.Graph | nx.DiGraph = nx.DiGraph() if bool(directed) else nx.Graph()
    graph.add_nodes_from(range(int(node_count_int)))
    removed_node = 0
    target_nodes = tuple(range(1, int(target_count_int) + 1))
    non_target_nodes = tuple(range(int(target_count_int) + 1, int(node_count_int)))

    for node in target_nodes:
        if bool(directed):
            if rng.random() < 0.5:
                graph.add_edge(int(removed_node), int(node))
            else:
                graph.add_edge(int(node), int(removed_node))
        else:
            graph.add_edge(int(removed_node), int(node))

    if len(non_target_nodes) == 1:
        raise ValueError("isolated-node-after-removal sampler cannot protect a single non-target node")
    if len(non_target_nodes) >= 2:
        ordered_non_targets = list(int(node) for node in non_target_nodes)
        rng.shuffle(ordered_non_targets)
        for left, right in zip(ordered_non_targets[:-1], ordered_non_targets[1:]):
            if bool(directed):
                if rng.random() < 0.5:
                    graph.add_edge(int(left), int(right))
                else:
                    graph.add_edge(int(right), int(left))
            else:
                graph.add_edge(int(left), int(right))

    graph = _add_isolated_removal_distractor_edges(
        graph,
        rng,
        removed_node=int(removed_node),
        target_nodes=tuple(int(node) for node in target_nodes),
        topology_profile=str(topology_profile),
        directed=bool(directed),
    )
    if not _isolated_after_node_removal_is_valid(
        graph,
        removed_node=int(removed_node),
        target_nodes=tuple(int(node) for node in target_nodes),
        directed=bool(directed),
    ):
        raise ValueError("isolated-node-after-removal sampler produced inconsistent target metadata")
    return graph

def sample_isolated_node_count_after_node_removal_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    node_count: int,
    target_count: int,
    topology_profile: str,
    label_variant: str,
) -> GraphIsolatedAfterNodeRemovalSample:
    """Construct one labeled graph for post-removal isolated-node count queries."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_ISOLATED_AFTER_NODE_REMOVAL_DIRECTIONS:
        raise ValueError(f"unsupported graph_directionality: {graph_directionality}")
    directed = bool(directionality == "directed")
    graph = _sample_isolated_after_node_removal_base_graph(
        rng,
        graph_directionality=str(directionality),
        node_count=int(node_count),
        target_count=int(target_count),
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directed),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )

    removed_node = 0
    post_graph = graph.copy()
    post_graph.remove_node(int(removed_node))
    target_nodes = _post_removal_isolated_nodes(
        graph,
        removed_node=int(removed_node),
        directed=bool(directed),
    )
    target_labels = tuple(
        sorted((str(label_by_node[int(node)]) for node in target_nodes), key=graph_label_sort_key)
    )
    if int(len(target_labels)) != int(target_count):
        raise ValueError("isolated-node-after-removal sampler produced the wrong target count")

    if bool(directed):
        pre_adjacency = _directed_adjacency_by_label_for_graph(graph, label_by_node=label_by_node)  # type: ignore[arg-type]
        post_adjacency = _directed_adjacency_by_label_for_graph(post_graph, label_by_node=label_by_node)  # type: ignore[arg-type]
        pre_successors = _directed_successors_by_label_for_graph(graph, label_by_node=label_by_node)  # type: ignore[arg-type]
        post_successors = _directed_successors_by_label_for_graph(post_graph, label_by_node=label_by_node)  # type: ignore[arg-type]
        pre_predecessors = _directed_predecessors_by_label_for_graph(graph, label_by_node=label_by_node)  # type: ignore[arg-type]
        post_predecessors = _directed_predecessors_by_label_for_graph(post_graph, label_by_node=label_by_node)  # type: ignore[arg-type]
    else:
        pre_adjacency = _undirected_adjacency_by_label_for_graph(graph, label_by_node=label_by_node)  # type: ignore[arg-type]
        post_adjacency = _undirected_adjacency_by_label_for_graph(post_graph, label_by_node=label_by_node)  # type: ignore[arg-type]
        pre_successors = dict(pre_adjacency)
        post_successors = dict(post_adjacency)
        pre_predecessors = dict(pre_adjacency)
        post_predecessors = dict(post_adjacency)

    post_degrees, post_in_degrees, post_out_degrees = _post_removal_degree_maps_by_label(
        post_graph,
        label_by_node=label_by_node,
        directed=bool(directed),
    )
    pre_total_degrees = {
        str(label): (
            int(topology_sample.in_degrees_by_label[str(label)]) + int(topology_sample.out_degrees_by_label[str(label)])
            if bool(directed)
            else int(topology_sample.degrees_by_label[str(label)])
        )
        for label in topology_sample.node_labels
    }
    post_edge_labels = sort_graph_edge_labels(
        tuple((str(label_by_node[int(left)]), str(label_by_node[int(right)])) for left, right in post_graph.edges()),
        directed=bool(directed),
    )

    return GraphIsolatedAfterNodeRemovalSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
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
        query_label=str(label_by_node[int(removed_node)]),
        removed_node_label=str(label_by_node[int(removed_node)]),
        target_labels=tuple(str(label) for label in target_labels),
        target_count=int(target_count),
        graph_directionality=str(directionality),
        pre_removal_adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in pre_adjacency.items()},
        post_removal_adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in post_adjacency.items()},
        pre_removal_successors_by_label={str(key): tuple(str(value) for value in values) for key, values in pre_successors.items()},
        post_removal_successors_by_label={str(key): tuple(str(value) for value in values) for key, values in post_successors.items()},
        pre_removal_predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in pre_predecessors.items()},
        post_removal_predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in post_predecessors.items()},
        pre_removal_degrees_by_label={str(key): int(value) for key, value in pre_total_degrees.items()},
        post_removal_degrees_by_label={str(key): int(value) for key, value in post_degrees.items()},
        pre_removal_in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        post_removal_in_degrees_by_label={str(key): int(value) for key, value in post_in_degrees.items()},
        pre_removal_out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        post_removal_out_degrees_by_label={str(key): int(value) for key, value in post_out_degrees.items()},
        post_removal_edge_labels=tuple((str(left), str(right)) for left, right in post_edge_labels),
    )


__all__ = [
    'feasible_node_counts_for_isolated_node_count_after_node_removal',
    'sample_isolated_node_count_after_node_removal_graph',
]

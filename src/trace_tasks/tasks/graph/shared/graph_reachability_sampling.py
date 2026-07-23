"""Shared graph-domain directed reachability samplers."""

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

def _reachable_nodes_for_graph(graph: nx.DiGraph, *, source: int) -> Tuple[int, ...]:
    """Return directed nodes reachable from ``source``, including ``source``."""

    successors = _digraph_successor_adjacency_by_node(graph)
    dist_start, _ = bfs_dist_count_by_adjacency(successors, start=int(source))
    return tuple(sorted((int(node) for node in dist_start.keys())))

def _add_reachable_directed_region(
    graph: nx.DiGraph,
    rng: random.Random,
    *,
    ordered_nodes: Sequence[int],
    topology_profile: str,
) -> None:
    """Add a small directed region where the first node reaches all later nodes."""

    nodes = tuple(int(node) for node in ordered_nodes)
    if not nodes:
        return
    graph.add_nodes_from(nodes)
    for index, node in enumerate(nodes[1:], start=1):
        parent = _choose_attachment_parent(
            rng,
            graph=graph.subgraph(nodes[: int(index)]).copy(),
            topology_profile=str(topology_profile),
        )
        graph.add_edge(int(parent), int(node))

def _edge_edit_reachability_pair_is_valid(
    pre_graph: nx.DiGraph,
    post_graph: nx.DiGraph,
    *,
    edit_operation: str,
    edit_edge: Tuple[int, int],
    query_node: int,
    expected_post_nodes: Sequence[int],
) -> bool:
    """Return whether one directed pre/post edit pair preserves the target witness."""

    operation = str(edit_operation)
    edit_source, edit_target = int(edit_edge[0]), int(edit_edge[1])
    if set(int(node) for node in pre_graph.nodes()) != set(int(node) for node in post_graph.nodes()):
        return False
    if _has_reciprocal_edges(pre_graph) or _has_reciprocal_edges(post_graph):
        return False

    pre_edges = {(int(left), int(right)) for left, right in pre_graph.edges()}
    post_edges = {(int(left), int(right)) for left, right in post_graph.edges()}
    edit_edge_set = {(int(edit_source), int(edit_target))}
    if operation == "edge_addition":
        if pre_graph.has_edge(int(edit_source), int(edit_target)) or not post_graph.has_edge(int(edit_source), int(edit_target)):
            return False
        if post_edges != pre_edges | edit_edge_set:
            return False
    elif operation == "edge_removal":
        if not pre_graph.has_edge(int(edit_source), int(edit_target)) or post_graph.has_edge(int(edit_source), int(edit_target)):
            return False
        if pre_edges != post_edges | edit_edge_set:
            return False
    else:
        return False

    expected_set = {int(node) for node in expected_post_nodes}
    post_reachable = set(_reachable_nodes_for_graph(post_graph, source=int(query_node)))
    if post_reachable != expected_set:
        return False
    pre_reachable = set(_reachable_nodes_for_graph(pre_graph, source=int(query_node)))
    return pre_reachable != post_reachable

def _add_reachable_edge_edit_distractor_edges(
    rng: random.Random,
    *,
    pre_graph: nx.DiGraph,
    post_graph: nx.DiGraph,
    edit_operation: str,
    edit_edge: Tuple[int, int],
    query_node: int,
    expected_post_nodes: Sequence[int],
    topology_profile: str,
) -> Tuple[nx.DiGraph, nx.DiGraph]:
    """Add safe directed distractor edges without changing the post-edit witness."""

    operation = str(edit_operation)
    nodes = tuple(sorted((int(node) for node in pre_graph.nodes())))
    candidates = [(int(left), int(right)) for left in nodes for right in nodes if int(left) != int(right)]
    rng.shuffle(candidates)
    extra_budget = _profile_extra_edge_budget(
        node_count=len(nodes),
        topology_profile=str(topology_profile),
        directed=True,
    )
    added = 0
    for source, target in candidates:
        if int(added) >= int(extra_budget):
            break
        if (int(source), int(target)) == (int(edit_edge[0]), int(edit_edge[1])):
            continue
        if operation == "edge_addition":
            if pre_graph.has_edge(int(source), int(target)):
                continue
            candidate_pre = pre_graph.copy()
            candidate_pre.add_edge(int(source), int(target))
            candidate_post = candidate_pre.copy()
            if candidate_post.has_edge(int(edit_edge[0]), int(edit_edge[1])):
                continue
            candidate_post.add_edge(int(edit_edge[0]), int(edit_edge[1]))
        elif operation == "edge_removal":
            if post_graph.has_edge(int(source), int(target)):
                continue
            candidate_post = post_graph.copy()
            candidate_post.add_edge(int(source), int(target))
            candidate_pre = candidate_post.copy()
            if candidate_pre.has_edge(int(edit_edge[0]), int(edit_edge[1])):
                continue
            candidate_pre.add_edge(int(edit_edge[0]), int(edit_edge[1]))
        else:
            raise ValueError(f"unsupported edit_operation: {edit_operation}")
        if not _edge_edit_reachability_pair_is_valid(
            candidate_pre,
            candidate_post,
            edit_operation=str(operation),
            edit_edge=(int(edit_edge[0]), int(edit_edge[1])),
            query_node=int(query_node),
            expected_post_nodes=tuple(int(node) for node in expected_post_nodes),
        ):
            continue
        pre_graph = candidate_pre
        post_graph = candidate_post
        added += 1
    return pre_graph, post_graph

def _sample_reachable_count_after_edge_addition_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_reachable_count: int,
    topology_profile: str,
) -> Tuple[nx.DiGraph, nx.DiGraph, Tuple[int, int], int, Tuple[int, ...]]:
    """Build a pre/post graph pair where adding one arrow yields the target reachability."""

    node_count_int = int(node_count)
    target_count_int = int(target_reachable_count)
    if int(target_count_int) < 2 or int(target_count_int) > int(node_count_int):
        raise ValueError("edge-addition target_reachable_count must be in [2, node_count]")

    pre_count = int(rng.randint(1, int(target_count_int) - 1))
    pre_reachable_nodes = tuple(range(int(pre_count)))
    added_region_nodes = tuple(range(int(pre_count), int(target_count_int)))
    remaining_nodes = tuple(range(int(target_count_int), int(node_count_int)))
    query_node = int(pre_reachable_nodes[0])

    pre_graph = nx.DiGraph()
    pre_graph.add_nodes_from(range(int(node_count_int)))
    _add_reachable_directed_region(
        pre_graph,
        rng,
        ordered_nodes=pre_reachable_nodes,
        topology_profile=str(topology_profile),
    )
    _add_reachable_directed_region(
        pre_graph,
        rng,
        ordered_nodes=added_region_nodes,
        topology_profile=str(topology_profile),
    )
    _add_reachable_directed_region(
        pre_graph,
        rng,
        ordered_nodes=remaining_nodes,
        topology_profile=str(topology_profile),
    )

    edit_edge = (int(rng.choice(pre_reachable_nodes)), int(added_region_nodes[0]))
    post_graph = pre_graph.copy()
    post_graph.add_edge(int(edit_edge[0]), int(edit_edge[1]))
    expected_post_nodes = tuple(range(int(target_count_int)))
    if not _edge_edit_reachability_pair_is_valid(
        pre_graph,
        post_graph,
        edit_operation="edge_addition",
        edit_edge=edit_edge,
        query_node=int(query_node),
        expected_post_nodes=expected_post_nodes,
    ):
        raise ValueError("failed to construct a valid reachable-count edge-addition pair")
    pre_graph, post_graph = _add_reachable_edge_edit_distractor_edges(
        rng,
        pre_graph=pre_graph,
        post_graph=post_graph,
        edit_operation="edge_addition",
        edit_edge=edit_edge,
        query_node=int(query_node),
        expected_post_nodes=expected_post_nodes,
        topology_profile=str(topology_profile),
    )
    return pre_graph, post_graph, edit_edge, int(query_node), tuple(int(node) for node in expected_post_nodes)

def _sample_reachable_count_after_edge_removal_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_reachable_count: int,
    topology_profile: str,
) -> Tuple[nx.DiGraph, nx.DiGraph, Tuple[int, int], int, Tuple[int, ...]]:
    """Build a pre/post graph pair where removing one arrow yields the target reachability."""

    node_count_int = int(node_count)
    target_count_int = int(target_reachable_count)
    if int(target_count_int) < 1 or int(target_count_int) >= int(node_count_int):
        raise ValueError("edge-removal target_reachable_count must be in [1, node_count - 1]")

    detachable_size = int(rng.randint(1, int(node_count_int) - int(target_count_int)))
    target_nodes = tuple(range(int(target_count_int)))
    detachable_nodes = tuple(range(int(target_count_int), int(target_count_int + detachable_size)))
    remaining_nodes = tuple(range(int(target_count_int + detachable_size), int(node_count_int)))
    query_node = int(target_nodes[0])

    post_graph = nx.DiGraph()
    post_graph.add_nodes_from(range(int(node_count_int)))
    _add_reachable_directed_region(
        post_graph,
        rng,
        ordered_nodes=target_nodes,
        topology_profile=str(topology_profile),
    )
    _add_reachable_directed_region(
        post_graph,
        rng,
        ordered_nodes=detachable_nodes,
        topology_profile=str(topology_profile),
    )
    _add_reachable_directed_region(
        post_graph,
        rng,
        ordered_nodes=remaining_nodes,
        topology_profile=str(topology_profile),
    )

    edit_edge = (int(rng.choice(target_nodes)), int(detachable_nodes[0]))
    pre_graph = post_graph.copy()
    pre_graph.add_edge(int(edit_edge[0]), int(edit_edge[1]))
    expected_post_nodes = tuple(int(node) for node in target_nodes)
    if not _edge_edit_reachability_pair_is_valid(
        pre_graph,
        post_graph,
        edit_operation="edge_removal",
        edit_edge=edit_edge,
        query_node=int(query_node),
        expected_post_nodes=expected_post_nodes,
    ):
        raise ValueError("failed to construct a valid reachable-count edge-removal pair")
    pre_graph, post_graph = _add_reachable_edge_edit_distractor_edges(
        rng,
        pre_graph=pre_graph,
        post_graph=post_graph,
        edit_operation="edge_removal",
        edit_edge=edit_edge,
        query_node=int(query_node),
        expected_post_nodes=expected_post_nodes,
        topology_profile=str(topology_profile),
    )
    return pre_graph, post_graph, edit_edge, int(query_node), tuple(int(node) for node in expected_post_nodes)

def sample_reachable_count_after_edge_edit_graph(
    rng: random.Random,
    *,
    edit_operation: str,
    node_count: int,
    target_reachable_count: int,
    topology_profile: str,
    label_variant: str,
) -> GraphReachableAfterEdgeEditSample:
    """Construct one directed graph with a post-edit reachable-count witness."""

    operation = str(edit_operation)
    if operation not in SUPPORTED_REACHABLE_EDGE_EDIT_MODES:
        raise ValueError(f"unsupported edit_operation: {edit_operation}")
    feasible_node_support = feasible_node_counts_for_reachable_count_after_edge_edit(
        edit_operation=str(operation),
        target_reachable_count=int(target_reachable_count),
        node_count_min=int(node_count),
        node_count_max=int(node_count),
    )
    if int(node_count) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested reachable edge-edit query")

    if operation == "edge_removal":
        pre_graph, post_graph, edit_edge_nodes, query_node, expected_target_nodes = _sample_reachable_count_after_edge_removal_graph(
            rng,
            node_count=int(node_count),
            target_reachable_count=int(target_reachable_count),
            topology_profile=str(topology_profile),
        )
    else:
        pre_graph, post_graph, edit_edge_nodes, query_node, expected_target_nodes = _sample_reachable_count_after_edge_addition_graph(
            rng,
            node_count=int(node_count),
            target_reachable_count=int(target_reachable_count),
            topology_profile=str(topology_profile),
        )

    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=pre_graph,
        directed=True,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    query_label = str(label_by_node[int(query_node)])
    edit_edge = canonicalize_graph_edge_label(
        str(label_by_node[int(edit_edge_nodes[0])]),
        str(label_by_node[int(edit_edge_nodes[1])]),
        directed=True,
    )
    pre_edit_successors = _directed_successors_by_label_for_graph(pre_graph, label_by_node=label_by_node)
    post_edit_successors = _directed_successors_by_label_for_graph(post_graph, label_by_node=label_by_node)
    pre_edit_predecessors = _directed_predecessors_by_label_for_graph(pre_graph, label_by_node=label_by_node)
    post_edit_predecessors = _directed_predecessors_by_label_for_graph(post_graph, label_by_node=label_by_node)
    pre_reachable_nodes = _reachable_nodes_for_graph(pre_graph, source=int(query_node))
    post_reachable_nodes = _reachable_nodes_for_graph(post_graph, source=int(query_node))
    expected_target_label_set = {str(label_by_node[int(node)]) for node in expected_target_nodes}
    target_labels = tuple(sorted(expected_target_label_set, key=graph_label_sort_key))
    post_reachable_labels = tuple(
        sorted((str(label_by_node[int(node)]) for node in post_reachable_nodes), key=graph_label_sort_key)
    )
    pre_reachable_labels = tuple(
        sorted((str(label_by_node[int(node)]) for node in pre_reachable_nodes), key=graph_label_sort_key)
    )
    if set(post_reachable_labels) != set(target_labels):
        raise ValueError("reachable edge-edit sampler produced inconsistent post-edit reachable metadata")
    if int(len(target_labels)) != int(target_reachable_count):
        raise ValueError("reachable edge-edit sampler produced an unexpected target reachable count")
    if set(pre_reachable_labels) == set(post_reachable_labels):
        raise ValueError("reachable edge-edit sampler produced a non-changing edit")
    unreachable_labels = tuple(
        sorted(
            (str(label) for label in topology_sample.node_labels if str(label) not in set(target_labels)),
            key=graph_label_sort_key,
        )
    )
    pre_edit_edge_labels = sort_graph_edge_labels(
        tuple((str(label_by_node[int(left)]), str(label_by_node[int(right)])) for left, right in pre_graph.edges()),
        directed=True,
    )
    post_edit_edge_labels = sort_graph_edge_labels(
        tuple((str(label_by_node[int(left)]), str(label_by_node[int(right)])) for left, right in post_graph.edges()),
        directed=True,
    )

    return GraphReachableAfterEdgeEditSample(
        graph=topology_sample.graph,
        directed=True,
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
        target_reachable_count=int(target_reachable_count),
        unreachable_labels=tuple(str(label) for label in unreachable_labels),
        pre_edit_reachable_labels=tuple(str(label) for label in pre_reachable_labels),
        post_edit_reachable_labels=tuple(str(label) for label in post_reachable_labels),
        pre_edit_successors_by_label={str(key): tuple(str(value) for value in values) for key, values in pre_edit_successors.items()},
        post_edit_successors_by_label={str(key): tuple(str(value) for value in values) for key, values in post_edit_successors.items()},
        pre_edit_predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in pre_edit_predecessors.items()},
        post_edit_predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in post_edit_predecessors.items()},
        pre_edit_edge_labels=tuple((str(left), str(right)) for left, right in pre_edit_edge_labels),
        post_edit_edge_labels=tuple((str(left), str(right)) for left, right in post_edit_edge_labels),
    )

def sample_reachable_count_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_reachable_count: int,
    topology_profile: str,
    label_variant: str,
) -> GraphReachableSample:
    """Construct one directed graph with an exact reachable-count witness set.

    The queried source node is included in the answer/annotation set. Generation
    preserves at least one unreachable node by construction and verifies the
    final directed successor adjacency before returning.
    """

    node_count_int = int(node_count)
    target_count_int = int(target_reachable_count)
    feasible_node_support = feasible_node_counts_for_reachable_count(
        target_reachable_count=int(target_count_int),
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested reachable-count query")

    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(node_count_int)))
    source_node = 0
    reachable_nodes = tuple(range(int(target_count_int)))
    unreachable_nodes = tuple(range(int(target_count_int), int(node_count_int)))
    profile = str(topology_profile)

    for node in reachable_nodes[1:]:
        parent = _choose_attachment_parent(
            rng,
            graph=graph.subgraph(reachable_nodes[: int(node)]).copy(),
            topology_profile=profile,
        )
        graph.add_edge(int(parent), int(node))

    if unreachable_nodes:
        for offset, node in enumerate(unreachable_nodes):
            if int(offset) == 0:
                target = int(rng.choice(reachable_nodes))
                graph.add_edge(int(node), int(target))
                continue
            existing_unreachable = tuple(int(value) for value in unreachable_nodes[: int(offset)])
            target_pool = tuple(int(value) for value in (*reachable_nodes, *existing_unreachable))
            target = int(rng.choice(target_pool))
            if int(target) in reachable_nodes:
                graph.add_edge(int(node), int(target))
            else:
                if bool(rng.randrange(2)):
                    graph.add_edge(int(target), int(node))
                else:
                    graph.add_edge(int(node), int(target))

    candidate_edges = [
        (int(left), int(right))
        for left, right in nx.non_edges(graph)
        if not graph.has_edge(int(right), int(left))
        and not (int(left) in reachable_nodes and int(right) in unreachable_nodes)
    ]
    rng.shuffle(candidate_edges)
    if profile == "hub_heavy":
        extra_edge_budget = min(4, len(candidate_edges))
    elif profile == "balanced":
        extra_edge_budget = min(3, len(candidate_edges))
    else:
        extra_edge_budget = min(2, len(candidate_edges))

    for left, right in candidate_edges:
        if int(extra_edge_budget) <= 0:
            break
        graph.add_edge(int(left), int(right))
        successors = _digraph_successor_adjacency_by_node(graph)
        dist_start, _ = bfs_dist_count_by_adjacency(successors, start=int(source_node))
        reachable_after = {int(node) for node in dist_start.keys()}
        if reachable_after != {int(node) for node in reachable_nodes}:
            graph.remove_edge(int(left), int(right))
            continue
        extra_edge_budget -= 1

    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=True,
        topology_profile=profile,
        label_variant=str(label_variant),
    )
    successors_by_label = {
        str(key): tuple(str(value) for value in values)
        for key, values in topology_sample.successors_by_label.items()
    }
    dist_start, _ = bfs_dist_count_by_adjacency(successors_by_label, start=str(label_by_node[int(source_node)]))
    target_labels = tuple(
        sorted((str(label) for label in dist_start.keys()), key=graph_label_sort_key)
    )
    if int(len(target_labels)) != int(target_count_int):
        raise ValueError("reachable-count sampler failed to preserve the requested reachable set")
    unreachable_labels = tuple(
        sorted(
            (str(label_by_node[int(node)]) for node in unreachable_nodes if str(label_by_node[int(node)]) not in set(target_labels)),
            key=graph_label_sort_key,
        )
    )
    reachable_edge_count = sum(
        1
        for left, right in graph.edges()
        if int(left) in reachable_nodes and int(right) in reachable_nodes
    )
    unreachable_edge_count = int(graph.number_of_edges()) - int(reachable_edge_count)
    return GraphReachableSample(
        graph=topology_sample.graph,
        directed=True,
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
        query_label=str(label_by_node[int(source_node)]),
        target_labels=tuple(str(label) for label in target_labels),
        target_reachable_count=int(target_count_int),
        unreachable_labels=tuple(str(label) for label in unreachable_labels),
        reachable_edge_count=int(reachable_edge_count),
        unreachable_edge_count=int(unreachable_edge_count),
    )


__all__ = [
    'sample_reachable_count_after_edge_edit_graph',
    'sample_reachable_count_graph',
]

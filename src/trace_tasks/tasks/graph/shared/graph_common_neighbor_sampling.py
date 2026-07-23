"""Common-neighbor relation samplers for graph-domain node-link tasks."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Tuple

import networkx as nx

from .graph_edge_sampling import (
    _add_directed_edge_without_reciprocal,
    _profile_extra_edge_budget,
)
from .graph_sample_types import (
    SUPPORTED_COMMON_NEIGHBOR_MODES,
    GraphCommonNeighborSample,
    graph_label_sort_key,
)
from .graph_topology_helpers import _build_labeled_graph_topology_sample, _has_reciprocal_edges


def _common_neighbor_nodes(
    graph: nx.Graph | nx.DiGraph,
    *,
    query_a: int,
    query_b: int,
    common_neighbor_mode: str,
) -> Tuple[int, ...]:
    """Return node ids matching one common-neighbor style query."""

    mode = str(common_neighbor_mode)
    query_a_int = int(query_a)
    query_b_int = int(query_b)
    if mode == "undirected_common_neighbor":
        left = set(int(node) for node in graph.neighbors(int(query_a_int)))
        right = set(int(node) for node in graph.neighbors(int(query_b_int)))
    elif mode == "directed_common_successor":
        if not isinstance(graph, nx.DiGraph):
            return ()
        left = set(int(node) for node in graph.successors(int(query_a_int)))
        right = set(int(node) for node in graph.successors(int(query_b_int)))
    elif mode == "directed_common_predecessor":
        if not isinstance(graph, nx.DiGraph):
            return ()
        left = set(int(node) for node in graph.predecessors(int(query_a_int)))
        right = set(int(node) for node in graph.predecessors(int(query_b_int)))
    else:
        raise ValueError(f"unsupported common-neighbor mode: {common_neighbor_mode}")
    return tuple(
        sorted(
            int(node)
            for node in (left & right)
            if int(node) not in {int(query_a_int), int(query_b_int)}
        )
    )


def _try_add_common_neighbor_undirected_edge(
    graph: nx.Graph,
    *,
    left: int,
    right: int,
    query_a: int,
    query_b: int,
    target_nodes: set[int],
    max_degree: int,
) -> bool:
    """Add one undirected edge if it preserves exact common-neighbor support."""

    left_int = int(left)
    right_int = int(right)
    if int(left_int) == int(right_int) or graph.has_edge(int(left_int), int(right_int)):
        return False
    if int(graph.degree(int(left_int))) >= int(max_degree) or int(graph.degree(int(right_int))) >= int(max_degree):
        return False
    graph.add_edge(int(left_int), int(right_int))
    observed = set(
        _common_neighbor_nodes(
            graph,
            query_a=int(query_a),
            query_b=int(query_b),
            common_neighbor_mode="undirected_common_neighbor",
        )
    )
    if observed != {int(node) for node in target_nodes}:
        graph.remove_edge(int(left_int), int(right_int))
        return False
    return True


def _try_sample_common_neighbor_undirected_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
) -> nx.Graph | None:
    """Construct one undirected graph with an exact common-neighbor count."""

    node_count_int = int(node_count)
    target_count_int = int(target_count)
    max_degree_int = max(1, int(max_degree))
    if (
        int(node_count_int) < 3
        or int(target_count_int) < 0
        or int(target_count_int) > int(node_count_int) - 2
        or int(target_count_int) > int(max_degree_int)
    ):
        return None

    graph = nx.Graph()
    graph.add_nodes_from(range(int(node_count_int)))
    nodes = [int(node) for node in range(int(node_count_int))]
    query_a, query_b = rng.sample(nodes, 2)
    remaining_nodes = [int(node) for node in nodes if int(node) not in {int(query_a), int(query_b)}]
    target_nodes = set(rng.sample(remaining_nodes, int(target_count_int))) if int(target_count_int) else set()
    distractor_nodes = [int(node) for node in remaining_nodes if int(node) not in target_nodes]

    for target in sorted(target_nodes):
        graph.add_edge(int(query_a), int(target))
        graph.add_edge(int(query_b), int(target))

    shuffled_distractors = list(distractor_nodes)
    rng.shuffle(shuffled_distractors)
    for distractor in shuffled_distractors:
        if rng.random() < 0.70:
            query = int(query_a) if rng.random() < 0.5 else int(query_b)
            _try_add_common_neighbor_undirected_edge(
                graph,
                left=int(query),
                right=int(distractor),
                query_a=int(query_a),
                query_b=int(query_b),
                target_nodes=set(target_nodes),
                max_degree=int(max_degree_int),
            )

    if rng.random() < 0.35:
        _try_add_common_neighbor_undirected_edge(
            graph,
            left=int(query_a),
            right=int(query_b),
            query_a=int(query_a),
            query_b=int(query_b),
            target_nodes=set(target_nodes),
            max_degree=int(max_degree_int),
        )

    candidates = [
        (int(nodes[left_index]), int(nodes[right_index]))
        for left_index in range(len(nodes))
        for right_index in range(left_index + 1, len(nodes))
    ]
    rng.shuffle(candidates)
    added_extra = 0
    extra_edges = _profile_extra_edge_budget(
        node_count=int(node_count_int),
        topology_profile=str(topology_profile),
        directed=False,
    )
    for left, right in candidates:
        if int(added_extra) >= int(extra_edges):
            break
        if _try_add_common_neighbor_undirected_edge(
            graph,
            left=int(left),
            right=int(right),
            query_a=int(query_a),
            query_b=int(query_b),
            target_nodes=set(target_nodes),
            max_degree=int(max_degree_int),
        ):
            added_extra += 1

    observed = _common_neighbor_nodes(
        graph,
        query_a=int(query_a),
        query_b=int(query_b),
        common_neighbor_mode="undirected_common_neighbor",
    )
    if set(observed) != {int(node) for node in target_nodes}:
        return None
    graph.graph["common_neighbor_query_nodes"] = (int(query_a), int(query_b))
    graph.graph["common_neighbor_target_nodes"] = tuple(sorted(int(node) for node in target_nodes))
    return graph


def _try_add_common_neighbor_directed_edge(
    graph: nx.DiGraph,
    *,
    source: int,
    target: int,
    query_a: int,
    query_b: int,
    target_nodes: set[int],
    common_neighbor_mode: str,
    max_degree: int,
) -> bool:
    """Add one directed edge if it preserves exact common successor/predecessor support."""

    source_int = int(source)
    target_int = int(target)
    mode = str(common_neighbor_mode)
    if not _add_directed_edge_without_reciprocal(
        graph,
        source=int(source_int),
        target=int(target_int),
        max_degree=int(max_degree),
    ):
        return False
    observed = set(
        _common_neighbor_nodes(
            graph,
            query_a=int(query_a),
            query_b=int(query_b),
            common_neighbor_mode=str(mode),
        )
    )
    if observed != {int(node) for node in target_nodes}:
        graph.remove_edge(int(source_int), int(target_int))
        return False
    return True


def _try_sample_common_neighbor_directed_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
    common_neighbor_mode: str,
) -> nx.DiGraph | None:
    """Construct one directed graph with an exact common successor/predecessor count."""

    mode = str(common_neighbor_mode)
    if mode not in {"directed_common_successor", "directed_common_predecessor"}:
        raise ValueError(f"unsupported directed common-neighbor mode: {common_neighbor_mode}")
    node_count_int = int(node_count)
    target_count_int = int(target_count)
    max_degree_int = max(1, int(max_degree))
    if (
        int(node_count_int) < 3
        or int(target_count_int) < 0
        or int(target_count_int) > int(node_count_int) - 2
        or int(target_count_int) > int(max_degree_int)
    ):
        return None

    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(node_count_int)))
    nodes = [int(node) for node in range(int(node_count_int))]
    query_a, query_b = rng.sample(nodes, 2)
    remaining_nodes = [int(node) for node in nodes if int(node) not in {int(query_a), int(query_b)}]
    target_nodes = set(rng.sample(remaining_nodes, int(target_count_int))) if int(target_count_int) else set()
    distractor_nodes = [int(node) for node in remaining_nodes if int(node) not in target_nodes]

    for target_node in sorted(target_nodes):
        if mode == "directed_common_successor":
            if not _add_directed_edge_without_reciprocal(
                graph,
                source=int(query_a),
                target=int(target_node),
                max_degree=int(max_degree_int),
            ):
                return None
            if not _add_directed_edge_without_reciprocal(
                graph,
                source=int(query_b),
                target=int(target_node),
                max_degree=int(max_degree_int),
            ):
                return None
        else:
            if not _add_directed_edge_without_reciprocal(
                graph,
                source=int(target_node),
                target=int(query_a),
                max_degree=int(max_degree_int),
            ):
                return None
            if not _add_directed_edge_without_reciprocal(
                graph,
                source=int(target_node),
                target=int(query_b),
                max_degree=int(max_degree_int),
            ):
                return None

    shuffled_distractors = list(distractor_nodes)
    rng.shuffle(shuffled_distractors)
    for distractor in shuffled_distractors:
        if rng.random() >= 0.70:
            continue
        query = int(query_a) if rng.random() < 0.5 else int(query_b)
        if mode == "directed_common_successor":
            _try_add_common_neighbor_directed_edge(
                graph,
                source=int(query),
                target=int(distractor),
                query_a=int(query_a),
                query_b=int(query_b),
                target_nodes=set(target_nodes),
                common_neighbor_mode=str(mode),
                max_degree=int(max_degree_int),
            )
        else:
            _try_add_common_neighbor_directed_edge(
                graph,
                source=int(distractor),
                target=int(query),
                query_a=int(query_a),
                query_b=int(query_b),
                target_nodes=set(target_nodes),
                common_neighbor_mode=str(mode),
                max_degree=int(max_degree_int),
            )

    candidates = [
        (int(source), int(target))
        for source in nodes
        for target in nodes
        if int(source) != int(target)
    ]
    rng.shuffle(candidates)
    added_extra = 0
    extra_edges = _profile_extra_edge_budget(
        node_count=int(node_count_int),
        topology_profile=str(topology_profile),
        directed=True,
    )
    for source, target in candidates:
        if int(added_extra) >= int(extra_edges):
            break
        if _try_add_common_neighbor_directed_edge(
            graph,
            source=int(source),
            target=int(target),
            query_a=int(query_a),
            query_b=int(query_b),
            target_nodes=set(target_nodes),
            common_neighbor_mode=str(mode),
            max_degree=int(max_degree_int),
        ):
            added_extra += 1

    if _has_reciprocal_edges(graph):
        return None
    observed = _common_neighbor_nodes(
        graph,
        query_a=int(query_a),
        query_b=int(query_b),
        common_neighbor_mode=str(mode),
    )
    if set(observed) != {int(node) for node in target_nodes}:
        return None
    graph.graph["common_neighbor_query_nodes"] = (int(query_a), int(query_b))
    graph.graph["common_neighbor_target_nodes"] = tuple(sorted(int(node) for node in target_nodes))
    return graph


@lru_cache(maxsize=128)
def feasible_node_counts_for_common_neighbor_count(
    *,
    common_neighbor_mode: str,
    target_count: int,
    node_count_min: int,
    node_count_max: int,
    max_degree: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize the requested common-neighbor count."""

    if str(common_neighbor_mode) not in SUPPORTED_COMMON_NEIGHBOR_MODES:
        return ()
    if int(max_degree) < 1 or int(target_count) < 0:
        return ()
    return tuple(
        int(node_count)
        for node_count in range(int(node_count_min), int(node_count_max) + 1)
        if int(node_count) >= 3
        and int(target_count) <= int(node_count) - 2
        and int(target_count) <= int(max_degree)
    )


def sample_common_neighbor_count_graph(
    rng: random.Random,
    *,
    common_neighbor_mode: str,
    node_count: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
    label_variant: str,
    search_attempts: int = 100,
) -> GraphCommonNeighborSample:
    """Construct one labeled graph with the requested common-neighbor support."""

    mode = str(common_neighbor_mode)
    if mode not in SUPPORTED_COMMON_NEIGHBOR_MODES:
        raise ValueError(f"unsupported common-neighbor mode: {common_neighbor_mode}")
    graph: nx.Graph | nx.DiGraph | None = None
    for _ in range(max(1, int(search_attempts))):
        if mode == "undirected_common_neighbor":
            graph = _try_sample_common_neighbor_undirected_graph(
                rng,
                node_count=int(node_count),
                target_count=int(target_count),
                max_degree=int(max_degree),
                topology_profile=str(topology_profile),
            )
        else:
            graph = _try_sample_common_neighbor_directed_graph(
                rng,
                node_count=int(node_count),
                target_count=int(target_count),
                max_degree=int(max_degree),
                topology_profile=str(topology_profile),
                common_neighbor_mode=str(mode),
            )
        if graph is not None:
            break
    if graph is None:
        raise ValueError("failed to sample a graph for the requested common-neighbor count")

    directed = mode != "undirected_common_neighbor"
    query_nodes = tuple(int(node) for node in graph.graph.get("common_neighbor_query_nodes", ()))
    target_nodes = tuple(int(node) for node in graph.graph.get("common_neighbor_target_nodes", ()))
    if not query_nodes or len(query_nodes) != 2:
        common_targets = _common_neighbor_nodes(
            graph,
            query_a=0,
            query_b=1,
            common_neighbor_mode=str(mode),
        )
        # The sampler always stores query nodes, but keep this branch as a
        # defensive fallback for graph objects without normalized metadata.
        query_nodes = (0, 1)
        target_nodes = tuple(common_targets)

    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directed),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    query_a, query_b = int(query_nodes[0]), int(query_nodes[1])
    observed_target_nodes = _common_neighbor_nodes(
        graph,
        query_a=int(query_a),
        query_b=int(query_b),
        common_neighbor_mode=str(mode),
    )
    if target_nodes and set(int(node) for node in observed_target_nodes) != set(int(node) for node in target_nodes):
        raise ValueError("common-neighbor sampler produced inconsistent target metadata")
    target_labels = tuple(
        sorted(
            (str(label_by_node[int(node)]) for node in observed_target_nodes),
            key=graph_label_sort_key,
        )
    )
    graph_directionality = "directed" if bool(directed) else "undirected"
    return GraphCommonNeighborSample(
        graph=topology_sample.graph,
        directed=bool(directed),
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
        query_label_a=str(label_by_node[int(query_a)]),
        query_label_b=str(label_by_node[int(query_b)]),
        target_labels=tuple(str(label) for label in target_labels),
        target_count=int(len(target_labels)),
        common_neighbor_mode=str(mode),
        graph_directionality=str(graph_directionality),
    )


__all__ = [
    "_common_neighbor_nodes",
    "_try_add_common_neighbor_directed_edge",
    "_try_add_common_neighbor_undirected_edge",
    "_try_sample_common_neighbor_directed_graph",
    "_try_sample_common_neighbor_undirected_graph",
    "feasible_node_counts_for_common_neighbor_count",
    "sample_common_neighbor_count_graph",
]

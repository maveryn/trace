"""Sampling primitives for pipe-network graph scenes."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence, Tuple

import networkx as nx

from ....shared.graph_algorithms import bfs_dist_count_by_adjacency, reconstruct_unique_shortest_path_by_adjacency
from ...shared.graph_sample_types import graph_label_sort_key
from ...shared.label_assets import resolve_graph_node_labels
from .algorithms import (
    add_random_open_edges,
    candidate_edges_for_cells,
    connected_node_subset,
    induced_edges,
    label_edge,
    new_node_graph,
    open_adjacency_by_node,
    parse_pipe_grid_shape,
    random_spanning_tree_edges,
    sample_blocked_edges,
    sample_connected_cells,
    sort_node_edges,
)
from .state import GridCell, LabelEdge, NodeEdge, PipeJunctionNetworkSample


def feasible_pipe_bridge_target_counts(node_count: int, configured_targets: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return bridge-count targets realizable by the grid pipe sampler."""

    if int(node_count) <= 6:
        support = (0, 2, 5)
    elif int(node_count) == 7:
        support = (0, 1, 3)
    elif int(node_count) == 8:
        support = (0, 1, 2, 4)
    elif int(node_count) == 9:
        support = (0, 1, 2, 3, 5)
    elif int(node_count) == 10:
        support = (0, 1, 2, 3, 4)
    else:
        support = tuple(range(0, max(configured_targets or (0,)) + 1))
    configured = {int(value) for value in configured_targets}
    return tuple(int(value) for value in support if int(value) in configured)


def _build_pipe_sample(
    rng: random.Random,
    *,
    graph: nx.Graph,
    node_grid_cells: Mapping[int, GridCell],
    blocked_edges: Sequence[NodeEdge] | None = None,
    label_variant: str,
    grid_shape_variant: str,
    label_by_node_override: Mapping[int, str] | None = None,
    query_label: str = "",
    source_label: str = "",
    goal_label: str = "",
    target_labels: Sequence[str] = (),
    target_edges: Sequence[LabelEdge] = (),
    target_shortest_path_length: int = 0,
    target_reachable_count: int = 0,
    target_bridge_count: int = 0,
    query_distance: int = 0,
    target_exact_distance_count: int = 0,
) -> PipeJunctionNetworkSample:
    """Build the labeled trace sample wrapper for one pipe graph."""

    node_ids = tuple(sorted(int(node) for node in graph.nodes()))
    if isinstance(label_by_node_override, Mapping):
        label_by_node = {int(node): str(label_by_node_override[int(node)]) for node in node_ids}
        label_variant = str(label_variant)
        label_source_kind = str(label_variant)
        label_bucket = ""
        label_manifest = ""
        label_filter: Mapping[str, Any] = {}
        label_bucket_probabilities: Mapping[str, float] = {}
    else:
        resolved_labels = resolve_graph_node_labels(
            rng,
            label_variant=str(label_variant),
            object_count=len(node_ids),
            max_chars=4,
            sequential_numbers=False,
        )
        node_labels = tuple(str(label) for label in resolved_labels.labels)
        label_variant = str(resolved_labels.label_variant)
        label_source_kind = str(resolved_labels.label_source_kind)
        label_bucket = str(resolved_labels.label_bucket)
        label_manifest = str(resolved_labels.label_manifest)
        label_filter = dict(resolved_labels.label_filter)
        label_bucket_probabilities = dict(resolved_labels.label_bucket_probabilities)
        label_by_node = {int(node): str(label) for node, label in zip(node_ids, node_labels)}
    node_by_label = {str(label): int(node) for node, label in label_by_node.items()}
    open_edges = sort_node_edges((int(left), int(right)) for left, right in graph.edges())
    blocked = sort_node_edges(blocked_edges or ())
    open_edge_labels = tuple(
        sorted((label_edge(label_by_node, edge) for edge in open_edges), key=lambda pair: (graph_label_sort_key(pair[0]), graph_label_sort_key(pair[1])))
    )
    blocked_edge_labels = tuple(
        sorted((label_edge(label_by_node, edge) for edge in blocked), key=lambda pair: (graph_label_sort_key(pair[0]), graph_label_sort_key(pair[1])))
    )
    adjacency_by_label = {
        str(label_by_node[int(node)]): tuple(sorted((str(label_by_node[int(neighbor)]) for neighbor in graph.neighbors(int(node))), key=graph_label_sort_key))
        for node in node_ids
    }
    degrees_by_label = {str(label_by_node[int(node)]): int(graph.degree(int(node))) for node in node_ids}
    return PipeJunctionNetworkSample(
        graph=graph.copy(),
        node_labels=tuple(str(label_by_node[int(node)]) for node in node_ids),
        label_by_node=dict(label_by_node),
        node_by_label=dict(node_by_label),
        node_grid_cells={int(node): tuple(int(value) for value in node_grid_cells[int(node)]) for node in node_ids},
        open_edges=tuple(open_edges),
        blocked_edges=tuple(blocked),
        open_edge_labels=tuple(open_edge_labels),
        blocked_edge_labels=tuple(blocked_edge_labels),
        adjacency_by_label=dict(adjacency_by_label),
        degrees_by_label=dict(degrees_by_label),
        query_label=str(query_label),
        source_label=str(source_label),
        goal_label=str(goal_label),
        target_labels=tuple(str(label) for label in target_labels),
        target_edges=tuple((str(left), str(right)) for left, right in target_edges),
        target_shortest_path_length=int(target_shortest_path_length),
        target_reachable_count=int(target_reachable_count),
        target_bridge_count=int(target_bridge_count),
        query_distance=int(query_distance),
        target_exact_distance_count=int(target_exact_distance_count),
        grid_shape_variant=str(grid_shape_variant),
        label_variant=str(label_variant),
        label_source_kind=str(label_source_kind),
        label_bucket=str(label_bucket),
        label_manifest=str(label_manifest),
        label_filter=dict(label_filter),
        label_bucket_probabilities=dict(label_bucket_probabilities),
    )


def _sample_grid_support(
    rng: random.Random,
    *,
    node_count: int,
    grid_shape_variant: str,
) -> Tuple[Dict[int, GridCell], Tuple[NodeEdge, ...]]:
    """Sample selected grid cells and their candidate edges."""

    rows, cols = parse_pipe_grid_shape(str(grid_shape_variant))
    cells = sample_connected_cells(rng, rows=int(rows), cols=int(cols), node_count=int(node_count))
    cells_by_node = {int(index): tuple(int(value) for value in cell) for index, cell in enumerate(cells)}
    candidate_edges = candidate_edges_for_cells(cells_by_node)
    candidate_graph = new_node_graph(int(node_count))
    candidate_graph.add_edges_from(candidate_edges)
    if int(node_count) > 1 and not nx.is_connected(candidate_graph):
        raise ValueError("sampled pipe cells are not candidate-connected")
    return dict(cells_by_node), tuple(candidate_edges)


def sample_pipe_shortest_path_network(
    rng: random.Random,
    *,
    node_count: int,
    target_shortest_path_length: int,
    grid_shape_variant: str,
    label_variant: str,
    max_attempts: int = 500,
) -> PipeJunctionNetworkSample:
    """Sample a pipe network with a unique shortest open route of a target length."""

    for _ in range(max(1, int(max_attempts))):
        cells_by_node, candidate_edges = _sample_grid_support(rng, node_count=int(node_count), grid_shape_variant=str(grid_shape_variant))
        graph = new_node_graph(int(node_count))
        graph.add_edges_from(random_spanning_tree_edges(rng, nodes=tuple(range(int(node_count))), candidate_edges=candidate_edges))
        adjacency = open_adjacency_by_node(graph)
        pairs: list[Tuple[int, int, Tuple[int, ...]]] = []
        for source in sorted(graph.nodes()):
            dist_start, count_start = bfs_dist_count_by_adjacency(adjacency, start=int(source))
            for goal in sorted(graph.nodes()):
                if int(source) >= int(goal):
                    continue
                if int(dist_start.get(int(goal), -1)) != int(target_shortest_path_length):
                    continue
                if int(count_start.get(int(goal), 0)) != 1:
                    continue
                dist_goal, _ = bfs_dist_count_by_adjacency(adjacency, start=int(goal))
                path = reconstruct_unique_shortest_path_by_adjacency(
                    adjacency,
                    start=int(source),
                    goal=int(goal),
                    dist_start=dist_start,
                    dist_goal=dist_goal,
                )
                if path is not None:
                    pairs.append((int(source), int(goal), tuple(int(node) for node in path)))
        if not pairs:
            continue
        source_node, goal_node, path_nodes = rng.choice(pairs)

        available = [edge for edge in sort_node_edges(candidate_edges) if not graph.has_edge(edge[0], edge[1])]
        rng.shuffle(available)
        for left, right in available[:3]:
            graph.add_edge(int(left), int(right))
            adjacency = open_adjacency_by_node(graph)
            dist_start, count_start = bfs_dist_count_by_adjacency(adjacency, start=int(source_node))
            dist_goal, _ = bfs_dist_count_by_adjacency(adjacency, start=int(goal_node))
            path = reconstruct_unique_shortest_path_by_adjacency(
                adjacency,
                start=int(source_node),
                goal=int(goal_node),
                dist_start=dist_start,
                dist_goal=dist_goal,
            )
            if (
                int(dist_start.get(int(goal_node), -1)) != int(target_shortest_path_length)
                or int(count_start.get(int(goal_node), 0)) != 1
                or path is None
                or tuple(int(node) for node in path) != tuple(path_nodes)
            ):
                graph.remove_edge(int(left), int(right))

        blocked_edges = sample_blocked_edges(rng, candidate_edges=candidate_edges, open_edges=tuple(graph.edges()), min_count=1, max_count=5)
        provisional = _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
        )
        path_labels = tuple(str(provisional.label_by_node[int(node)]) for node in path_nodes)
        return _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
            label_by_node_override=provisional.label_by_node,
            source_label=path_labels[0],
            goal_label=path_labels[-1],
            target_labels=path_labels,
            target_shortest_path_length=int(target_shortest_path_length),
        )
    raise ValueError("failed to sample pipe shortest-path network")


def sample_pipe_reachable_network(
    rng: random.Random,
    *,
    node_count: int,
    target_reachable_count: int,
    grid_shape_variant: str,
    label_variant: str,
    max_attempts: int = 500,
) -> PipeJunctionNetworkSample:
    """Sample a pipe network with one open reachable component of target size."""

    if int(target_reachable_count) >= int(node_count):
        raise ValueError("reachable-count task needs at least one unreachable junction")
    for _ in range(max(1, int(max_attempts))):
        cells_by_node, candidate_edges = _sample_grid_support(rng, node_count=int(node_count), grid_shape_variant=str(grid_shape_variant))
        try:
            reachable_nodes = connected_node_subset(
                rng,
                nodes=tuple(range(int(node_count))),
                candidate_edges=candidate_edges,
                subset_size=int(target_reachable_count),
            )
        except ValueError:
            continue
        reachable_set = {int(node) for node in reachable_nodes}
        graph = new_node_graph(int(node_count))
        reachable_edges = induced_edges(candidate_edges, reachable_nodes)
        graph.add_edges_from(random_spanning_tree_edges(rng, nodes=reachable_nodes, candidate_edges=reachable_edges))
        add_random_open_edges(graph, rng, candidate_edges=reachable_edges, max_extra_edges=2)

        remaining_nodes = [node for node in range(int(node_count)) if int(node) not in reachable_set]
        remaining_edges = induced_edges(candidate_edges, remaining_nodes)
        for edge in remaining_edges:
            if rng.random() < 0.35:
                graph.add_edge(int(edge[0]), int(edge[1]))

        query_node = int(rng.choice(reachable_nodes))
        adjacency = open_adjacency_by_node(graph)
        dist, _ = bfs_dist_count_by_adjacency(adjacency, start=int(query_node))
        observed = tuple(sorted(int(node) for node in dist.keys()))
        if len(observed) != int(target_reachable_count) or set(observed) != reachable_set:
            continue
        blocked_edges = sample_blocked_edges(rng, candidate_edges=candidate_edges, open_edges=tuple(graph.edges()), min_count=2, max_count=6)
        provisional = _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
        )
        target_labels = tuple(sorted((str(provisional.label_by_node[int(node)]) for node in observed), key=graph_label_sort_key))
        query_label = str(provisional.label_by_node[int(query_node)])
        return _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
            label_by_node_override=provisional.label_by_node,
            query_label=query_label,
            target_labels=target_labels,
            target_reachable_count=int(target_reachable_count),
        )
    raise ValueError("failed to sample pipe reachable-count network")


def sample_pipe_bridge_network(
    rng: random.Random,
    *,
    node_count: int,
    target_bridge_count: int,
    grid_shape_variant: str,
    label_variant: str,
    max_attempts: int = 800,
) -> PipeJunctionNetworkSample:
    """Sample a connected pipe network with a target number of open bridge pipes."""

    def search_exact_bridge_graph(
        base_graph: nx.Graph,
        available_edges: Sequence[NodeEdge],
        *,
        target_count: int,
    ) -> nx.Graph | None:
        """Return a graph formed by adding support edges with exactly target bridges."""

        start_count = len(tuple(nx.bridges(base_graph)))
        if int(start_count) == int(target_count):
            return base_graph.copy()
        if int(start_count) < int(target_count):
            return None

        shuffled_edges = [(int(left), int(right)) for left, right in available_edges]
        rng.shuffle(shuffled_edges)
        frontier: list[tuple[int, int, nx.Graph, tuple[NodeEdge, ...]]] = [
            (abs(int(start_count) - int(target_count)), int(start_count), base_graph.copy(), tuple(shuffled_edges))
        ]
        seen: set[tuple[NodeEdge, ...]] = {tuple(sorted(tuple(edge) for edge in base_graph.edges()))}
        expansion_budget = 400
        while frontier and expansion_budget > 0:
            frontier.sort(key=lambda item: (int(item[0]), int(item[1]), len(item[3])))
            _gap, current_count, graph_state, remaining_edges = frontier.pop(0)
            expansion_budget -= 1
            candidate_edges = list(remaining_edges)
            rng.shuffle(candidate_edges)
            for edge in candidate_edges:
                candidate = graph_state.copy()
                candidate.add_edge(int(edge[0]), int(edge[1]))
                new_count = len(tuple(nx.bridges(candidate)))
                if int(new_count) < int(target_count) or int(new_count) >= int(current_count):
                    continue
                if int(new_count) == int(target_count):
                    return candidate
                next_remaining = tuple(candidate_edge for candidate_edge in remaining_edges if tuple(candidate_edge) != tuple(edge))
                state_key = tuple(sorted(tuple(sorted(edge_key)) for edge_key in candidate.edges()))
                if state_key in seen:
                    continue
                seen.add(state_key)
                frontier.append(
                    (
                        abs(int(new_count) - int(target_count)),
                        int(new_count),
                        candidate,
                        next_remaining,
                    )
                )
        return None

    for _ in range(max(1, int(max_attempts))):
        cells_by_node, candidate_edges = _sample_grid_support(rng, node_count=int(node_count), grid_shape_variant=str(grid_shape_variant))
        graph = new_node_graph(int(node_count))
        tree_edges = sort_node_edges(random_spanning_tree_edges(rng, nodes=tuple(range(int(node_count))), candidate_edges=candidate_edges))
        graph.add_edges_from(tree_edges)
        available = [edge for edge in sort_node_edges(candidate_edges) if edge not in set(tree_edges)]

        target_count = int(target_bridge_count)
        exact_graph = search_exact_bridge_graph(graph, available, target_count=int(target_count))
        if exact_graph is None:
            continue
        graph = exact_graph
        remaining = [edge for edge in available if not graph.has_edge(int(edge[0]), int(edge[1]))]
        current_count = len(tuple(nx.bridges(graph)))

        if int(current_count) == int(target_count) and remaining:
            stable_edges: list[tuple[int, int]] = []
            for edge in remaining:
                candidate = graph.copy()
                candidate.add_edge(int(edge[0]), int(edge[1]))
                if len(tuple(nx.bridges(candidate))) == int(target_count):
                    stable_edges.append((int(edge[0]), int(edge[1])))
            rng.shuffle(stable_edges)
            for edge in stable_edges[: rng.randint(0, min(2, len(stable_edges)))]:
                graph.add_edge(int(edge[0]), int(edge[1]))

        bridge_edges = sort_node_edges((int(left), int(right)) for left, right in nx.bridges(graph))
        if len(bridge_edges) != int(target_bridge_count):
            continue
        blocked_edges = sample_blocked_edges(rng, candidate_edges=candidate_edges, open_edges=tuple(graph.edges()), min_count=1, max_count=5)
        provisional = _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
        )
        target_edges = tuple(
            sorted((label_edge(provisional.label_by_node, edge) for edge in bridge_edges), key=lambda pair: (graph_label_sort_key(pair[0]), graph_label_sort_key(pair[1])))
        )
        return _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
            label_by_node_override=provisional.label_by_node,
            target_edges=target_edges,
            target_bridge_count=int(target_bridge_count),
        )
    raise ValueError("failed to sample pipe bridge-count network")


def sample_pipe_exact_distance_network(
    rng: random.Random,
    *,
    node_count: int,
    query_distance: int,
    target_exact_distance_count: int,
    grid_shape_variant: str,
    label_variant: str,
    max_attempts: int = 800,
) -> PipeJunctionNetworkSample:
    """Sample a pipe network with a target number of nodes exactly k open pipes away."""

    for _ in range(max(1, int(max_attempts))):
        cells_by_node, candidate_edges = _sample_grid_support(rng, node_count=int(node_count), grid_shape_variant=str(grid_shape_variant))
        graph = new_node_graph(int(node_count))
        graph.add_edges_from(random_spanning_tree_edges(rng, nodes=tuple(range(int(node_count))), candidate_edges=candidate_edges))
        available_extra = max(0, min(4, len(candidate_edges) - (int(node_count) - 1)))
        add_random_open_edges(graph, rng, candidate_edges=candidate_edges, max_extra_edges=rng.randint(0, available_extra))
        adjacency = open_adjacency_by_node(graph)
        matching_sources: list[Tuple[int, Tuple[int, ...]]] = []
        for source in sorted(graph.nodes()):
            dist, _ = bfs_dist_count_by_adjacency(adjacency, start=int(source))
            exact_nodes = tuple(sorted(int(node) for node, distance in dist.items() if int(distance) == int(query_distance)))
            if len(exact_nodes) == int(target_exact_distance_count):
                matching_sources.append((int(source), exact_nodes))
        if not matching_sources:
            continue
        query_node, exact_nodes = rng.choice(matching_sources)
        blocked_edges = sample_blocked_edges(rng, candidate_edges=candidate_edges, open_edges=tuple(graph.edges()), min_count=1, max_count=5)
        provisional = _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
        )
        query_label = str(provisional.label_by_node[int(query_node)])
        target_labels = tuple(sorted((str(provisional.label_by_node[int(node)]) for node in exact_nodes), key=graph_label_sort_key))
        return _build_pipe_sample(
            rng,
            graph=graph,
            node_grid_cells=cells_by_node,
            blocked_edges=blocked_edges,
            label_variant=str(label_variant),
            grid_shape_variant=str(grid_shape_variant),
            label_by_node_override=provisional.label_by_node,
            query_label=query_label,
            target_labels=target_labels,
            query_distance=int(query_distance),
            target_exact_distance_count=int(target_exact_distance_count),
        )
    raise ValueError("failed to sample pipe exact-distance network")


__all__ = [
    "feasible_pipe_bridge_target_counts",
    "sample_pipe_bridge_network",
    "sample_pipe_exact_distance_network",
    "sample_pipe_reachable_network",
    "sample_pipe_shortest_path_network",
]

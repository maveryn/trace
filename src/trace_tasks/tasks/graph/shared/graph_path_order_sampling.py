"""Shared graph-domain path, cycle, and topological-order samplers."""

from __future__ import annotations

import itertools
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

def _random_positive_composition(
    rng: random.Random,
    *,
    total: int,
    parts: int,
) -> Tuple[int, ...]:
    """Split ``total`` into ``parts`` positive integers."""

    if int(parts) <= 0:
        if int(total) != 0:
            raise ValueError("cannot split a non-zero total into zero parts")
        return ()
    if int(total) < int(parts):
        raise ValueError("positive composition requires total >= parts")
    remaining = int(total) - int(parts)
    buckets = [1] * int(parts)
    for _ in range(int(remaining)):
        buckets[int(rng.randrange(int(parts)))] += 1
    rng.shuffle(buckets)
    return tuple(int(value) for value in buckets)

def _random_bounded_positive_composition(
    rng: random.Random,
    *,
    total: int,
    parts: int,
    max_value: int,
    attempts: int = 200,
) -> Tuple[int, ...] | None:
    """Split ``total`` into bounded positive integers when possible."""

    total_int = int(total)
    parts_int = int(parts)
    max_value_int = int(max_value)
    if int(parts_int) <= 0:
        return () if int(total_int) == 0 else None
    if int(total_int) < int(parts_int) or int(max_value_int) <= 0:
        return None
    if int(total_int) > int(parts_int * max_value_int):
        return None
    for _ in range(max(1, int(attempts))):
        values = [1] * int(parts_int)
        remaining = int(total_int - parts_int)
        while int(remaining) > 0:
            adjustable = [index for index, value in enumerate(values) if int(value) < int(max_value_int)]
            if not adjustable:
                break
            index = int(rng.choice(adjustable))
            values[index] += 1
            remaining -= 1
        if int(remaining) == 0:
            rng.shuffle(values)
            return tuple(int(value) for value in values)
    return None

def _random_tree_graph(rng: random.Random, *, size: int) -> nx.Graph:
    """Return one deterministic connected tree with ``size`` nodes."""

    graph = nx.Graph()
    graph.add_nodes_from(range(int(size)))
    for node in range(1, int(size)):
        graph.add_edge(int(node), int(rng.randrange(int(node))))
    return graph

def _add_random_non_edges(
    graph: nx.Graph,
    rng: random.Random,
    *,
    extra_edges: int,
) -> None:
    """Add up to ``extra_edges`` random non-edges to one simple graph."""

    non_edges = list(nx.non_edges(graph))
    rng.shuffle(non_edges)
    for left, right in non_edges[: max(0, int(extra_edges))]:
        graph.add_edge(int(left), int(right))

def _sample_unicyclic_graph(
    rng: random.Random,
    *,
    node_count: int,
    cycle_size: int,
    topology_profile: str,
) -> nx.Graph:
    """Return one connected unicyclic graph with the requested unique cycle size."""

    node_count_int = int(node_count)
    cycle_size_int = int(cycle_size)
    if int(cycle_size_int) < 3 or int(cycle_size_int) > int(node_count_int):
        raise ValueError("cycle size must lie in [3, node_count] for unicyclic sampling")

    graph = nx.cycle_graph(int(cycle_size_int))
    next_node = int(cycle_size_int)
    while int(next_node) < int(node_count_int):
        parent = _choose_attachment_parent(
            rng,
            graph=graph,
            topology_profile=str(topology_profile),
        )
        graph.add_node(int(next_node))
        graph.add_edge(int(parent), int(next_node))
        next_node += 1

    cycle_basis = nx.cycle_basis(graph)
    if len(cycle_basis) != 1 or len(cycle_basis[0]) != int(cycle_size_int):
        raise ValueError("unicyclic sampler failed to preserve the requested unique cycle")
    return graph


def _canonical_cycle_order(cycle: Sequence[int]) -> Tuple[int, ...]:
    """Return a deterministic cyclic order for one undirected simple cycle."""

    nodes = [int(node) for node in cycle]
    if len(nodes) > 1 and int(nodes[0]) == int(nodes[-1]):
        nodes = nodes[:-1]
    if len(nodes) < 3:
        return tuple(int(node) for node in nodes)
    rotations: list[Tuple[int, ...]] = []
    for ordered in (nodes, list(reversed(nodes))):
        for index in range(len(ordered)):
            rotations.append(tuple(int(value) for value in (ordered[index:] + ordered[:index])))
    return min(rotations)


def _is_chordless_cycle(graph: nx.Graph, cycle: Sequence[int]) -> bool:
    """Return whether ``cycle`` is a chordless cycle in ``graph``."""

    ordered = tuple(int(node) for node in cycle)
    if len(ordered) < 3 or len(set(ordered)) != len(ordered):
        return False
    cycle_edges = {
        frozenset((int(ordered[index]), int(ordered[(index + 1) % len(ordered)])))
        for index in range(len(ordered))
    }
    for index, source in enumerate(ordered):
        target = int(ordered[(index + 1) % len(ordered)])
        if not graph.has_edge(int(source), int(target)):
            return False
    for left_index in range(len(ordered)):
        for right_index in range(left_index + 1, len(ordered)):
            edge = frozenset((int(ordered[left_index]), int(ordered[right_index])))
            if edge in cycle_edges:
                continue
            if graph.has_edge(int(ordered[left_index]), int(ordered[right_index])):
                return False
    return True


def _chordless_cycles(graph: nx.Graph) -> Tuple[Tuple[int, ...], ...]:
    """Return canonical chordless cycles from one undirected graph."""

    seen: set[Tuple[int, ...]] = set()
    cycles: list[Tuple[int, ...]] = []
    for cycle in nx.chordless_cycles(graph):
        canonical = _canonical_cycle_order(tuple(int(node) for node in cycle))
        if canonical in seen or not _is_chordless_cycle(graph, canonical):
            continue
        seen.add(canonical)
        cycles.append(canonical)
    cycles.sort(key=lambda item: (-len(item), item))
    return tuple(cycles)


def _hamiltonian_cycles(graph: nx.Graph) -> Tuple[Tuple[int, ...], ...]:
    """Return canonical Hamiltonian cycles from one small undirected graph."""

    nodes = tuple(sorted(int(node) for node in graph.nodes()))
    if len(nodes) < 3:
        return ()
    anchor = int(nodes[0])
    seen: set[Tuple[int, ...]] = set()
    cycles: list[Tuple[int, ...]] = []
    for suffix in itertools.permutations(nodes[1:]):
        candidate = (anchor, *tuple(int(node) for node in suffix))
        if all(graph.has_edge(int(candidate[index]), int(candidate[(index + 1) % len(candidate)])) for index in range(len(candidate))):
            canonical = _canonical_cycle_order(candidate)
            if canonical in seen:
                continue
            seen.add(canonical)
            cycles.append(canonical)
    cycles.sort()
    return tuple(cycles)


def _sample_largest_chordless_cycle_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_cycle_size: int,
    topology_profile: str,
) -> Tuple[nx.Graph, Tuple[int, ...], Tuple[Tuple[int, ...], ...], int, int]:
    """Construct a connected graph whose largest chordless cycle has target size."""

    node_count_int = int(node_count)
    target_size_int = int(target_cycle_size)
    feasible_node_support = feasible_node_counts_for_largest_chordless_cycle_size(
        target_cycle_size=int(target_size_int),
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested largest-chordless-cycle query")

    graph = nx.cycle_graph(int(target_size_int))
    next_node = int(target_size_int)
    secondary_edge = (0, 1)
    if int(target_size_int) > 3 and int(next_node) < int(node_count_int):
        first_extra = int(next_node)
        graph.add_node(first_extra)
        graph.add_edge(int(secondary_edge[0]), first_extra)
        graph.add_edge(first_extra, int(secondary_edge[1]))
        next_node += 1

    attachment_count = 0
    while int(next_node) < int(node_count_int):
        parent = _choose_attachment_parent(
            rng,
            graph=graph,
            topology_profile=str(topology_profile),
        )
        graph.add_node(int(next_node))
        graph.add_edge(int(parent), int(next_node))
        attachment_count += 1
        next_node += 1

    cycles = _chordless_cycles(graph)
    largest_cycles = tuple(cycle for cycle in cycles if len(cycle) == int(target_size_int))
    if not cycles or len(cycles[0]) != int(target_size_int) or len(largest_cycles) != 1:
        raise ValueError("largest-chordless-cycle construction failed before adding distractor edges")

    if str(topology_profile) == "hub_heavy":
        extra_edge_budget = 3
    elif str(topology_profile) == "balanced":
        extra_edge_budget = 2
    else:
        extra_edge_budget = 1

    extra_edges_kept = 0
    non_edges = list(nx.non_edges(graph))
    rng.shuffle(non_edges)
    for left, right in non_edges:
        if int(extra_edges_kept) >= int(extra_edge_budget):
            break
        graph.add_edge(int(left), int(right))
        candidate_cycles = _chordless_cycles(graph)
        candidate_largest_cycles = tuple(cycle for cycle in candidate_cycles if len(cycle) == int(target_size_int))
        if candidate_cycles and len(candidate_cycles[0]) == int(target_size_int) and len(candidate_largest_cycles) == 1:
            extra_edges_kept += 1
            cycles = candidate_cycles
            continue
        graph.remove_edge(int(left), int(right))

    cycles = _chordless_cycles(graph)
    largest_cycles = tuple(cycle for cycle in cycles if len(cycle) == int(target_size_int))
    if not cycles or len(cycles[0]) != int(target_size_int) or len(largest_cycles) != 1:
        raise ValueError("largest-chordless-cycle sampler failed to preserve one unique target-size cycle")
    target_cycle = tuple(largest_cycles[0])
    return graph, tuple(int(node) for node in target_cycle), cycles, int(attachment_count), int(extra_edges_kept)


def _sample_unique_hamiltonian_cycle_graph(
    rng: random.Random,
    *,
    node_count: int,
    topology_profile: str,
) -> Tuple[nx.Graph, Tuple[int, ...], Tuple[Tuple[int, ...], ...], int]:
    """Construct a small connected graph with exactly one Hamiltonian cycle."""

    node_count_int = int(node_count)
    feasible_node_support = feasible_node_counts_for_hamiltonian_cycle_neighbor(
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested Hamiltonian-cycle query")

    graph = nx.cycle_graph(int(node_count_int))
    cycles = _hamiltonian_cycles(graph)
    if len(cycles) != 1:
        raise ValueError("base Hamiltonian-cycle construction failed")

    profile = str(topology_profile)
    if profile == "hub_heavy":
        extra_edge_budget = 2
    elif profile == "balanced":
        extra_edge_budget = 1
    else:
        extra_edge_budget = int(rng.random() < 0.5)

    extra_edges_kept = 0
    non_edges = list(nx.non_edges(graph))
    rng.shuffle(non_edges)
    for left, right in non_edges:
        if int(extra_edges_kept) >= int(extra_edge_budget):
            break
        graph.add_edge(int(left), int(right))
        candidate_cycles = _hamiltonian_cycles(graph)
        if len(candidate_cycles) == 1:
            cycles = candidate_cycles
            extra_edges_kept += 1
            continue
        graph.remove_edge(int(left), int(right))

    cycles = _hamiltonian_cycles(graph)
    if len(cycles) != 1:
        raise ValueError("Hamiltonian-cycle sampler failed to preserve uniqueness")
    return graph, tuple(int(node) for node in cycles[0]), cycles, int(extra_edges_kept)


def _sample_unique_shortest_path_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_shortest_path_length: int,
    topology_profile: str,
) -> Tuple[nx.Graph, Tuple[int, ...], int]:
    """Return one connected graph with a unique shortest path of the requested length.

    The construction starts from a backbone path of the requested edge length,
    attaches at least one off-path node, and optionally adds cycle-forming
    non-edges inside the same branch anchor so the unique shortest path between
    the two backbone endpoints remains unchanged.
    """

    node_count_int = int(node_count)
    target_length_int = int(target_shortest_path_length)
    feasible_node_support = feasible_node_counts_for_shortest_path_length(
        target_shortest_path_length=int(target_length_int),
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested shortest-path query")

    path_nodes = tuple(range(int(target_length_int) + 1))
    graph = nx.path_graph(path_nodes)
    anchor_by_node = {int(node): int(node) for node in path_nodes}

    next_node = int(target_length_int) + 1
    while int(next_node) < int(node_count_int):
        parent = _choose_attachment_parent(
            rng,
            graph=graph,
            topology_profile=str(topology_profile),
        )
        graph.add_node(int(next_node))
        graph.add_edge(int(parent), int(next_node))
        anchor_by_node[int(next_node)] = int(anchor_by_node[int(parent)])
        next_node += 1

    source_node = int(path_nodes[0])
    goal_node = int(path_nodes[-1])
    adjacency = _graph_adjacency_by_node(graph)
    dist_start, count_start = bfs_dist_count_by_adjacency(adjacency, start=int(source_node))
    dist_goal, _ = bfs_dist_count_by_adjacency(adjacency, start=int(goal_node))
    unique_path = reconstruct_unique_shortest_path_by_adjacency(
        adjacency,
        start=int(source_node),
        goal=int(goal_node),
        dist_start=dist_start,
        dist_goal=dist_goal,
    )
    if unique_path is None or tuple(int(node) for node in unique_path) != tuple(int(node) for node in path_nodes):
        raise ValueError("backbone construction failed to preserve the requested unique shortest path")

    extra_edge_candidates = [
        (int(left), int(right))
        for left, right in nx.non_edges(graph)
        if int(anchor_by_node[int(left)]) == int(anchor_by_node[int(right)])
        and not (int(left) in path_nodes and int(right) in path_nodes)
    ]
    rng.shuffle(extra_edge_candidates)
    if str(topology_profile) == "hub_heavy":
        extra_edge_budget = min(3, len(extra_edge_candidates))
    elif str(topology_profile) == "balanced":
        extra_edge_budget = min(2, len(extra_edge_candidates))
    else:
        extra_edge_budget = min(1, len(extra_edge_candidates))

    extra_edges_kept = 0
    for left, right in extra_edge_candidates:
        if int(extra_edges_kept) >= int(extra_edge_budget):
            break
        graph.add_edge(int(left), int(right))
        adjacency = _graph_adjacency_by_node(graph)
        dist_start, count_start = bfs_dist_count_by_adjacency(adjacency, start=int(source_node))
        if int(dist_start.get(int(goal_node), -1)) != int(target_length_int) or int(count_start.get(int(goal_node), 0)) != 1:
            graph.remove_edge(int(left), int(right))
            continue
        dist_goal, _ = bfs_dist_count_by_adjacency(adjacency, start=int(goal_node))
        unique_path = reconstruct_unique_shortest_path_by_adjacency(
            adjacency,
            start=int(source_node),
            goal=int(goal_node),
            dist_start=dist_start,
            dist_goal=dist_goal,
        )
        if unique_path is None or tuple(int(node) for node in unique_path) != tuple(int(node) for node in path_nodes):
            graph.remove_edge(int(left), int(right))
            continue
        extra_edges_kept += 1

    return graph, tuple(int(node) for node in path_nodes), int(extra_edges_kept)

def _sample_unique_shortest_path_digraph(
    rng: random.Random,
    *,
    node_count: int,
    target_shortest_path_length: int,
    topology_profile: str,
) -> Tuple[nx.DiGraph, Tuple[int, ...], int]:
    """Return one directed graph with a unique shortest directed path of the requested length.

    The source-to-goal witness path is realized by one directed backbone chain.
    Off-path nodes attach as incoming or outgoing branches, and optional extra
    off-path edges are only retained when they preserve the unique directed
    shortest path between the backbone endpoints.
    """

    node_count_int = int(node_count)
    target_length_int = int(target_shortest_path_length)
    feasible_node_support = feasible_node_counts_for_shortest_path_length(
        target_shortest_path_length=int(target_length_int),
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested directed shortest-path query")

    path_nodes = tuple(range(int(target_length_int) + 1))
    graph = nx.DiGraph()
    graph.add_nodes_from(path_nodes)
    graph.add_edges_from((int(left), int(right)) for left, right in zip(path_nodes[:-1], path_nodes[1:]))
    anchor_by_node = {int(node): int(node) for node in path_nodes}

    profile = str(topology_profile)
    next_node = int(target_length_int) + 1
    while int(next_node) < int(node_count_int):
        parent = _choose_attachment_parent(
            rng,
            graph=graph,
            topology_profile=profile,
        )
        graph.add_node(int(next_node))
        branch_direction = str(
            rng.choices(
                ("out", "in"),
                weights=(2.0, 1.0) if profile == "hub_heavy" else (1.0, 1.0),
                k=1,
            )[0]
        )
        if branch_direction == "out":
            graph.add_edge(int(parent), int(next_node))
        else:
            graph.add_edge(int(next_node), int(parent))
        anchor_by_node[int(next_node)] = int(anchor_by_node[int(parent)])
        next_node += 1

    source_node = int(path_nodes[0])
    goal_node = int(path_nodes[-1])
    successors = _digraph_successor_adjacency_by_node(graph)
    predecessors = _digraph_predecessor_adjacency_by_node(graph)
    dist_start, count_start = bfs_dist_count_by_adjacency(successors, start=int(source_node))
    dist_goal, _ = bfs_dist_count_by_adjacency(predecessors, start=int(goal_node))
    unique_path = reconstruct_unique_shortest_path_by_adjacency(
        successors,
        start=int(source_node),
        goal=int(goal_node),
        dist_start=dist_start,
        dist_goal=dist_goal,
    )
    if unique_path is None or tuple(int(node) for node in unique_path) != tuple(int(node) for node in path_nodes):
        raise ValueError("directed backbone construction failed to preserve the requested unique shortest path")

    extra_edge_candidates = [
        (int(left), int(right))
        for left, right in nx.non_edges(graph)
        if int(anchor_by_node[int(left)]) == int(anchor_by_node[int(right)])
        and not (int(left) in path_nodes and int(right) in path_nodes)
        and not graph.has_edge(int(right), int(left))
    ]
    rng.shuffle(extra_edge_candidates)
    if profile == "hub_heavy":
        extra_edge_budget = min(1, len(extra_edge_candidates))
    elif profile == "balanced":
        extra_edge_budget = min(1, len(extra_edge_candidates))
    else:
        extra_edge_budget = 0

    extra_edges_kept = 0
    for left, right in extra_edge_candidates:
        if int(extra_edges_kept) >= int(extra_edge_budget):
            break
        graph.add_edge(int(left), int(right))
        successors = _digraph_successor_adjacency_by_node(graph)
        predecessors = _digraph_predecessor_adjacency_by_node(graph)
        dist_start, count_start = bfs_dist_count_by_adjacency(successors, start=int(source_node))
        if int(dist_start.get(int(goal_node), -1)) != int(target_length_int) or int(count_start.get(int(goal_node), 0)) != 1:
            graph.remove_edge(int(left), int(right))
            continue
        dist_goal, _ = bfs_dist_count_by_adjacency(predecessors, start=int(goal_node))
        unique_path = reconstruct_unique_shortest_path_by_adjacency(
            successors,
            start=int(source_node),
            goal=int(goal_node),
            dist_start=dist_start,
            dist_goal=dist_goal,
        )
        if unique_path is None or tuple(int(node) for node in unique_path) != tuple(int(node) for node in path_nodes):
            graph.remove_edge(int(left), int(right))
            continue
        extra_edges_kept += 1

    return graph, tuple(int(node) for node in path_nodes), int(extra_edges_kept)

def _unique_longest_path_in_dag(graph: nx.DiGraph) -> Tuple[int, Tuple[int, ...]] | None:
    """Return the unique longest directed path in a DAG, or None when tied."""

    if not nx.is_directed_acyclic_graph(graph):
        return None
    topo_order = tuple(int(node) for node in nx.topological_sort(graph))
    if not topo_order:
        return None
    best_length = {int(node): 0 for node in topo_order}
    best_count = {int(node): 1 for node in topo_order}
    best_path = {int(node): (int(node),) for node in topo_order}
    for source in topo_order:
        for target in sorted((int(node) for node in graph.successors(int(source)))):
            candidate_length = int(best_length[int(source)]) + 1
            if int(candidate_length) > int(best_length[int(target)]):
                best_length[int(target)] = int(candidate_length)
                best_count[int(target)] = int(best_count[int(source)])
                best_path[int(target)] = (*best_path[int(source)], int(target))
            elif int(candidate_length) == int(best_length[int(target)]):
                best_count[int(target)] += int(best_count[int(source)])

    max_length = max(int(value) for value in best_length.values())
    endpoints = [int(node) for node in topo_order if int(best_length[int(node)]) == int(max_length)]
    max_path_count = sum(int(best_count[int(node)]) for node in endpoints)
    if int(max_path_count) != 1:
        return None
    endpoint = next(int(node) for node in endpoints if int(best_count[int(node)]) == 1)
    return int(max_length), tuple(int(node) for node in best_path[int(endpoint)])

def _try_add_dag_edge_preserving_unique_longest_path(
    graph: nx.DiGraph,
    *,
    source: int,
    target: int,
    path_nodes: Tuple[int, ...],
    target_longest_path_length: int,
) -> bool:
    """Add one edge if the requested unique longest path remains unchanged."""

    source_int = int(source)
    target_int = int(target)
    if int(source_int) == int(target_int):
        return False
    if graph.has_edge(int(source_int), int(target_int)) or graph.has_edge(int(target_int), int(source_int)):
        return False
    graph.add_edge(int(source_int), int(target_int))
    observed = _unique_longest_path_in_dag(graph)
    if observed is None:
        graph.remove_edge(int(source_int), int(target_int))
        return False
    observed_length, observed_path = observed
    if int(observed_length) != int(target_longest_path_length) or tuple(int(node) for node in observed_path) != tuple(
        int(node) for node in path_nodes
    ):
        graph.remove_edge(int(source_int), int(target_int))
        return False
    return True

def _sample_unique_longest_path_dag(
    rng: random.Random,
    *,
    node_count: int,
    target_longest_path_length: int,
    topology_profile: str,
) -> Tuple[nx.DiGraph, Tuple[int, ...], int, int]:
    """Return a DAG with one unique global longest directed path."""

    node_count_int = int(node_count)
    target_length_int = int(target_longest_path_length)
    feasible_node_support = feasible_node_counts_for_longest_path_length(
        target_longest_path_length=int(target_length_int),
        node_count_min=int(node_count_int),
        node_count_max=int(node_count_int),
    )
    if int(node_count_int) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested longest-path query")

    path_nodes = tuple(range(int(target_length_int) + 1))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(node_count_int)))
    graph.add_edges_from((int(left), int(right)) for left, right in zip(path_nodes[:-1], path_nodes[1:]))

    observed = _unique_longest_path_in_dag(graph)
    if observed is None or int(observed[0]) != int(target_length_int) or tuple(observed[1]) != tuple(path_nodes):
        raise ValueError("longest-path backbone construction failed")

    off_path_nodes = tuple(range(int(target_length_int) + 1, int(node_count_int)))
    attached_off_path_nodes: set[int] = set()
    all_nodes = tuple(range(int(node_count_int)))
    for off_path_node in off_path_nodes:
        existing_nodes = [int(node) for node in all_nodes if int(node) != int(off_path_node)]
        candidates = [
            (int(source), int(target))
            for neighbor in existing_nodes
            for source, target in ((int(neighbor), int(off_path_node)), (int(off_path_node), int(neighbor)))
        ]
        rng.shuffle(candidates)
        attached = False
        for source, target in candidates:
            if _try_add_dag_edge_preserving_unique_longest_path(
                graph,
                source=int(source),
                target=int(target),
                path_nodes=path_nodes,
                target_longest_path_length=int(target_length_int),
            ):
                attached_off_path_nodes.add(int(off_path_node))
                attached = True
                break
        if not bool(attached):
            raise ValueError("failed to attach off-path node while preserving unique longest path")

    extra_edge_candidates = [
        (int(source), int(target))
        for source in all_nodes
        for target in all_nodes
        if int(source) != int(target)
        and not graph.has_edge(int(source), int(target))
        and not graph.has_edge(int(target), int(source))
    ]
    rng.shuffle(extra_edge_candidates)
    if str(topology_profile) == "hub_heavy":
        extra_edge_budget = min(4, len(extra_edge_candidates))
    elif str(topology_profile) == "balanced":
        extra_edge_budget = min(2, len(extra_edge_candidates))
    else:
        extra_edge_budget = min(1, len(extra_edge_candidates))

    extra_edges_kept = 0
    for source, target in extra_edge_candidates:
        if int(extra_edges_kept) >= int(extra_edge_budget):
            break
        if _try_add_dag_edge_preserving_unique_longest_path(
            graph,
            source=int(source),
            target=int(target),
            path_nodes=path_nodes,
            target_longest_path_length=int(target_length_int),
        ):
            if int(source) in off_path_nodes:
                attached_off_path_nodes.add(int(source))
            if int(target) in off_path_nodes:
                attached_off_path_nodes.add(int(target))
            extra_edges_kept += 1

    observed = _unique_longest_path_in_dag(graph)
    if observed is None or int(observed[0]) != int(target_length_int) or tuple(int(node) for node in observed[1]) != tuple(path_nodes):
        raise ValueError("longest-path DAG construction failed final validation")
    return graph, tuple(int(node) for node in path_nodes), int(len(attached_off_path_nodes)), int(extra_edges_kept)

def _sample_unique_topological_order_digraph(
    rng: random.Random,
    *,
    node_count: int,
    topology_profile: str,
) -> Tuple[nx.DiGraph, Tuple[int, ...], int]:
    """Return one DAG whose topological order is unique by construction."""

    node_count_int = int(node_count)
    if int(node_count_int) < 2:
        raise ValueError("topological-order sampling requires at least two nodes")

    order_nodes = tuple(range(int(node_count_int)))
    graph = nx.DiGraph()
    graph.add_nodes_from(order_nodes)
    graph.add_edges_from((int(left), int(right)) for left, right in zip(order_nodes[:-1], order_nodes[1:]))

    candidate_edges = [
        (int(left), int(right))
        for left in order_nodes
        for right in order_nodes
        if int(left) < int(right) - 1 and not graph.has_edge(int(left), int(right))
    ]
    profile = str(topology_profile)
    if profile == "hub_heavy":
        ordered_candidates = sorted(candidate_edges, key=lambda pair: (int(pair[0]), -(int(pair[1]) - int(pair[0]))))
        extra_edge_budget = min(len(ordered_candidates), max(2, min(6, int(node_count_int) - 2)))
    elif profile == "low_degree":
        ordered_candidates = sorted(candidate_edges, key=lambda pair: (int(pair[1]) - int(pair[0]), int(pair[0]), int(pair[1])))
        extra_edge_budget = min(len(ordered_candidates), max(1, min(2, int(node_count_int) - 3)))
    else:
        ordered_candidates = list(candidate_edges)
        rng.shuffle(ordered_candidates)
        extra_edge_budget = min(len(ordered_candidates), max(1, min(4, int(node_count_int) - 2)))

    extra_edges_kept = 0
    for left, right in ordered_candidates:
        if int(extra_edges_kept) >= int(extra_edge_budget):
            break
        graph.add_edge(int(left), int(right))
        extra_edges_kept += 1

    return graph, tuple(int(node) for node in order_nodes), int(extra_edges_kept)

def sample_unique_cycle_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_cycle_size: int,
    topology_profile: str,
    label_variant: str,
) -> GraphUniqueCycleSample:
    """Construct one connected unicyclic graph for unique-cycle-size queries."""

    feasible_node_support = feasible_node_counts_for_unique_cycle_size(
        target_cycle_size=int(target_cycle_size),
        node_count_min=int(node_count),
        node_count_max=int(node_count),
    )
    if int(node_count) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested unique-cycle query")

    graph = _sample_unicyclic_graph(
        rng,
        node_count=int(node_count),
        cycle_size=int(target_cycle_size),
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    cycle_basis = nx.cycle_basis(graph)
    if len(cycle_basis) != 1:
        raise ValueError("unique-cycle sampler failed to produce exactly one cycle")
    cycle_nodes = tuple(int(node) for node in cycle_basis[0])
    target_labels = tuple(sorted((str(label_by_node[int(node)]) for node in cycle_nodes), key=graph_label_sort_key))
    return GraphUniqueCycleSample(
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
        target_cycle_size=int(target_cycle_size),
        attachment_count=int(node_count) - int(target_cycle_size),
    )


def sample_largest_chordless_cycle_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_cycle_size: int,
    topology_profile: str,
    label_variant: str,
) -> GraphLargestChordlessCycleSample:
    """Construct one connected graph for largest-chordless-cycle-size queries."""

    graph, target_cycle_nodes, chordless_cycles, attachment_count, extra_edge_count = _sample_largest_chordless_cycle_graph(
        rng,
        node_count=int(node_count),
        target_cycle_size=int(target_cycle_size),
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    target_labels = tuple(str(label_by_node[int(node)]) for node in target_cycle_nodes)
    chordless_cycle_labels = tuple(
        tuple(str(label_by_node[int(node)]) for node in cycle)
        for cycle in chordless_cycles
    )
    return GraphLargestChordlessCycleSample(
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
        target_cycle_size=int(target_cycle_size),
        chordless_cycle_sizes=tuple(int(len(cycle)) for cycle in chordless_cycles),
        chordless_cycle_labels=tuple(tuple(str(label) for label in cycle) for cycle in chordless_cycle_labels),
        attachment_count=int(attachment_count),
        extra_edge_count=int(extra_edge_count),
    )


def sample_hamiltonian_cycle_neighbor_graph(
    rng: random.Random,
    *,
    query_id: str,
    node_count: int,
    topology_profile: str,
    label_variant: str,
) -> GraphHamiltonianCycleNeighborSample:
    """Construct one connected graph for Hamiltonian-cycle neighbor queries."""

    graph, cycle_nodes, hamiltonian_cycles, extra_edge_count = _sample_unique_hamiltonian_cycle_graph(
        rng,
        node_count=int(node_count),
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=False,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    cycle_labels = tuple(str(label_by_node[int(node)]) for node in cycle_nodes)
    if len(cycle_labels) < 3:
        raise ValueError("Hamiltonian-cycle neighbor queries require at least three cycle nodes")
    orientation_start_label = str(cycle_labels[0])
    orientation_next_label = str(cycle_labels[1])
    orientation_final_label = str(cycle_labels[-1])
    if str(query_id) == "next_in_hamiltonian_cycle_label":
        query_label = str(orientation_start_label)
        answer_label = str(orientation_next_label)
        relation_mode = "next"
    elif str(query_id) == "previous_in_hamiltonian_cycle_label":
        query_label = str(orientation_final_label)
        answer_label = str(cycle_labels[-2])
        relation_mode = "previous"
    else:
        raise ValueError(f"unsupported Hamiltonian-cycle query id: {query_id}")

    return GraphHamiltonianCycleNeighborSample(
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
        target_labels=tuple(str(label) for label in cycle_labels),
        query_label=str(query_label),
        answer_label=str(answer_label),
        relation_mode=str(relation_mode),
        orientation_start_label=str(orientation_start_label),
        orientation_next_label=str(orientation_next_label),
        orientation_final_label=str(orientation_final_label),
        hamiltonian_cycle_count=int(len(hamiltonian_cycles)),
        extra_edge_count=int(extra_edge_count),
    )


def sample_shortest_path_length_graph(
    rng: random.Random,
    *,
    query_id: str,
    node_count: int,
    target_shortest_path_length: int,
    topology_profile: str,
    label_variant: str,
) -> GraphShortestPathSample:
    """Construct one connected graph with a unique shortest path of the requested length."""

    feasible_node_support = feasible_node_counts_for_shortest_path_length(
        target_shortest_path_length=int(target_shortest_path_length),
        node_count_min=int(node_count),
        node_count_max=int(node_count),
    )
    if int(node_count) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested shortest-path query")

    graph_directionality = str(graph_directionality_for_query_id(str(query_id)))
    if graph_directionality == "directed":
        graph, path_nodes, extra_edge_count = _sample_unique_shortest_path_digraph(
            rng,
            node_count=int(node_count),
            target_shortest_path_length=int(target_shortest_path_length),
            topology_profile=str(topology_profile),
        )
    else:
        graph, path_nodes, extra_edge_count = _sample_unique_shortest_path_graph(
            rng,
            node_count=int(node_count),
            target_shortest_path_length=int(target_shortest_path_length),
            topology_profile=str(topology_profile),
        )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(graph_directionality == "directed"),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )

    source_node = int(path_nodes[0])
    goal_node = int(path_nodes[-1])
    source_label = str(label_by_node[int(source_node)])
    goal_label = str(label_by_node[int(goal_node)])
    target_labels = tuple(str(label_by_node[int(node)]) for node in path_nodes)
    return GraphShortestPathSample(
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
        source_label=str(source_label),
        goal_label=str(goal_label),
        target_labels=tuple(str(label) for label in target_labels),
        target_shortest_path_length=int(target_shortest_path_length),
        attachment_count=max(0, int(node_count) - len(path_nodes)),
        extra_edge_count=int(extra_edge_count),
    )

def sample_longest_path_length_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_longest_path_length: int,
    topology_profile: str,
    label_variant: str,
) -> GraphLongestPathSample:
    """Construct one DAG with a unique global longest directed path."""

    feasible_node_support = feasible_node_counts_for_longest_path_length(
        target_longest_path_length=int(target_longest_path_length),
        node_count_min=int(node_count),
        node_count_max=int(node_count),
    )
    if int(node_count) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested longest-path query")

    graph, path_nodes, attachment_count, extra_edge_count = _sample_unique_longest_path_dag(
        rng,
        node_count=int(node_count),
        target_longest_path_length=int(target_longest_path_length),
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=True,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )

    source_node = int(path_nodes[0])
    goal_node = int(path_nodes[-1])
    source_label = str(label_by_node[int(source_node)])
    goal_label = str(label_by_node[int(goal_node)])
    target_labels = tuple(str(label_by_node[int(node)]) for node in path_nodes)
    return GraphLongestPathSample(
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
        source_label=str(source_label),
        goal_label=str(goal_label),
        target_labels=tuple(str(label) for label in target_labels),
        target_longest_path_length=int(target_longest_path_length),
        attachment_count=int(attachment_count),
        extra_edge_count=int(extra_edge_count),
    )

def sample_topological_position_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_position: int,
    topology_profile: str,
    label_variant: str,
) -> GraphTopologicalOrderSample:
    """Construct one directed DAG with a unique topological order."""

    feasible_node_support = feasible_node_counts_for_topological_position(
        target_position=int(target_position),
        node_count_min=int(node_count),
        node_count_max=int(node_count),
    )
    if int(node_count) not in feasible_node_support:
        raise ValueError("node_count is outside feasible support for the requested topological-position query")

    graph, order_nodes, extra_edge_count = _sample_unique_topological_order_digraph(
        rng,
        node_count=int(node_count),
        topology_profile=str(topology_profile),
    )
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=True,
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    query_node = int(order_nodes[int(target_position) - 1])
    target_labels = tuple(str(label_by_node[int(node)]) for node in order_nodes)
    return GraphTopologicalOrderSample(
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
        query_label=str(label_by_node[int(query_node)]),
        target_labels=tuple(str(label) for label in target_labels),
        target_position=int(target_position),
        extra_edge_count=int(extra_edge_count),
    )


__all__ = [
    '_add_random_non_edges',
    '_random_bounded_positive_composition',
    '_random_positive_composition',
    '_random_tree_graph',
    'sample_hamiltonian_cycle_neighbor_graph',
    'sample_largest_chordless_cycle_graph',
    'sample_longest_path_length_graph',
    'sample_shortest_path_length_graph',
    'sample_topological_position_graph',
    'sample_unique_cycle_graph',
]

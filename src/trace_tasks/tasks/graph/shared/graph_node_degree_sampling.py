"""Shared graph-domain named-node and extreme-degree samplers."""

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

def _sample_named_node_undirected_degree_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_degree: int,
    max_degree: int,
    topology_profile: str,
    target_node: int,
) -> nx.Graph:
    """Construct one simple graph where ``target_node`` has exact degree."""

    node_count_int = int(node_count)
    target_degree_int = int(target_degree)
    max_degree_int = max(0, int(max_degree))
    if int(target_degree_int) < 0 or int(target_degree_int) > min(int(max_degree_int), int(node_count_int) - 1):
        raise ValueError("target_degree is infeasible for named-node undirected degree sampling")

    graph = nx.Graph()
    graph.add_nodes_from(range(int(node_count_int)))
    other_nodes = [int(node) for node in range(int(node_count_int)) if int(node) != int(target_node)]
    target_neighbors = rng.sample(other_nodes, int(target_degree_int))
    for neighbor in target_neighbors:
        graph.add_edge(int(target_node), int(neighbor))

    shuffled = list(other_nodes)
    rng.shuffle(shuffled)
    for left, right in zip(shuffled, shuffled[1:]):
        if int(graph.degree(int(left))) >= int(max_degree_int) or int(graph.degree(int(right))) >= int(max_degree_int):
            continue
        graph.add_edge(int(left), int(right))

    _add_random_undirected_distractor_edges(
        graph,
        rng,
        nodes=other_nodes,
        extra_edges=_profile_extra_edge_budget(
            node_count=int(node_count_int),
            topology_profile=str(topology_profile),
            directed=False,
        ),
        max_degree=int(max_degree_int),
    )
    return graph

def _sample_named_node_directed_degree_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_degree: int,
    max_degree: int,
    topology_profile: str,
    degree_mode: str,
    target_node: int,
) -> nx.DiGraph:
    """Construct one simple digraph where ``target_node`` has exact queried degree."""

    node_count_int = int(node_count)
    target_degree_int = int(target_degree)
    max_degree_int = max(0, int(max_degree))
    mode = str(degree_mode)
    if mode not in SUPPORTED_NAMED_NODE_DIRECTED_DEGREE_MODES:
        raise ValueError(f"unsupported directed named-node degree mode: {degree_mode}")
    if int(target_degree_int) < 0 or int(target_degree_int) > min(int(max_degree_int), int(node_count_int) - 1):
        raise ValueError("target_degree is infeasible for named-node directed degree sampling")

    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(node_count_int)))
    other_nodes = [int(node) for node in range(int(node_count_int)) if int(node) != int(target_node)]

    if mode == "in_degree":
        incoming_nodes = rng.sample(other_nodes, int(target_degree_int))
        for source in incoming_nodes:
            graph.add_edge(int(source), int(target_node))
        outgoing_candidates = [int(node) for node in other_nodes if int(node) not in set(incoming_nodes)]
        max_outgoing = min(len(outgoing_candidates), 2, int(max_degree_int) - int(graph.out_degree(int(target_node))))
        outgoing_count = int(rng.randrange(max(0, int(max_outgoing)) + 1))
        for target in rng.sample(outgoing_candidates, int(outgoing_count)):
            _add_directed_edge_without_reciprocal(
                graph,
                source=int(target_node),
                target=int(target),
                max_degree=int(max_degree_int),
            )
    elif mode == "out_degree":
        outgoing_nodes = rng.sample(other_nodes, int(target_degree_int))
        for target in outgoing_nodes:
            graph.add_edge(int(target_node), int(target))
        incoming_candidates = [int(node) for node in other_nodes if int(node) not in set(outgoing_nodes)]
        max_incoming = min(len(incoming_candidates), 2, int(max_degree_int) - int(graph.in_degree(int(target_node))))
        incoming_count = int(rng.randrange(max(0, int(max_incoming)) + 1))
        for source in rng.sample(incoming_candidates, int(incoming_count)):
            _add_directed_edge_without_reciprocal(
                graph,
                source=int(source),
                target=int(target_node),
                max_degree=int(max_degree_int),
            )
    else:
        incident_nodes = rng.sample(other_nodes, int(target_degree_int))
        rng.shuffle(incident_nodes)
        incoming_quota = int(target_degree_int // 2)
        if int(target_degree_int) > 0:
            incoming_quota = int(rng.randint(0, int(target_degree_int)))
        incoming_quota = min(int(incoming_quota), int(max_degree_int))
        outgoing_quota = int(target_degree_int) - int(incoming_quota)
        if int(outgoing_quota) > int(max_degree_int):
            outgoing_quota = int(max_degree_int)
            incoming_quota = int(target_degree_int) - int(outgoing_quota)
        for index, other_node in enumerate(incident_nodes):
            if int(index) < int(incoming_quota):
                graph.add_edge(int(other_node), int(target_node))
            else:
                graph.add_edge(int(target_node), int(other_node))

    shuffled = list(other_nodes)
    rng.shuffle(shuffled)
    for left, right in zip(shuffled, shuffled[1:]):
        if rng.random() < 0.5:
            source, target = int(left), int(right)
        else:
            source, target = int(right), int(left)
        _add_directed_edge_without_reciprocal(
            graph,
            source=int(source),
            target=int(target),
            max_degree=int(max_degree_int),
        )

    _add_random_directed_distractor_edges(
        graph,
        rng,
        nodes=other_nodes,
        extra_edges=_profile_extra_edge_budget(
            node_count=int(node_count_int),
            topology_profile=str(topology_profile),
            directed=True,
        ),
        max_degree=int(max_degree_int),
    )
    return graph

@lru_cache(maxsize=128)
def feasible_node_counts_for_named_node_degree_value(
    *,
    graph_directionality: str,
    degree_mode: str,
    target_degree: int,
    node_count_min: int,
    node_count_max: int,
    max_degree: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize a named-node degree-value query."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_NAMED_NODE_DEGREE_DIRECTIONS:
        return ()
    mode = str(degree_mode)
    if directionality == "undirected" and mode != "degree":
        return ()
    if directionality == "directed" and mode not in SUPPORTED_NAMED_NODE_DIRECTED_DEGREE_MODES:
        return ()
    target_degree_int = int(target_degree)
    max_degree_int = int(max_degree)
    if int(target_degree_int) < 0 or int(target_degree_int) > int(max_degree_int):
        return ()
    feasible = []
    for node_count in range(int(node_count_min), int(node_count_max) + 1):
        if int(target_degree_int) <= int(node_count) - 1:
            feasible.append(int(node_count))
    return tuple(int(value) for value in feasible)

def sample_named_node_degree_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    degree_mode: str,
    node_count: int,
    target_degree: int,
    max_degree: int,
    topology_profile: str,
    label_variant: str,
) -> GraphNamedNodeDegreeSample:
    """Construct one labeled graph for a queried node's degree value."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_NAMED_NODE_DEGREE_DIRECTIONS:
        raise ValueError(f"unsupported graph directionality: {graph_directionality}")
    mode = str(degree_mode)
    if directionality == "undirected":
        if mode != "degree":
            raise ValueError("undirected named-node degree queries require degree_mode=degree")
    elif mode not in SUPPORTED_NAMED_NODE_DIRECTED_DEGREE_MODES:
        raise ValueError(f"unsupported directed named-node degree mode: {degree_mode}")

    node_count_int = int(node_count)
    target_node = int(rng.randrange(int(node_count_int)))
    if directionality == "directed":
        graph = _sample_named_node_directed_degree_graph(
            rng,
            node_count=int(node_count_int),
            target_degree=int(target_degree),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
            degree_mode=str(mode),
            target_node=int(target_node),
        )
        directed = True
    else:
        graph = _sample_named_node_undirected_degree_graph(
            rng,
            node_count=int(node_count_int),
            target_degree=int(target_degree),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
            target_node=int(target_node),
        )
        directed = False

    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directed),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    query_label = str(label_by_node[int(target_node)])
    if bool(directed):
        in_degree_sequence = tuple(int(graph.in_degree(int(node))) for node in graph.nodes())
        out_degree_sequence = tuple(int(graph.out_degree(int(node))) for node in graph.nodes())
        degree_sequence = tuple(
            int(graph.in_degree(int(node))) + int(graph.out_degree(int(node)))
            for node in graph.nodes()
        )
        total_degrees_by_label = {
            str(label_by_node[int(node)]): int(graph.in_degree(int(node))) + int(graph.out_degree(int(node)))
            for node in graph.nodes()
        }
        if mode == "in_degree":
            target_edges = tuple(
                (str(label_by_node[int(source)]), str(label_by_node[int(target_node)]))
                for source, _ in graph.in_edges(int(target_node))
            )
        elif mode == "out_degree":
            target_edges = tuple(
                (str(label_by_node[int(target_node)]), str(label_by_node[int(target)]))
                for _, target in graph.out_edges(int(target_node))
            )
        else:
            target_edges = tuple(
                [
                    (str(label_by_node[int(source)]), str(label_by_node[int(target_node)]))
                    for source, _ in graph.in_edges(int(target_node))
                ]
                + [
                    (str(label_by_node[int(target_node)]), str(label_by_node[int(target)]))
                    for _, target in graph.out_edges(int(target_node))
                ]
            )
        sorted_target_edges = sort_graph_edge_labels(target_edges, directed=True)
    else:
        degree_sequence = tuple(int(graph.degree(int(node))) for node in graph.nodes())
        in_degree_sequence = tuple(int(value) for value in degree_sequence)
        out_degree_sequence = tuple(int(value) for value in degree_sequence)
        total_degrees_by_label = {
            str(label_by_node[int(node)]): int(graph.degree(int(node)))
            for node in graph.nodes()
        }
        sorted_target_edges = sort_graph_edge_labels(
            tuple(
                (str(label_by_node[int(target_node)]), str(label_by_node[int(neighbor)]))
                for neighbor in graph.neighbors(int(target_node))
            ),
            directed=False,
        )

    if int(len(sorted_target_edges)) != int(target_degree):
        raise ValueError("named-node degree sampler produced the wrong annotation-edge count")

    return GraphNamedNodeDegreeSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in total_degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        query_label=str(query_label),
        target_edges=tuple((str(left), str(right)) for left, right in sorted_target_edges),
        target_degree=int(target_degree),
        degree_mode=str(mode),
        degree_sequence=tuple(int(value) for value in degree_sequence),
        in_degree_sequence=tuple(int(value) for value in in_degree_sequence),
        out_degree_sequence=tuple(int(value) for value in out_degree_sequence),
        total_degrees_by_label={str(key): int(value) for key, value in total_degrees_by_label.items()},
    )

def _add_directed_edge_with_total_cap(
    graph: nx.DiGraph,
    *,
    source: int,
    target: int,
    total_degree_cap: int,
) -> bool:
    """Add one directed edge when both endpoints stay within a total-degree cap."""

    source_int = int(source)
    target_int = int(target)
    cap_int = int(total_degree_cap)
    if int(source_int) == int(target_int):
        return False
    if graph.has_edge(int(source_int), int(target_int)) or graph.has_edge(int(target_int), int(source_int)):
        return False
    source_total = int(graph.in_degree(int(source_int))) + int(graph.out_degree(int(source_int)))
    target_total = int(graph.in_degree(int(target_int))) + int(graph.out_degree(int(target_int)))
    if int(source_total) >= int(cap_int) or int(target_total) >= int(cap_int):
        return False
    graph.add_edge(int(source_int), int(target_int))
    return True

def _sample_extreme_undirected_degree_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_degree: int,
    max_degree: int,
    topology_profile: str,
    extremum_mode: str,
) -> nx.Graph:
    """Construct one simple graph with the requested undirected extreme degree."""

    node_count_int = int(node_count)
    target_degree_int = int(target_degree)
    max_degree_int = max(0, int(max_degree))
    if int(target_degree_int) < 0 or int(target_degree_int) > min(int(max_degree_int), int(node_count_int) - 1):
        raise ValueError("target_degree is infeasible for undirected extreme-degree sampling")

    mode = str(extremum_mode)
    if mode == "max":
        graph = nx.Graph()
        graph.add_nodes_from(range(int(node_count_int)))
        if int(target_degree_int) == 0:
            return graph
        target_node = int(rng.randrange(int(node_count_int)))
        other_nodes = [int(node) for node in range(int(node_count_int)) if int(node) != int(target_node)]
        for neighbor in rng.sample(other_nodes, int(target_degree_int)):
            graph.add_edge(int(target_node), int(neighbor))
        _add_random_undirected_distractor_edges(
            graph,
            rng,
            nodes=tuple(range(int(node_count_int))),
            extra_edges=_profile_extra_edge_budget(
                node_count=int(node_count_int),
                topology_profile=str(topology_profile),
                directed=False,
            ),
            max_degree=int(target_degree_int),
        )
        return graph

    if mode != "min":
        raise ValueError(f"unsupported extreme degree mode: {extremum_mode}")

    if int(target_degree_int) == 0:
        graph = nx.Graph()
        graph.add_nodes_from(range(int(node_count_int)))
        protected = int(rng.randrange(int(node_count_int)))
        other_nodes = [int(node) for node in range(int(node_count_int)) if int(node) != int(protected)]
        shuffled = list(other_nodes)
        rng.shuffle(shuffled)
        for left, right in zip(shuffled, shuffled[1:]):
            graph.add_edge(int(left), int(right))
        _add_random_undirected_distractor_edges(
            graph,
            rng,
            nodes=other_nodes,
            extra_edges=_profile_extra_edge_budget(
                node_count=int(node_count_int),
                topology_profile=str(topology_profile),
                directed=False,
            ),
            max_degree=min(int(max_degree_int), int(node_count_int) - 1),
        )
        return graph

    for _ in range(256):
        graph = nx.Graph()
        graph.add_nodes_from(range(int(node_count_int)))
        steps = 0
        max_steps = max(32, int(node_count_int * node_count_int * 4))
        while min((int(degree) for _, degree in graph.degree()), default=0) < int(target_degree_int):
            deficient = [int(node) for node, degree in graph.degree() if int(degree) < int(target_degree_int)]
            if not deficient:
                break
            left = int(rng.choice(deficient))
            candidates = [
                int(node)
                for node in range(int(node_count_int))
                if int(node) != int(left)
                and not graph.has_edge(int(left), int(node))
                and int(graph.degree(int(node))) < min(int(max_degree_int), int(node_count_int) - 1)
            ]
            if not candidates:
                break
            right = int(rng.choice(candidates))
            graph.add_edge(int(left), int(right))
            steps += 1
            if int(steps) > int(max_steps):
                break
        degrees = [int(degree) for _, degree in graph.degree()]
        if not degrees or min(degrees) != int(target_degree_int):
            continue
        protected_nodes = {int(node) for node, degree in graph.degree() if int(degree) == int(target_degree_int)}
        candidates = [
            (int(left), int(right))
            for left, right in nx.non_edges(graph)
            if int(left) not in protected_nodes
            and int(right) not in protected_nodes
            and int(graph.degree(int(left))) < int(max_degree_int)
            and int(graph.degree(int(right))) < int(max_degree_int)
        ]
        rng.shuffle(candidates)
        extra_budget = _profile_extra_edge_budget(
            node_count=int(node_count_int),
            topology_profile=str(topology_profile),
            directed=False,
        )
        for left, right in candidates[: int(extra_budget)]:
            graph.add_edge(int(left), int(right))
        if min((int(degree) for _, degree in graph.degree()), default=0) == int(target_degree_int):
            return graph
    raise ValueError("failed to sample an undirected graph for the requested min-degree value")

def _sample_extreme_directed_degree_graph(
    rng: random.Random,
    *,
    node_count: int,
    target_degree: int,
    max_degree: int,
    topology_profile: str,
    degree_mode: str,
    extremum_mode: str,
) -> nx.DiGraph:
    """Construct one simple digraph with the requested directed extreme degree."""

    node_count_int = int(node_count)
    target_degree_int = int(target_degree)
    max_degree_int = max(0, int(max_degree))
    degree_mode_text = str(degree_mode)
    extremum_text = str(extremum_mode)
    if degree_mode_text not in SUPPORTED_EXTREME_DEGREE_DIRECTED_MODES:
        raise ValueError(f"unsupported directed extreme-degree mode: {degree_mode}")
    if int(target_degree_int) < 0 or int(target_degree_int) > min(int(max_degree_int), int(node_count_int) - 1):
        raise ValueError("target_degree is infeasible for directed extreme-degree sampling")

    if extremum_text == "max":
        graph = nx.DiGraph()
        graph.add_nodes_from(range(int(node_count_int)))
        if int(target_degree_int) == 0:
            return graph
        target_node = int(rng.randrange(int(node_count_int)))
        other_nodes = [int(node) for node in range(int(node_count_int)) if int(node) != int(target_node)]
        if degree_mode_text == "in_degree":
            for source in rng.sample(other_nodes, int(target_degree_int)):
                graph.add_edge(int(source), int(target_node))
            _add_random_directed_distractor_edges(
                graph,
                rng,
                nodes=tuple(range(int(node_count_int))),
                extra_edges=_profile_extra_edge_budget(
                    node_count=int(node_count_int),
                    topology_profile=str(topology_profile),
                    directed=True,
                ),
                max_degree=max(1, int(target_degree_int)),
            )
            while max((int(degree) for _, degree in graph.in_degree()), default=0) > int(target_degree_int):
                graph.remove_edge(*next(iter(graph.edges())))
        elif degree_mode_text == "out_degree":
            for target in rng.sample(other_nodes, int(target_degree_int)):
                graph.add_edge(int(target_node), int(target))
            _add_random_directed_distractor_edges(
                graph,
                rng,
                nodes=tuple(range(int(node_count_int))),
                extra_edges=_profile_extra_edge_budget(
                    node_count=int(node_count_int),
                    topology_profile=str(topology_profile),
                    directed=True,
                ),
                max_degree=max(1, int(target_degree_int)),
            )
            while max((int(degree) for _, degree in graph.out_degree()), default=0) > int(target_degree_int):
                graph.remove_edge(*next(iter(graph.edges())))
        else:
            incident_nodes = rng.sample(other_nodes, int(target_degree_int))
            rng.shuffle(incident_nodes)
            incoming_count = int(rng.randint(0, int(target_degree_int)))
            for index, other_node in enumerate(incident_nodes):
                if int(index) < int(incoming_count):
                    graph.add_edge(int(other_node), int(target_node))
                else:
                    graph.add_edge(int(target_node), int(other_node))
            candidates = [
                (int(left), int(right))
                for left in range(int(node_count_int))
                for right in range(int(node_count_int))
                if int(left) != int(right)
            ]
            rng.shuffle(candidates)
            added = 0
            extra_budget = _profile_extra_edge_budget(
                node_count=int(node_count_int),
                topology_profile=str(topology_profile),
                directed=True,
            )
            for source, target in candidates:
                if int(added) >= int(extra_budget):
                    break
                if _add_directed_edge_with_total_cap(
                    graph,
                    source=int(source),
                    target=int(target),
                    total_degree_cap=max(1, int(target_degree_int)),
                ):
                    added += 1
        return graph

    if extremum_text != "min":
        raise ValueError(f"unsupported extreme degree mode: {extremum_mode}")

    if int(target_degree_int) == 0:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(int(node_count_int)))
        protected = int(rng.randrange(int(node_count_int)))
        candidates = []
        for source in range(int(node_count_int)):
            for target in range(int(node_count_int)):
                if int(source) == int(target):
                    continue
                if degree_mode_text == "in_degree" and int(target) == int(protected):
                    continue
                if degree_mode_text == "out_degree" and int(source) == int(protected):
                    continue
                if degree_mode_text == "total_degree" and int(protected) in {int(source), int(target)}:
                    continue
                candidates.append((int(source), int(target)))
        rng.shuffle(candidates)
        added = 0
        extra_budget = _profile_extra_edge_budget(
            node_count=int(node_count_int),
            topology_profile=str(topology_profile),
            directed=True,
        )
        for source, target in candidates:
            if int(added) >= int(extra_budget):
                break
            if _add_directed_edge_without_reciprocal(
                graph,
                source=int(source),
                target=int(target),
                max_degree=min(int(max_degree_int), int(node_count_int) - 1),
            ):
                added += 1
        return graph

    for _ in range(256):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(int(node_count_int)))
        steps = 0
        max_steps = max(64, int(node_count_int * node_count_int * 8))
        while True:
            if degree_mode_text == "in_degree":
                values = {int(node): int(graph.in_degree(int(node))) for node in graph.nodes()}
                deficient = [int(node) for node, value in values.items() if int(value) < int(target_degree_int)]
                if not deficient:
                    break
                target = int(rng.choice(deficient))
                sources = [
                    int(node)
                    for node in graph.nodes()
                    if int(node) != int(target)
                    and not graph.has_edge(int(node), int(target))
                    and not graph.has_edge(int(target), int(node))
                ]
                if not sources:
                    break
                graph.add_edge(int(rng.choice(sources)), int(target))
            elif degree_mode_text == "out_degree":
                values = {int(node): int(graph.out_degree(int(node))) for node in graph.nodes()}
                deficient = [int(node) for node, value in values.items() if int(value) < int(target_degree_int)]
                if not deficient:
                    break
                source = int(rng.choice(deficient))
                targets = [
                    int(node)
                    for node in graph.nodes()
                    if int(node) != int(source)
                    and not graph.has_edge(int(source), int(node))
                    and not graph.has_edge(int(node), int(source))
                ]
                if not targets:
                    break
                graph.add_edge(int(source), int(rng.choice(targets)))
            else:
                values = {
                    int(node): int(graph.in_degree(int(node))) + int(graph.out_degree(int(node)))
                    for node in graph.nodes()
                }
                deficient = [int(node) for node, value in values.items() if int(value) < int(target_degree_int)]
                if not deficient:
                    break
                endpoint = int(rng.choice(deficient))
                others = [
                    int(node)
                    for node in graph.nodes()
                    if int(node) != int(endpoint)
                    and not graph.has_edge(int(endpoint), int(node))
                    and not graph.has_edge(int(node), int(endpoint))
                ]
                if not others:
                    break
                other = int(rng.choice(others))
                if rng.random() < 0.5:
                    graph.add_edge(int(endpoint), int(other))
                else:
                    graph.add_edge(int(other), int(endpoint))
            steps += 1
            if int(steps) > int(max_steps):
                break
        if degree_mode_text == "in_degree":
            queried_values = [int(degree) for _, degree in graph.in_degree()]
        elif degree_mode_text == "out_degree":
            queried_values = [int(degree) for _, degree in graph.out_degree()]
        else:
            queried_values = [
                int(graph.in_degree(int(node))) + int(graph.out_degree(int(node)))
                for node in graph.nodes()
            ]
        if queried_values and min(queried_values) == int(target_degree_int):
            return graph
    raise ValueError("failed to sample a directed graph for the requested min-degree value")


def _sample_unique_extreme_random_graph(
    rng: random.Random,
    *,
    node_count: int,
    directed: bool,
    edge_probability: float,
) -> nx.Graph | nx.DiGraph:
    """Sample one simple graph candidate for unique extreme-degree search."""

    node_count_int = int(node_count)
    probability = max(0.0, min(1.0, float(edge_probability)))
    if bool(directed):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(int(node_count_int)))
        for left in range(int(node_count_int)):
            for right in range(int(left) + 1, int(node_count_int)):
                if rng.random() >= probability:
                    continue
                if rng.random() < 0.5:
                    graph.add_edge(int(left), int(right))
                else:
                    graph.add_edge(int(right), int(left))
        return graph

    graph = nx.Graph()
    graph.add_nodes_from(range(int(node_count_int)))
    for left in range(int(node_count_int)):
        for right in range(int(left) + 1, int(node_count_int)):
            if rng.random() < probability:
                graph.add_edge(int(left), int(right))
    return graph


def _unique_extreme_queried_values(
    graph: nx.Graph | nx.DiGraph,
    *,
    directed: bool,
    degree_mode: str,
) -> Dict[int, int]:
    """Return the degree map used by an extreme-degree query."""

    if not bool(directed):
        return {int(node): int(graph.degree(int(node))) for node in graph.nodes()}
    mode = str(degree_mode)
    if mode == "in_degree":
        return {int(node): int(graph.in_degree(int(node))) for node in graph.nodes()}
    if mode == "out_degree":
        return {int(node): int(graph.out_degree(int(node))) for node in graph.nodes()}
    if mode == "total_degree":
        return {
            int(node): int(graph.in_degree(int(node))) + int(graph.out_degree(int(node)))
            for node in graph.nodes()
        }
    raise ValueError(f"unsupported directed degree mode: {degree_mode}")


def _unique_extreme_search_plan(
    *,
    graph_directionality: str,
    degree_mode: str,
    extremum_mode: str,
    target_degree: int,
    requested_node_count: int,
) -> Tuple[Tuple[int, float], ...]:
    """Return deterministic node-count/density candidates for unique extrema."""

    directionality = str(graph_directionality)
    mode = str(degree_mode)
    extremum = str(extremum_mode)
    target = int(target_degree)
    requested = max(5, int(requested_node_count))

    if directionality == "undirected" and extremum == "max":
        defaults = {
            2: ((7, 0.10), (7, 0.18), (8, 0.18), (6, 0.26)),
            3: ((6, 0.34), (7, 0.34), (6, 0.42), (8, 0.26)),
            4: ((5, 0.62), (6, 0.62), (7, 0.50), (6, 0.74)),
            5: ((6, 0.74), (7, 0.62), (6, 0.88), (8, 0.50)),
        }.get(target, ())
    elif directionality == "undirected" and extremum == "min":
        defaults = {
            0: ((7, 0.26), (6, 0.26), (8, 0.18), (7, 0.34)),
            1: ((8, 0.50), (7, 0.50), (8, 0.62), (9, 0.42)),
            2: ((9, 0.50), (8, 0.62), (9, 0.62), (10, 0.50)),
            3: ((7, 0.74), (8, 0.74), (9, 0.74), (8, 0.88)),
        }.get(target, ())
    elif directionality == "directed" and extremum == "max" and mode in {"in_degree", "out_degree"}:
        defaults = {
            1: ((5, 0.10), (6, 0.10), (5, 0.18), (7, 0.10)),
            2: ((5, 0.42), (6, 0.34), (7, 0.26), (5, 0.50)),
            3: ((5, 0.74), (6, 0.62), (5, 0.88), (7, 0.50)),
            4: ((6, 0.88), (7, 0.74), (6, 1.00), (8, 0.74)),
        }.get(target, ())
    elif directionality == "directed" and extremum == "min" and mode in {"in_degree", "out_degree"}:
        defaults = {
            0: ((5, 0.74), (6, 0.74), (5, 0.88), (7, 0.62)),
            1: ((7, 1.00), (8, 0.88), (7, 0.88), (8, 1.00)),
        }.get(target, ())
    else:
        defaults = ()

    candidates = [(int(node_count), float(probability)) for node_count, probability in defaults]
    for probability in (0.18, 0.26, 0.34, 0.42, 0.50, 0.62, 0.74, 0.88):
        candidates.append((int(requested), float(probability)))

    deduped: list[Tuple[int, float]] = []
    seen: set[Tuple[int, float]] = set()
    for node_count, probability in candidates:
        key = (max(5, int(node_count)), round(float(probability), 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((int(key[0]), float(key[1])))
    return tuple(deduped)


@lru_cache(maxsize=128)
def feasible_node_counts_for_extreme_degree_value(
    *,
    graph_directionality: str,
    degree_mode: str,
    extremum_mode: str,
    target_degree: int,
    node_count_min: int,
    node_count_max: int,
    max_degree: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize an extreme degree-value query."""

    directionality = str(graph_directionality)
    mode = str(degree_mode)
    extremum = str(extremum_mode)
    target_degree_int = int(target_degree)
    max_degree_int = int(max_degree)
    if directionality not in SUPPORTED_EXTREME_DEGREE_DIRECTIONS:
        return ()
    if extremum not in SUPPORTED_EXTREME_DEGREE_EXTREMA:
        return ()
    if directionality == "undirected" and mode != "degree":
        return ()
    if directionality == "directed" and mode not in SUPPORTED_EXTREME_DEGREE_DIRECTED_MODES:
        return ()
    if int(target_degree_int) < 0 or int(target_degree_int) > int(max_degree_int):
        return ()

    feasible = []
    for node_count in range(int(node_count_min), int(node_count_max) + 1):
        node_count_int = int(node_count)
        if int(target_degree_int) > int(node_count_int) - 1:
            continue
        if directionality == "directed" and extremum == "min" and mode in {"in_degree", "out_degree"}:
            if int(target_degree_int) > int((node_count_int - 1) // 2):
                continue
        feasible.append(int(node_count_int))
    return tuple(int(value) for value in feasible)


def _extreme_degree_sample_from_graph(
    rng: random.Random,
    *,
    graph: nx.Graph | nx.DiGraph,
    directed: bool,
    degree_mode: str,
    extremum_mode: str,
    target_degree: int,
    topology_profile: str,
    label_variant: str,
) -> GraphExtremeDegreeSample:
    """Finalize one graph as a labeled extreme-degree sample."""

    degree_mode_text = str(degree_mode)
    extremum_text = str(extremum_mode)
    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directed),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    if bool(directed):
        in_degree_sequence = tuple(int(graph.in_degree(int(node))) for node in graph.nodes())
        out_degree_sequence = tuple(int(graph.out_degree(int(node))) for node in graph.nodes())
        degree_sequence = tuple(
            int(graph.in_degree(int(node))) + int(graph.out_degree(int(node)))
            for node in graph.nodes()
        )
        total_degrees_by_label = {
            str(label_by_node[int(node)]): int(graph.in_degree(int(node))) + int(graph.out_degree(int(node)))
            for node in graph.nodes()
        }
        if degree_mode_text == "in_degree":
            queried_degrees_by_label = {str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()}
        elif degree_mode_text == "out_degree":
            queried_degrees_by_label = {str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()}
        else:
            queried_degrees_by_label = {str(key): int(value) for key, value in total_degrees_by_label.items()}
    else:
        degree_sequence = tuple(int(graph.degree(int(node))) for node in graph.nodes())
        in_degree_sequence = tuple(int(value) for value in degree_sequence)
        out_degree_sequence = tuple(int(value) for value in degree_sequence)
        total_degrees_by_label = {
            str(label_by_node[int(node)]): int(graph.degree(int(node)))
            for node in graph.nodes()
        }
        queried_degrees_by_label = {str(key): int(value) for key, value in total_degrees_by_label.items()}
        degree_mode_text = "degree"

    observed_values = list(int(value) for value in queried_degrees_by_label.values())
    if not observed_values:
        raise ValueError("extreme degree graph has no observed degree values")
    observed_extreme = max(observed_values) if extremum_text == "max" else min(observed_values)
    if int(observed_extreme) != int(target_degree):
        raise ValueError("extreme degree sampler produced the wrong target degree")
    target_labels = tuple(
        sorted(
            (str(label) for label, value in queried_degrees_by_label.items() if int(value) == int(target_degree)),
            key=graph_label_sort_key,
        )
    )

    return GraphExtremeDegreeSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in total_degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        target_labels=tuple(str(label) for label in target_labels),
        target_degree=int(target_degree),
        extremum_mode=str(extremum_text),
        degree_mode=str(degree_mode_text),
        degree_sequence=tuple(int(value) for value in degree_sequence),
        in_degree_sequence=tuple(int(value) for value in in_degree_sequence),
        out_degree_sequence=tuple(int(value) for value in out_degree_sequence),
        queried_degrees_by_label={str(key): int(value) for key, value in queried_degrees_by_label.items()},
        total_degrees_by_label={str(key): int(value) for key, value in total_degrees_by_label.items()},
    )


def sample_extreme_degree_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    degree_mode: str,
    extremum_mode: str,
    node_count: int,
    target_degree: int,
    max_degree: int,
    topology_profile: str,
    label_variant: str,
) -> GraphExtremeDegreeSample:
    """Construct one labeled graph whose queried extreme degree equals target_degree."""

    directionality = str(graph_directionality)
    degree_mode_text = str(degree_mode)
    extremum_text = str(extremum_mode)
    if directionality == "undirected":
        graph = _sample_extreme_undirected_degree_graph(
            rng,
            node_count=int(node_count),
            target_degree=int(target_degree),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
            extremum_mode=str(extremum_text),
        )
        directed = False
    elif directionality == "directed":
        graph = _sample_extreme_directed_degree_graph(
            rng,
            node_count=int(node_count),
            target_degree=int(target_degree),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
            degree_mode=str(degree_mode_text),
            extremum_mode=str(extremum_text),
        )
        directed = True
    else:
        raise ValueError(f"unsupported graph directionality: {graph_directionality}")

    return _extreme_degree_sample_from_graph(
        rng,
        graph=graph,
        directed=bool(directed),
        degree_mode=str(degree_mode_text),
        extremum_mode=str(extremum_text),
        target_degree=int(target_degree),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )


def sample_unique_extreme_degree_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    degree_mode: str,
    extremum_mode: str,
    node_count: int,
    target_degree: int,
    max_degree: int,
    topology_profile: str,
    label_variant: str,
    search_attempts: int = 512,
) -> GraphExtremeDegreeSample:
    """Construct a graph where exactly one node attains the queried extreme value."""

    directionality = str(graph_directionality)
    directed = bool(directionality == "directed")
    degree_mode_text = str(degree_mode)
    extremum_text = str(extremum_mode)
    target_degree_int = int(target_degree)
    if directionality not in SUPPORTED_EXTREME_DEGREE_DIRECTIONS:
        raise ValueError(f"unsupported graph directionality: {graph_directionality}")
    if extremum_text not in SUPPORTED_EXTREME_DEGREE_EXTREMA:
        raise ValueError(f"unsupported extreme degree mode: {extremum_mode}")
    if not directed and degree_mode_text != "degree":
        raise ValueError("undirected unique extreme-degree queries require degree mode")
    if directed and degree_mode_text not in SUPPORTED_EXTREME_DEGREE_DIRECTED_MODES:
        raise ValueError(f"unsupported directed degree mode: {degree_mode}")

    max_degree_int = max(0, int(max_degree))
    if int(target_degree_int) < 0 or int(target_degree_int) > int(max_degree_int):
        raise ValueError("target_degree is infeasible for unique extreme-degree sampling")

    candidates = _unique_extreme_search_plan(
        graph_directionality=str(directionality),
        degree_mode=str(degree_mode_text),
        extremum_mode=str(extremum_text),
        target_degree=int(target_degree_int),
        requested_node_count=int(node_count),
    )
    if not candidates:
        raise ValueError("no unique-extreme search plan for requested degree branch")

    attempts_per_candidate = max(32, int(search_attempts) // max(1, len(candidates)))
    last_error: Exception | None = None
    for candidate_node_count, probability in candidates:
        if int(target_degree_int) > int(candidate_node_count) - 1:
            continue
        for _attempt in range(int(attempts_per_candidate)):
            graph = _sample_unique_extreme_random_graph(
                rng,
                node_count=int(candidate_node_count),
                directed=bool(directed),
                edge_probability=float(probability),
            )
            values_by_node = _unique_extreme_queried_values(
                graph,
                directed=bool(directed),
                degree_mode=str(degree_mode_text),
            )
            if not values_by_node:
                continue
            observed_extreme = max(values_by_node.values()) if extremum_text == "max" else min(values_by_node.values())
            if int(observed_extreme) != int(target_degree_int):
                continue
            target_nodes = [node for node, value in values_by_node.items() if int(value) == int(target_degree_int)]
            if len(target_nodes) != 1:
                continue
            try:
                return _extreme_degree_sample_from_graph(
                    rng,
                    graph=graph,
                    directed=bool(directed),
                    degree_mode=str(degree_mode_text),
                    extremum_mode=str(extremum_text),
                    target_degree=int(target_degree_int),
                    topology_profile=str(topology_profile),
                    label_variant=str(label_variant),
                )
            except Exception as exc:
                last_error = exc
                continue
    raise ValueError("failed to sample a unique extreme-degree graph") from last_error


__all__ = [
    'feasible_node_counts_for_extreme_degree_value',
    'feasible_node_counts_for_named_node_degree_value',
    'sample_extreme_degree_graph',
    'sample_unique_extreme_degree_graph',
    'sample_named_node_degree_graph',
]

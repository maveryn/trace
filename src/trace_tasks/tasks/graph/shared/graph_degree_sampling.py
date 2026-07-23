"""Degree-count graph sampler helpers."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Sequence, Tuple

import networkx as nx

from .graph_sample_types import (
    GraphCountSample,
    graph_degree_mode_for_query_id,
    graph_directionality_for_query_id,
    graph_label_sort_key,
)
from .graph_topology_helpers import _build_labeled_graph_topology_sample, _has_reciprocal_edges


def _profile_weight(degree_value: int, *, topology_profile: str) -> float:
    """Return one degree-sampling weight for the requested topology profile."""

    degree = int(degree_value)
    profile = str(topology_profile)
    if profile == "low_degree":
        return float({0: 4.0, 1: 4.0, 2: 3.0, 3: 1.5, 4: 1.0, 5: 0.75}.get(degree, 0.5))
    if profile == "hub_heavy":
        return float({0: 0.8, 1: 1.0, 2: 1.5, 3: 2.5, 4: 3.5, 5: 4.0}.get(degree, 1.0))
    return 1.0


def _draw_degree(
    rng: random.Random,
    *,
    allowed_degrees: Sequence[int],
    topology_profile: str,
) -> int:
    """Draw one degree value under the requested topology profile."""

    candidates = [int(value) for value in allowed_degrees]
    if not candidates:
        raise ValueError("no degree candidates available")
    weights = [_profile_weight(value, topology_profile=str(topology_profile)) for value in candidates]
    return int(rng.choices(candidates, weights=weights, k=1)[0])


def _draw_other_degree(
    rng: random.Random,
    *,
    allowed_degrees: Sequence[int],
    query_degree: int,
    topology_profile: str,
) -> int:
    """Draw one non-query degree under the requested topology profile."""

    candidates = [int(value) for value in allowed_degrees if int(value) != int(query_degree)]
    if not candidates:
        raise ValueError("no non-query degrees available")
    return _draw_degree(rng, allowed_degrees=candidates, topology_profile=str(topology_profile))


def _sample_exact_query_sequence(
    rng: random.Random,
    *,
    node_count: int,
    query_degree: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
) -> Tuple[int, ...] | None:
    """Sample one degree sequence with exactly ``target_count`` query-degree entries."""

    allowed = tuple(range(0, min(int(max_degree), int(node_count) - 1) + 1))
    if int(query_degree) not in allowed:
        return None
    remainder_count = int(node_count) - int(target_count)
    if int(remainder_count) < 0:
        return None
    other_allowed = tuple(int(value) for value in allowed if int(value) != int(query_degree))
    if int(remainder_count) > 0 and not other_allowed:
        return None
    sequence = [int(query_degree)] * int(target_count)
    sequence.extend(
        _draw_other_degree(
            rng,
            allowed_degrees=other_allowed,
            query_degree=int(query_degree),
            topology_profile=str(topology_profile),
        )
        for _ in range(int(remainder_count))
    )
    rng.shuffle(sequence)
    return tuple(int(value) for value in sequence)


def _parity_adjusted_sequence(
    rng: random.Random,
    *,
    node_count: int,
    query_degree: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
) -> Tuple[int, ...] | None:
    """Sample one parity-valid candidate degree sequence with exact query-degree count."""

    allowed = tuple(range(0, min(int(max_degree), int(node_count) - 1) + 1))
    other_allowed = tuple(int(value) for value in allowed if int(value) != int(query_degree))
    sequence = _sample_exact_query_sequence(
        rng,
        node_count=int(node_count),
        query_degree=int(query_degree),
        target_count=int(target_count),
        max_degree=int(max_degree),
        topology_profile=str(topology_profile),
    )
    if not sequence:
        return None
    if sum(sequence) % 2 == 0:
        return tuple(int(value) for value in sequence)

    mutable = list(int(value) for value in sequence)
    adjustable_indices = list(range(int(target_count), len(mutable)))
    rng.shuffle(adjustable_indices)
    for index in adjustable_indices:
        current = int(mutable[index])
        parity_flipped = [
            int(value)
            for value in other_allowed
            if int(value) != int(current) and (int(value) % 2) != (int(current) % 2)
        ]
        if parity_flipped:
            mutable[index] = int(rng.choice(parity_flipped))
            rng.shuffle(mutable)
            return tuple(int(value) for value in mutable)
    return None


def _sample_sum_matched_sequence(
    rng: random.Random,
    *,
    node_count: int,
    total_sum: int,
    max_degree: int,
    topology_profile: str,
) -> Tuple[int, ...] | None:
    """Sample one degree sequence whose entries stay within bounds and sum to ``total_sum``."""

    allowed = tuple(range(0, min(int(max_degree), int(node_count) - 1) + 1))
    if not allowed:
        return None
    min_total = int(min(allowed)) * int(node_count)
    max_total = int(max(allowed)) * int(node_count)
    if int(total_sum) < int(min_total) or int(total_sum) > int(max_total):
        return None
    sequence = [
        _draw_degree(
            rng,
            allowed_degrees=allowed,
            topology_profile=str(topology_profile),
        )
        for _ in range(int(node_count))
    ]
    current_total = int(sum(sequence))
    max_steps = max(16, int(node_count) * int(max_degree) * 4)
    for _ in range(int(max_steps)):
        if int(current_total) == int(total_sum):
            rng.shuffle(sequence)
            return tuple(int(value) for value in sequence)
        if int(current_total) < int(total_sum):
            adjustable = [index for index, value in enumerate(sequence) if int(value) < int(max(allowed))]
            if not adjustable:
                return None
            index = int(rng.choice(adjustable))
            sequence[index] = int(sequence[index]) + 1
            current_total += 1
        else:
            adjustable = [index for index, value in enumerate(sequence) if int(value) > int(min(allowed))]
            if not adjustable:
                return None
            index = int(rng.choice(adjustable))
            sequence[index] = int(sequence[index]) - 1
            current_total -= 1
    return None


def _find_graph_with_degree_count(
    rng: random.Random,
    *,
    node_count: int,
    query_degree: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
    search_attempts: int,
) -> Tuple[nx.Graph, Tuple[int, ...]] | None:
    """Search for one simple graph with exactly ``target_count`` query-degree nodes."""

    for _ in range(max(1, int(search_attempts))):
        candidate_sequence = _parity_adjusted_sequence(
            rng,
            node_count=int(node_count),
            query_degree=int(query_degree),
            target_count=int(target_count),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
        )
        if candidate_sequence is None:
            continue
        if not nx.is_graphical(candidate_sequence, method="hh"):
            continue
        graph = nx.havel_hakimi_graph(candidate_sequence)
        if sum(1 for _, degree in graph.degree() if int(degree) == int(query_degree)) != int(target_count):
            continue
        if graph.number_of_edges() >= 2:
            nswap = max(1, min(int(graph.number_of_edges()) * 2, 16))
            try:
                nx.double_edge_swap(
                    graph,
                    nswap=int(nswap),
                    max_tries=int(nswap) * 10,
                    seed=int(rng.randrange(1, 2**31 - 1)),
                )
            except Exception:
                pass
        return graph, tuple(int(value) for value in candidate_sequence)
    return None


def _find_directed_graph_with_degree_count(
    rng: random.Random,
    *,
    node_count: int,
    query_degree: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
    degree_mode: str,
    search_attempts: int,
) -> Tuple[nx.DiGraph, Tuple[int, ...], Tuple[int, ...]] | None:
    """Search for one simple directed graph with exact in/out-degree support."""

    mode = str(degree_mode)
    if mode not in {"in_degree", "out_degree"}:
        raise ValueError(f"unsupported directed degree mode: {degree_mode}")

    for _ in range(max(1, int(search_attempts))):
        queried_sequence = _sample_exact_query_sequence(
            rng,
            node_count=int(node_count),
            query_degree=int(query_degree),
            target_count=int(target_count),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
        )
        if queried_sequence is None:
            continue
        other_sequence = _sample_sum_matched_sequence(
            rng,
            node_count=int(node_count),
            total_sum=int(sum(queried_sequence)),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
        )
        if other_sequence is None:
            continue
        if mode == "in_degree":
            in_sequence = tuple(int(value) for value in queried_sequence)
            out_sequence = tuple(int(value) for value in other_sequence)
        else:
            in_sequence = tuple(int(value) for value in other_sequence)
            out_sequence = tuple(int(value) for value in queried_sequence)
        if not nx.is_digraphical(in_sequence, out_sequence):
            continue
        try:
            graph = nx.directed_havel_hakimi_graph(in_sequence, out_sequence)
        except Exception:
            continue
        if any(int(node) == int(neighbor) for node, neighbor in graph.edges()):
            continue
        if _has_reciprocal_edges(graph):
            continue
        observed = graph.in_degree if mode == "in_degree" else graph.out_degree
        if sum(1 for _, degree in observed() if int(degree) == int(query_degree)) != int(target_count):
            continue
        return graph, tuple(int(value) for value in in_sequence), tuple(int(value) for value in out_sequence)
    return None


@lru_cache(maxsize=128)
def feasible_node_counts_for_degree_count(
    *,
    query_id: str,
    degree_mode: str | None = None,
    query_degree: int,
    target_count: int,
    node_count_min: int,
    node_count_max: int,
    max_degree: int,
    topology_profile: str = "balanced",
) -> Tuple[int, ...]:
    """Return node counts that can realize the requested degree-count query."""

    support = []
    degree_mode = graph_degree_mode_for_query_id(str(query_id), degree_mode=degree_mode)
    for node_count in range(int(node_count_min), int(node_count_max) + 1):
        feasibility_rng = random.Random(
            f"graph-degree-feasibility:{str(query_id)}:{int(node_count)}:{int(query_degree)}:{int(target_count)}:{int(max_degree)}"
        )
        if str(graph_directionality_for_query_id(str(query_id))) == "directed":
            result = _find_directed_graph_with_degree_count(
                feasibility_rng,
                node_count=int(node_count),
                query_degree=int(query_degree),
                target_count=int(target_count),
                max_degree=int(max_degree),
                topology_profile=str(topology_profile),
                degree_mode=str(degree_mode),
                search_attempts=600,
            )
        else:
            result = _find_graph_with_degree_count(
                feasibility_rng,
                node_count=int(node_count),
                query_degree=int(query_degree),
                target_count=int(target_count),
                max_degree=int(max_degree),
                topology_profile=str(topology_profile),
                search_attempts=600,
            )
        if result is not None:
            support.append(int(node_count))
    return tuple(int(value) for value in support)


def sample_degree_count_graph(
    rng: random.Random,
    *,
    query_id: str,
    degree_mode: str | None = None,
    node_count: int,
    query_degree: int,
    target_count: int,
    max_degree: int,
    topology_profile: str,
    label_variant: str,
    search_attempts: int,
) -> GraphCountSample:
    """Construct one labeled graph with the requested degree-count support."""

    query_id_text = str(query_id)
    directionality = str(graph_directionality_for_query_id(query_id_text))
    degree_mode = str(graph_degree_mode_for_query_id(query_id_text, degree_mode=degree_mode))
    if directionality == "directed":
        result = _find_directed_graph_with_degree_count(
            rng,
            node_count=int(node_count),
            query_degree=int(query_degree),
            target_count=int(target_count),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
            degree_mode=str(degree_mode),
            search_attempts=int(search_attempts),
        )
        if result is None:
            raise ValueError("failed to sample a simple directed graph for the requested degree-count support")
        graph, in_degree_sequence, out_degree_sequence = result
        degree_sequence = in_degree_sequence if degree_mode == "in_degree" else out_degree_sequence
    else:
        result = _find_graph_with_degree_count(
            rng,
            node_count=int(node_count),
            query_degree=int(query_degree),
            target_count=int(target_count),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
            search_attempts=int(search_attempts),
        )
        if result is None:
            raise ValueError("failed to sample a simple graph for the requested degree-count support")
        graph, degree_sequence = result
        in_degree_sequence = tuple(int(value) for value in degree_sequence)
        out_degree_sequence = tuple(int(value) for value in degree_sequence)

    topology_sample, _label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directionality == "directed"),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    if degree_mode == "in_degree":
        degrees_by_label = {str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()}
    elif degree_mode == "out_degree":
        degrees_by_label = {str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()}
    else:
        degrees_by_label = {str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()}
    target_labels = tuple(
        sorted((label for label, degree in degrees_by_label.items() if int(degree) == int(query_degree)), key=graph_label_sort_key)
    )
    return GraphCountSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        target_labels=tuple(str(label) for label in target_labels),
        degree_sequence=tuple(int(value) for value in degree_sequence),
        in_degree_sequence=tuple(int(value) for value in in_degree_sequence),
        out_degree_sequence=tuple(int(value) for value in out_degree_sequence),
        query_degree=int(query_degree),
        degree_mode=str(degree_mode),
    )


__all__ = [
    "feasible_node_counts_for_degree_count",
    "sample_degree_count_graph",
]

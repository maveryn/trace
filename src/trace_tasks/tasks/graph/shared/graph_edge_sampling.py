"""Shared graph-domain edge sampling helpers."""

from __future__ import annotations

import random
from typing import Sequence

import networkx as nx


def _profile_extra_edge_budget(
    *,
    node_count: int,
    topology_profile: str,
    directed: bool,
) -> int:
    """Return a small distractor-edge budget for one topology profile."""

    node_count_int = int(node_count)
    profile = str(topology_profile)
    multiplier = {
        "low_degree": 0.45,
        "hub_heavy": 1.15,
    }.get(profile, 0.75)
    directed_factor = 1.35 if bool(directed) else 1.0
    return max(0, int(round(float(max(0, node_count_int - 2)) * multiplier * directed_factor)))


def _add_random_undirected_distractor_edges(
    graph: nx.Graph,
    rng: random.Random,
    *,
    nodes: Sequence[int],
    extra_edges: int,
    max_degree: int,
) -> None:
    """Add random non-target undirected edges while respecting max degree."""

    node_list = [int(node) for node in nodes]
    candidates = [
        (int(node_list[left_index]), int(node_list[right_index]))
        for left_index in range(len(node_list))
        for right_index in range(left_index + 1, len(node_list))
    ]
    rng.shuffle(candidates)
    added = 0
    for left, right in candidates:
        if int(added) >= int(extra_edges):
            break
        if graph.has_edge(int(left), int(right)):
            continue
        if int(graph.degree(int(left))) >= int(max_degree) or int(graph.degree(int(right))) >= int(max_degree):
            continue
        graph.add_edge(int(left), int(right))
        added += 1


def _add_directed_edge_without_reciprocal(
    graph: nx.DiGraph,
    *,
    source: int,
    target: int,
    max_degree: int,
) -> bool:
    """Add one directed edge when it preserves simple no-reciprocal support."""

    source_int = int(source)
    target_int = int(target)
    if int(source_int) == int(target_int):
        return False
    if graph.has_edge(int(source_int), int(target_int)) or graph.has_edge(int(target_int), int(source_int)):
        return False
    if int(graph.out_degree(int(source_int))) >= int(max_degree):
        return False
    if int(graph.in_degree(int(target_int))) >= int(max_degree):
        return False
    graph.add_edge(int(source_int), int(target_int))
    return True


def _add_random_directed_distractor_edges(
    graph: nx.DiGraph,
    rng: random.Random,
    *,
    nodes: Sequence[int],
    extra_edges: int,
    max_degree: int,
) -> None:
    """Add random non-target directed edges while avoiding reciprocal pairs."""

    node_list = [int(node) for node in nodes]
    candidates = [
        (int(left), int(right))
        for left in node_list
        for right in node_list
        if int(left) != int(right)
    ]
    rng.shuffle(candidates)
    added = 0
    for source, target in candidates:
        if int(added) >= int(extra_edges):
            break
        if _add_directed_edge_without_reciprocal(
            graph,
            source=int(source),
            target=int(target),
            max_degree=int(max_degree),
        ):
            added += 1


__all__ = [
    "_add_directed_edge_without_reciprocal",
    "_add_random_directed_distractor_edges",
    "_add_random_undirected_distractor_edges",
    "_profile_extra_edge_budget",
]

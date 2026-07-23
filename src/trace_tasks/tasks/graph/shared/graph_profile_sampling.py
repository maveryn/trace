"""Topology-profile graph construction helpers shared by graph samplers."""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import networkx as nx


def _choose_attachment_parent(
    rng: random.Random,
    *,
    graph: nx.Graph,
    topology_profile: str,
) -> int:
    """Choose one attachment parent under the requested topology profile."""

    nodes = [int(node) for node in graph.nodes()]
    if not nodes:
        raise ValueError("attachment parent sampling requires at least one node")
    profile = str(topology_profile)
    if profile == "hub_heavy":
        weights = [float(graph.degree(int(node)) + 1) ** 2 for node in nodes]
    elif profile == "low_degree":
        weights = [1.0 / float(graph.degree(int(node)) + 1) for node in nodes]
    else:
        weights = [1.0 for _ in nodes]
    return int(rng.choices(nodes, weights=weights, k=1)[0])


def _edge_weight_for_profile(
    graph: nx.Graph,
    *,
    edge: Tuple[int, int],
    topology_profile: str,
) -> float:
    """Return one edge-selection weight for articulation-preserving expansion."""

    left, right = int(edge[0]), int(edge[1])
    profile = str(topology_profile)
    degree_sum = float(graph.degree(left) + graph.degree(right))
    if profile == "hub_heavy":
        return float((degree_sum + 2.0) ** 2)
    if profile == "low_degree":
        return 1.0 / float(degree_sum + 2.0)
    return 1.0


def _sample_profile_tree_graph(
    rng: random.Random,
    *,
    node_count: int,
    topology_profile: str,
) -> nx.Graph:
    """Return one connected tree whose shape follows the requested profile."""

    node_count_int = int(node_count)
    if int(node_count_int) <= 0:
        raise ValueError("tree sampling requires at least one node")
    graph = nx.Graph()
    graph.add_node(0)
    for next_node in range(1, int(node_count_int)):
        parent = _choose_attachment_parent(
            rng,
            graph=graph,
            topology_profile=str(topology_profile),
        )
        graph.add_node(int(next_node))
        graph.add_edge(int(parent), int(next_node))
    return graph


def _sample_profile_extra_edges(
    rng: random.Random,
    *,
    graph: nx.Graph,
    extra_edge_count: int,
    topology_profile: str,
) -> Tuple[Tuple[int, int], ...]:
    """Add non-tree edges under the requested profile and return them."""

    added_edges: list[Tuple[int, int]] = []
    for _ in range(max(0, int(extra_edge_count))):
        candidates = [(int(left), int(right)) for left, right in nx.non_edges(graph)]
        if not candidates:
            break
        weights = [
            _edge_weight_for_profile(
                graph,
                edge=(int(left), int(right)),
                topology_profile=str(topology_profile),
            )
            for left, right in candidates
        ]
        left, right = rng.choices(candidates, weights=weights, k=1)[0]
        graph.add_edge(int(left), int(right))
        added_edges.append(tuple(sorted((int(left), int(right)))))
    if int(len(added_edges)) != int(extra_edge_count):
        raise ValueError("failed to add the requested number of non-tree edges")
    return tuple(tuple(int(value) for value in edge) for edge in added_edges)


__all__ = [
    "_choose_attachment_parent",
    "_edge_weight_for_profile",
    "_sample_profile_extra_edges",
    "_sample_profile_tree_graph",
]

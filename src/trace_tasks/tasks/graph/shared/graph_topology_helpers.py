"""Shared graph-domain topology labeling helpers."""

from __future__ import annotations

import random
from typing import Dict, Mapping, Tuple

import networkx as nx

from .graph_sample_types import GraphTopologySample, graph_label_sort_key, sort_graph_edge_labels
from .label_assets import resolve_graph_node_labels


def _build_labeled_graph_topology_sample(
    rng: random.Random,
    *,
    graph: nx.Graph | nx.DiGraph,
    directed: bool,
    topology_profile: str,
    label_variant: str,
) -> Tuple[GraphTopologySample, Dict[int, str]]:
    """Attach prompt-facing labels and sorted adjacency metadata to one graph."""

    node_order = tuple(int(node) for node in graph.nodes())
    resolved_labels = resolve_graph_node_labels(
        rng,
        label_variant=str(label_variant),
        object_count=int(graph.number_of_nodes()),
        max_chars=3,
        sequential_numbers=True,
    )
    labels = tuple(str(label) for label in resolved_labels.labels)
    label_by_node = {int(node): str(label) for node, label in zip(node_order, labels)}

    if bool(directed):
        labeled_edges = sort_graph_edge_labels(
            tuple(
                (str(label_by_node[int(left)]), str(label_by_node[int(right)]))
                for left, right in graph.edges()
            ),
            directed=True,
        )
        in_degrees_by_label = {str(label_by_node[int(node)]): int(degree) for node, degree in graph.in_degree()}
        out_degrees_by_label = {str(label_by_node[int(node)]): int(degree) for node, degree in graph.out_degree()}
        adjacency_by_label = {
            str(label_by_node[int(node)]): tuple(
                sorted(
                    (
                        {
                            *[str(label_by_node[int(neighbor)]) for neighbor in graph.predecessors(int(node))],
                            *[str(label_by_node[int(neighbor)]) for neighbor in graph.successors(int(node))],
                        }
                    ),
                    key=graph_label_sort_key,
                )
            )
            for node in node_order
        }
        successors_by_label = {
            str(label_by_node[int(node)]): tuple(
                sorted((str(label_by_node[int(neighbor)]) for neighbor in graph.successors(int(node))), key=graph_label_sort_key)
            )
            for node in node_order
        }
        predecessors_by_label = {
            str(label_by_node[int(node)]): tuple(
                sorted((str(label_by_node[int(neighbor)]) for neighbor in graph.predecessors(int(node))), key=graph_label_sort_key)
            )
            for node in node_order
        }
    else:
        labeled_edges = sort_graph_edge_labels(
            tuple(
                (str(label_by_node[int(left)]), str(label_by_node[int(right)]))
                for left, right in graph.edges()
            ),
            directed=False,
        )
        in_degrees_by_label = {str(label_by_node[int(node)]): int(degree) for node, degree in graph.degree()}
        out_degrees_by_label = {str(label_by_node[int(node)]): int(degree) for node, degree in graph.degree()}
        adjacency_by_label = {
            str(label_by_node[int(node)]): tuple(
                sorted((str(label_by_node[int(neighbor)]) for neighbor in graph.neighbors(int(node))), key=graph_label_sort_key)
            )
            for node in node_order
        }
        successors_by_label = dict(adjacency_by_label)
        predecessors_by_label = dict(adjacency_by_label)

    topology = GraphTopologySample(
        graph=graph,
        directed=bool(directed),
        node_labels=tuple(str(label_by_node[int(node)]) for node in node_order),
        edge_labels=tuple((str(left), str(right)) for left, right in labeled_edges),
        degrees_by_label={str(key): int(value) for key, value in in_degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in predecessors_by_label.items()},
        edge_count=int(graph.number_of_edges()),
        topology_profile=str(topology_profile),
        label_variant=str(resolved_labels.label_variant),
        label_source_kind=str(resolved_labels.label_source_kind),
        label_bucket=str(resolved_labels.label_bucket),
        label_manifest=str(resolved_labels.label_manifest),
        label_filter=dict(resolved_labels.label_filter),
        label_bucket_probabilities=dict(resolved_labels.label_bucket_probabilities),
    )
    return topology, label_by_node


def _graph_adjacency_by_node(graph: nx.Graph) -> Dict[int, Tuple[int, ...]]:
    """Return a deterministic undirected adjacency mapping keyed by node id."""

    return {
        int(node): tuple(sorted((int(neighbor) for neighbor in graph.neighbors(int(node)))))
        for node in sorted((int(value) for value in graph.nodes()))
    }


def _digraph_successor_adjacency_by_node(graph: nx.DiGraph) -> Dict[int, Tuple[int, ...]]:
    """Return a deterministic directed successor adjacency keyed by node id."""

    return {
        int(node): tuple(sorted((int(neighbor) for neighbor in graph.successors(int(node)))))
        for node in sorted((int(value) for value in graph.nodes()))
    }


def _digraph_predecessor_adjacency_by_node(graph: nx.DiGraph) -> Dict[int, Tuple[int, ...]]:
    """Return a deterministic directed predecessor adjacency keyed by node id."""

    return {
        int(node): tuple(sorted((int(neighbor) for neighbor in graph.predecessors(int(node)))))
        for node in sorted((int(value) for value in graph.nodes()))
    }


def _has_reciprocal_edges(graph: nx.DiGraph) -> bool:
    """Return whether one directed graph contains any reciprocal edge pair."""

    for left, right in graph.edges():
        if int(left) == int(right):
            return True
        if graph.has_edge(int(right), int(left)):
            return True
    return False


def _directed_adjacency_by_label_for_graph(
    graph: nx.DiGraph,
    *,
    label_by_node: Mapping[int, str],
) -> Dict[str, Tuple[str, ...]]:
    """Return sorted total directed adjacency for one labeled graph."""

    return {
        str(label_by_node[int(node)]): tuple(
            sorted(
                (
                    {
                        *[str(label_by_node[int(neighbor)]) for neighbor in graph.predecessors(int(node))],
                        *[str(label_by_node[int(neighbor)]) for neighbor in graph.successors(int(node))],
                    }
                ),
                key=graph_label_sort_key,
            )
        )
        for node in sorted((int(node) for node in graph.nodes()))
    }

def _post_removal_degree_maps_by_label(
    graph: nx.Graph | nx.DiGraph,
    *,
    label_by_node: Mapping[int, str],
    directed: bool,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Return total, in-, and out-degree maps for one labeled post-removal graph."""

    if bool(directed):
        digraph = graph  # type: ignore[assignment]
        in_degrees = {str(label_by_node[int(node)]): int(digraph.in_degree(int(node))) for node in digraph.nodes()}
        out_degrees = {str(label_by_node[int(node)]): int(digraph.out_degree(int(node))) for node in digraph.nodes()}
        total_degrees = {
            str(label): int(in_degrees[str(label)]) + int(out_degrees[str(label)])
            for label in in_degrees.keys()
        }
        return total_degrees, in_degrees, out_degrees
    undirected_graph = graph  # type: ignore[assignment]
    degrees = {str(label_by_node[int(node)]): int(undirected_graph.degree(int(node))) for node in undirected_graph.nodes()}
    return dict(degrees), dict(degrees), dict(degrees)

def _undirected_adjacency_by_label_for_graph(
    graph: nx.Graph,
    *,
    label_by_node: Mapping[int, str],
) -> Dict[str, Tuple[str, ...]]:
    """Return sorted label adjacency for one undirected graph."""

    return {
        str(label_by_node[int(node)]): tuple(
            sorted(
                (str(label_by_node[int(neighbor)]) for neighbor in graph.neighbors(int(node))),
                key=graph_label_sort_key,
            )
        )
        for node in sorted((int(node) for node in graph.nodes()))
    }

def _components_by_label_for_graph(
    graph: nx.Graph,
    *,
    label_by_node: Mapping[int, str],
) -> Tuple[Tuple[str, ...], ...]:
    """Return connected components as sorted label tuples."""

    components = [
        tuple(sorted((str(label_by_node[int(node)]) for node in component), key=graph_label_sort_key))
        for component in nx.connected_components(graph)
    ]
    return tuple(
        sorted(
            components,
            key=lambda labels: graph_label_sort_key(labels[0]) if labels else (0, ""),
        )
    )

def _directed_successors_by_label_for_graph(
    graph: nx.DiGraph,
    *,
    label_by_node: Mapping[int, str],
) -> Dict[str, Tuple[str, ...]]:
    """Return sorted directed successor adjacency for one labeled graph."""

    return {
        str(label_by_node[int(node)]): tuple(
            sorted(
                (str(label_by_node[int(neighbor)]) for neighbor in graph.successors(int(node))),
                key=graph_label_sort_key,
            )
        )
        for node in sorted((int(node) for node in graph.nodes()))
    }

def _directed_predecessors_by_label_for_graph(
    graph: nx.DiGraph,
    *,
    label_by_node: Mapping[int, str],
) -> Dict[str, Tuple[str, ...]]:
    """Return sorted directed predecessor adjacency for one labeled graph."""

    return {
        str(label_by_node[int(node)]): tuple(
            sorted(
                (str(label_by_node[int(neighbor)]) for neighbor in graph.predecessors(int(node))),
                key=graph_label_sort_key,
            )
        )
        for node in sorted((int(node) for node in graph.nodes()))
    }


__all__ = [
    "_directed_predecessors_by_label_for_graph",
    "_directed_successors_by_label_for_graph",
    "_components_by_label_for_graph",
    "_undirected_adjacency_by_label_for_graph",
    "_post_removal_degree_maps_by_label",
    "_directed_adjacency_by_label_for_graph",
    "_build_labeled_graph_topology_sample",
    "_digraph_predecessor_adjacency_by_node",
    "_digraph_successor_adjacency_by_node",
    "_graph_adjacency_by_node",
    "_has_reciprocal_edges",
]

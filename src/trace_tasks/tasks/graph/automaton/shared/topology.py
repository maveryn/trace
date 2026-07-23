"""Topology conversion helpers for graph automaton renderings."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import networkx as nx

from ...shared.graph_sample_types import GraphTopologySample, graph_label_sort_key
from .labels import sorted_state_label_tuple


def build_automaton_topology_sample(
    *,
    graph: nx.DiGraph,
    labels: Sequence[str],
    transition_labels_by_edge: Mapping[Tuple[str, str], str],
) -> GraphTopologySample:
    """Build the generic graph topology record for one automaton diagram."""

    label_by_node = {int(node): str(labels[int(node)]) for node in graph.nodes()}
    edge_labels = tuple(
        sorted(
            ((str(label_by_node[int(left)]), str(label_by_node[int(right)])) for left, right in graph.edges()),
            key=lambda pair: (graph_label_sort_key(pair[0]), graph_label_sort_key(pair[1])),
        )
    )
    adjacency_by_label = {
        str(label_by_node[int(node)]): sorted_state_label_tuple(
            {
                *[str(label_by_node[int(neighbor)]) for neighbor in graph.predecessors(int(node))],
                *[str(label_by_node[int(neighbor)]) for neighbor in graph.successors(int(node))],
            }
        )
        for node in graph.nodes()
    }
    successors_by_label = {
        str(label_by_node[int(node)]): sorted_state_label_tuple(
            str(label_by_node[int(neighbor)]) for neighbor in graph.successors(int(node))
        )
        for node in graph.nodes()
    }
    predecessors_by_label = {
        str(label_by_node[int(node)]): sorted_state_label_tuple(
            str(label_by_node[int(neighbor)]) for neighbor in graph.predecessors(int(node))
        )
        for node in graph.nodes()
    }
    degrees_by_label = {
        str(label_by_node[int(node)]): int(graph.in_degree(int(node)) + graph.out_degree(int(node)))
        for node in graph.nodes()
    }
    return GraphTopologySample(
        graph=graph,
        directed=True,
        node_labels=tuple(str(label_by_node[int(node)]) for node in graph.nodes()),
        edge_labels=tuple((str(left), str(right)) for left, right in edge_labels),
        degrees_by_label={str(key): int(value) for key, value in degrees_by_label.items()},
        in_degrees_by_label={str(label_by_node[int(node)]): int(graph.in_degree(int(node))) for node in graph.nodes()},
        out_degrees_by_label={str(label_by_node[int(node)]): int(graph.out_degree(int(node))) for node in graph.nodes()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in predecessors_by_label.items()},
        edge_count=int(graph.number_of_edges()),
        topology_profile="automaton_partial_dfa",
        label_variant="automaton_state",
    )


__all__ = ["build_automaton_topology_sample"]

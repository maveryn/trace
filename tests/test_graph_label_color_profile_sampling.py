from __future__ import annotations

import random

import networkx as nx

from trace_tasks.tasks.graph.shared import graph_label_color_sampling
from trace_tasks.tasks.graph.shared import graph_profile_sampling


def test_profile_tree_and_extra_edge_helpers_construct_expected_graphs() -> None:
    rng = random.Random(3110)
    graph = graph_profile_sampling._sample_profile_tree_graph(
        rng,
        node_count=7,
        topology_profile="balanced",
    )

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 6
    assert nx.is_connected(graph)

    added_edges = graph_profile_sampling._sample_profile_extra_edges(
        rng,
        graph=graph,
        extra_edge_count=2,
        topology_profile="balanced",
    )
    assert len(added_edges) == 2
    assert graph.number_of_edges() == 8


def test_label_color_samplers_construct_requested_support() -> None:
    node_sample = graph_label_color_sampling.sample_node_color_count_graph(
        random.Random(3201),
        graph_directionality="undirected",
        node_count=6,
        target_count=2,
        target_color_name="red",
        color_support=("red", "blue", "green"),
        topology_profile="balanced",
        label_variant="letters",
        max_degree=4,
    )
    assert node_sample.target_count == 2
    assert node_sample.color_counts_by_name["red"] == 2

    edge_sample = graph_label_color_sampling.sample_edge_text_label_count_graph(
        random.Random(3202),
        graph_directionality="directed",
        node_count=6,
        target_count=2,
        target_edge_label="feeds",
        edge_label_support=("feeds", "blocks", "routes"),
        topology_profile="balanced",
        label_variant="letters",
        max_degree=4,
    )
    assert edge_sample.directed is True
    assert edge_sample.target_count == 2
    assert edge_sample.edge_label_counts_by_value["feeds"] == 2

    cross_sample = graph_label_color_sampling.sample_cross_color_edge_count_graph(
        random.Random(3203),
        graph_directionality="undirected",
        node_count=6,
        target_count=1,
        source_color_name="red",
        target_color_name="blue",
        color_support=("red", "blue", "green"),
        topology_profile="balanced",
        label_variant="letters",
        max_degree=4,
    )
    assert cross_sample.target_count == 1
    assert len(cross_sample.target_edges) == 1

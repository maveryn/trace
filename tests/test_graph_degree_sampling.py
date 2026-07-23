from __future__ import annotations

import random

from trace_tasks.tasks.graph.shared import graph_degree_sampling


def test_degree_sampler_constructs_requested_undirected_support() -> None:
    sample = graph_degree_sampling.sample_degree_count_graph(
        random.Random(1042),
        query_id="degree_count",
        node_count=6,
        query_degree=2,
        target_count=3,
        max_degree=4,
        topology_profile="balanced",
        label_variant="letters",
        search_attempts=300,
    )

    assert sample.directed is False
    assert sample.degree_mode == "degree"
    assert len(sample.target_labels) == 3
    assert all(sample.degrees_by_label[label] == 2 for label in sample.target_labels)


def test_degree_feasible_support_handles_directed_mode() -> None:
    support = graph_degree_sampling.feasible_node_counts_for_degree_count(
        query_id="directed_degree_count",
        degree_mode="out_degree",
        query_degree=1,
        target_count=2,
        node_count_min=4,
        node_count_max=6,
        max_degree=3,
    )

    assert support
    assert all(4 <= node_count <= 6 for node_count in support)

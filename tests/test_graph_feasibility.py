"""Regression tests for graph feasible-support helpers."""

from __future__ import annotations

from trace_tasks.tasks.graph.shared import graph_feasibility


def test_path_and_reachability_feasible_supports() -> None:
    assert graph_feasibility.feasible_node_counts_for_shortest_path_length(
        target_shortest_path_length=3,
        node_count_min=4,
        node_count_max=8,
    ) == (5, 6, 7, 8)
    assert graph_feasibility.feasible_node_counts_for_reachable_count(
        target_reachable_count=4,
        node_count_min=3,
        node_count_max=7,
    ) == (5, 6, 7)
    assert graph_feasibility.feasible_node_counts_for_reachable_count_after_edge_edit(
        edit_operation="edge_addition",
        target_reachable_count=1,
        node_count_min=5,
        node_count_max=9,
    ) == ()


def test_bridge_and_mst_feasible_supports() -> None:
    assert graph_feasibility.feasible_node_counts_for_bridge_count(
        target_count=2,
        node_count_min=3,
        node_count_max=7,
    ) == (3, 5, 6, 7)
    assert graph_feasibility.feasible_extra_edge_counts_for_minimum_spanning_tree(
        node_count=5,
        extra_edge_count_min=1,
        extra_edge_count_max=5,
        edge_weight_min=1,
        edge_weight_max=7,
    ) == (1, 2, 3)

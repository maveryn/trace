"""Unit tests for shared graph-algorithm helpers."""

from __future__ import annotations

from trace_tasks.tasks.shared.graph_algorithms import connected_components_by_adjacency


def test_connected_components_by_adjacency_respects_node_order() -> None:
    adjacency = {
        "b": ["a"],
        "a": ["b"],
        "d": ["c"],
        "c": ["d"],
        "z": [],
    }
    components = connected_components_by_adjacency(
        adjacency,
        node_order=["z", "a", "b", "c", "d"],
    )
    assert components == [["z"], ["a", "b"], ["c", "d"]]

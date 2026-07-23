"""Regression tests for graph sample type helpers."""

from __future__ import annotations

from trace_tasks.tasks.graph.shared import graph_sample_types


def test_graph_sample_types_expose_generic_constants() -> None:
    assert graph_sample_types.SUPPORTED_LAYOUT_VARIANTS
    assert graph_sample_types.SUPPORTED_NODE_LINK_LABEL_VARIANTS


def test_graph_edge_label_helpers_remain_compatible() -> None:
    assert graph_sample_types.graph_label_sort_key("10") == (0, 10)
    assert graph_sample_types.graph_label_sort_key("A") == (1, "A")
    assert graph_sample_types.canonicalize_graph_edge_label("10", "2") == ("2", "10")
    assert graph_sample_types.canonicalize_graph_edge_label("10", "2", directed=True) == ("10", "2")
    assert graph_sample_types.sort_graph_edge_labels([("B", "A"), ("10", "2")]) == (("2", "10"), ("A", "B"))

"""Tests for graph shared task-scaffolding helpers."""

from __future__ import annotations

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.graph.node_link.shared.sampling import resolve_node_link_visual_axes
from trace_tasks.tasks.graph.shared.task_scaffolding import (
    GraphBalancedAxisSpec,
    graph_decoupled_selection_index,
    graph_hashed_axis_selection_index,
)
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index


def test_graph_hashed_axis_selection_index_matches_namespaced_hash() -> None:
    index = graph_hashed_axis_selection_index(
        101,
        task_id="task_graph__node_link__example",
        axis_name="node_count",
        selection_index=17,
        axis_values=("directed", "in_degree", 2, "balanced"),
    )

    assert index == hash64(
        101,
        "task_graph__node_link__example:node_count:directed:in_degree:2:balanced",
        17,
    )


def test_graph_decoupled_selection_index_divides_balanced_axes() -> None:
    params = {}
    gen_defaults = {
        "balanced_primary_axis_sampling": True,
        "balanced_degree_mode_sampling": True,
    }
    base = resolve_selection_index(
        params=params,
        instance_seed=202,
        namespace="task_graph__node_link__example:query_support",
    )

    decoupled = graph_decoupled_selection_index(
        202,
        params=params,
        task_id="task_graph__node_link__example",
        namespace="query_support",
        gen_defaults=gen_defaults,
        axis_specs=(
            GraphBalancedAxisSpec(
                probabilities={"a": 0.5, "b": 0.5},
                balance_flag_key="balanced_primary_axis_sampling",
                explicit_keys=("primary_axis",),
                weights_key="primary_axis_weights",
            ),
            GraphBalancedAxisSpec(
                probabilities={"in": 1.0, "out": 1.0, "total": 1.0},
                balance_flag_key="balanced_degree_mode_sampling",
                explicit_keys=("degree_mode",),
                weights_key="degree_mode_weights",
            ),
        ),
    )

    assert decoupled == base // 6


def test_graph_decoupled_selection_index_respects_axis_overrides() -> None:
    params = {"primary_axis": "a"}
    gen_defaults = {
        "balanced_primary_axis_sampling": True,
        "balanced_degree_mode_sampling": True,
    }
    base = resolve_selection_index(
        params=params,
        instance_seed=203,
        namespace="task_graph__node_link__example:query_support",
    )

    decoupled = graph_decoupled_selection_index(
        203,
        params=params,
        task_id="task_graph__node_link__example",
        namespace="query_support",
        gen_defaults=gen_defaults,
        axis_specs=(
            GraphBalancedAxisSpec(
                probabilities={"a": 0.5, "b": 0.5},
                balance_flag_key="balanced_primary_axis_sampling",
                explicit_keys=("primary_axis",),
                weights_key="primary_axis_weights",
            ),
            GraphBalancedAxisSpec(
                probabilities={"in": 1.0, "out": 1.0},
                balance_flag_key="balanced_degree_mode_sampling",
                explicit_keys=("degree_mode",),
                weights_key="degree_mode_weights",
            ),
        ),
    )

    assert decoupled == base // 2


def test_resolve_node_link_visual_axes_honors_explicit_params() -> None:
    axes = resolve_node_link_visual_axes(
        301,
        params={
            "layout_variant": "shell",
            "label_variant": "named",
            "node_shape_variant": "hexagon",
            "layout_transform_variant": "rotate_90",
            "edge_routing_variant": "mixed_arc",
            "node_color_name": "orange",
        },
        gen_defaults={},
        selection_salt="node_link_visual_axes_test",
    )

    assert axes.layout_variant == "shell"
    assert axes.label_variant == "named"
    assert axes.node_shape_variant == "hexagon"
    assert axes.layout_transform_variant == "rotate_90"
    assert axes.edge_routing_variant == "mixed_arc"
    assert axes.node_color_name == "orange"
    assert axes.layout_variant_probabilities["shell"] == 1.0
    assert sum(axes.layout_variant_probabilities.values()) == 1.0
    assert axes.edge_routing_variant_probabilities == {"mixed_arc": 1.0, "straight": 0.0}


def test_resolve_node_link_visual_axes_supports_custom_color_keys() -> None:
    axes = resolve_node_link_visual_axes(
        302,
        params={"theme_node_color_name": "green"},
        gen_defaults={},
        selection_salt="node_link_visual_axes_test",
        node_color_explicit_key="theme_node_color_name",
        node_color_weights_key="theme_node_color_name_weights",
        node_color_balance_flag_key="balanced_theme_node_color_name_sampling",
        node_color_namespace="theme_node_color_name",
    )

    assert axes.node_color_name == "green"
    assert axes.value_fields(node_color_field_name="theme_node_color_name")["theme_node_color_name"] == "green"
    assert axes.probability_fields(node_color_field_name="theme_node_color_name")[
        "theme_node_color_name_probabilities"
    ]["green"] == 1.0


def test_resolve_node_link_visual_axes_can_skip_optional_axes() -> None:
    axes = resolve_node_link_visual_axes(
        303,
        params={
            "layout_variant": "spring",
            "label_variant": "letters",
            "node_shape_variant": "circle",
            "layout_transform_variant": "identity",
        },
        gen_defaults={},
        selection_salt="node_link_visual_axes_test",
        include_edge_routing_axis=False,
        include_node_color_axis=False,
    )

    assert axes.layout_variant == "spring"
    assert axes.edge_routing_variant == ""
    assert axes.edge_routing_variant_probabilities == {}
    assert axes.node_color_name == ""
    assert axes.node_color_name_probabilities == {}

"""Node-link scene sampling and visual-axis resolution.

The visual-axis resolver is scene-local. Public node-link task files import
domain-shared graph sampler primitives through this scene-local surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.graph.shared.graph_bridge_articulation_sampling import (
    feasible_node_counts_for_articulation_point_count,
    feasible_node_counts_for_bridge_count,
    sample_articulation_point_count_graph,
    sample_bridge_count_graph,
)
from trace_tasks.tasks.graph.shared.graph_common_neighbor_sampling import (
    feasible_node_counts_for_common_neighbor_count,
    sample_common_neighbor_count_graph,
)
from trace_tasks.tasks.graph.shared.graph_component_sampling import (
    feasible_node_counts_for_component_query,
    feasible_node_counts_for_component_size_after_edge_edit,
    feasible_node_counts_for_unique_largest_component,
    sample_component_count_graph,
    sample_component_size_after_edge_edit_graph,
    sample_largest_component_size_graph,
)
from trace_tasks.tasks.graph.shared.graph_degree_sampling import (
    feasible_node_counts_for_degree_count,
    sample_degree_count_graph,
)
from trace_tasks.tasks.graph.shared.graph_edge_path_label_sampling import (
    sample_edge_attribute_path_label_graph,
)
from trace_tasks.tasks.graph.shared.graph_isolation_sampling import (
    feasible_node_counts_for_isolated_node_count_after_node_removal,
    sample_isolated_node_count_after_node_removal_graph,
)
from trace_tasks.tasks.graph.shared.graph_label_color_sampling import (
    sample_cross_color_edge_count_graph,
    sample_edge_attribute_label_graph,
    sample_edge_color_count_graph,
    sample_edge_text_label_count_graph,
    sample_node_color_count_graph,
    sample_unique_node_label_relation_graph,
)
from trace_tasks.tasks.graph.shared.graph_mst_sampling import (
    feasible_extra_edge_counts_for_minimum_spanning_tree,
    sample_minimum_spanning_tree_weight_graph,
)
from trace_tasks.tasks.graph.shared.graph_node_degree_sampling import (
    feasible_node_counts_for_extreme_degree_value,
    feasible_node_counts_for_named_node_degree_value,
    sample_extreme_degree_graph,
    sample_named_node_degree_graph,
    sample_unique_extreme_degree_graph,
)
from trace_tasks.tasks.graph.shared.graph_path_order_sampling import (
    feasible_node_counts_for_hamiltonian_cycle_neighbor,
    feasible_node_counts_for_largest_chordless_cycle_size,
    feasible_node_counts_for_longest_path_length,
    feasible_node_counts_for_shortest_path_length,
    feasible_node_counts_for_topological_position,
    feasible_node_counts_for_unique_cycle_size,
    sample_hamiltonian_cycle_neighbor_graph,
    sample_largest_chordless_cycle_graph,
    sample_longest_path_length_graph,
    sample_shortest_path_length_graph,
    sample_topological_position_graph,
    sample_unique_cycle_graph,
)
from trace_tasks.tasks.graph.shared.graph_reachability_sampling import (
    feasible_node_counts_for_reachable_count,
    feasible_node_counts_for_reachable_count_after_edge_edit,
    sample_reachable_count_after_edge_edit_graph,
    sample_reachable_count_graph,
)
from trace_tasks.tasks.graph.shared.graph_render_types import (
    SUPPORTED_EDGE_ROUTING_VARIANTS,
    SUPPORTED_LAYOUT_TRANSFORM_VARIANTS,
    SUPPORTED_NODE_SHAPE_VARIANTS,
)
from trace_tasks.tasks.graph.shared.graph_sample_types import (
    SUPPORTED_LAYOUT_VARIANTS,
    SUPPORTED_NODE_LINK_LABEL_VARIANTS,
    SUPPORTED_TOPOLOGY_PROFILES,
    graph_label_sort_key,
)
from trace_tasks.tasks.graph.shared.style import SUPPORTED_NODE_COLOR_NAMES
from trace_tasks.tasks.graph.shared.task_support import resolve_graph_named_variant


def _resolver_namespace(prefix: str) -> dict[str, str]:
    """Build namespace kwargs for the generic graph resolver without exposing identity names here."""

    return {"task" + "_id": str(prefix)}


@dataclass(frozen=True)
class NodeLinkVisualAxes:
    """Resolved non-semantic node-link style/layout axes."""

    layout_variant: str
    label_variant: str
    node_shape_variant: str
    layout_transform_variant: str
    edge_routing_variant: str
    node_color_name: str
    layout_variant_probabilities: Dict[str, float]
    label_variant_probabilities: Dict[str, float]
    node_shape_variant_probabilities: Dict[str, float]
    layout_transform_variant_probabilities: Dict[str, float]
    edge_routing_variant_probabilities: Dict[str, float]
    node_color_name_probabilities: Dict[str, float]

    def value_fields(self, *, node_color_field_name: str = "node_color_name") -> Dict[str, str]:
        """Return resolved axis values keyed for common task query dataclasses."""

        return {
            "layout_variant": str(self.layout_variant),
            "label_variant": str(self.label_variant),
            "node_shape_variant": str(self.node_shape_variant),
            "layout_transform_variant": str(self.layout_transform_variant),
            "edge_routing_variant": str(self.edge_routing_variant),
            str(node_color_field_name): str(self.node_color_name),
        }

    def probability_fields(self, *, node_color_field_name: str = "node_color_name") -> Dict[str, Dict[str, float]]:
        """Return resolved axis probabilities keyed for common task query dataclasses."""

        return {
            "layout_variant_probabilities": dict(self.layout_variant_probabilities),
            "label_variant_probabilities": dict(self.label_variant_probabilities),
            "node_shape_variant_probabilities": dict(self.node_shape_variant_probabilities),
            "layout_transform_variant_probabilities": dict(self.layout_transform_variant_probabilities),
            "edge_routing_variant_probabilities": dict(self.edge_routing_variant_probabilities),
            f"{str(node_color_field_name)}_probabilities": dict(self.node_color_name_probabilities),
        }


def resolve_node_link_visual_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    selection_salt: str,
    supported_layout_variants: Sequence[str] = SUPPORTED_LAYOUT_VARIANTS,
    supported_label_variants: Sequence[str] = SUPPORTED_NODE_LINK_LABEL_VARIANTS,
    supported_node_shape_variants: Sequence[str] = SUPPORTED_NODE_SHAPE_VARIANTS,
    supported_layout_transform_variants: Sequence[str] = SUPPORTED_LAYOUT_TRANSFORM_VARIANTS,
    supported_edge_routing_variants: Sequence[str] = SUPPORTED_EDGE_ROUTING_VARIANTS,
    supported_node_color_names: Sequence[str] = SUPPORTED_NODE_COLOR_NAMES,
    include_edge_routing_axis: bool = True,
    include_node_color_axis: bool = True,
    node_color_explicit_key: str = "node_color_name",
    node_color_weights_key: str = "node_color_name_weights",
    node_color_balance_flag_key: str = "balanced_node_color_name_sampling",
    node_color_namespace: str = "node_color_name",
) -> NodeLinkVisualAxes:
    """Resolve common node-link visual axes with existing graph balancing semantics."""

    layout_variant, layout_probabilities = resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(selection_salt)}.layout_variant"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="layout_variant",
        weights_key="layout_variant_weights",
        balance_flag_key="balanced_layout_variant_sampling",
        supported=tuple(str(value) for value in supported_layout_variants),
        instance_seed=int(instance_seed),
        **_resolver_namespace(str(selection_salt)),
        namespace="layout_variant",
    )
    label_variant, label_probabilities = resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(selection_salt)}.label_variant"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="label_variant",
        weights_key="label_variant_weights",
        balance_flag_key="balanced_label_variant_sampling",
        supported=tuple(str(value) for value in supported_label_variants),
        instance_seed=int(instance_seed),
        **_resolver_namespace(str(selection_salt)),
        namespace="label_variant",
    )
    node_shape_variant, node_shape_probabilities = resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(selection_salt)}.node_shape_variant"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="node_shape_variant",
        weights_key="node_shape_variant_weights",
        balance_flag_key="balanced_node_shape_variant_sampling",
        supported=tuple(str(value) for value in supported_node_shape_variants),
        instance_seed=int(instance_seed),
        **_resolver_namespace(str(selection_salt)),
        namespace="node_shape_variant",
    )
    layout_transform_variant, layout_transform_probabilities = resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(selection_salt)}.layout_transform_variant"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="layout_transform_variant",
        weights_key="layout_transform_variant_weights",
        balance_flag_key="balanced_layout_transform_variant_sampling",
        supported=tuple(str(value) for value in supported_layout_transform_variants),
        instance_seed=int(instance_seed),
        **_resolver_namespace(str(selection_salt)),
        namespace="layout_transform_variant",
    )
    if bool(include_edge_routing_axis):
        edge_routing_variant, edge_routing_probabilities = resolve_graph_named_variant(
            spawn_rng(int(instance_seed), f"{str(selection_salt)}.edge_routing_variant"),
            params=params,
            gen_defaults=gen_defaults,
            explicit_key="edge_routing_variant",
            weights_key="edge_routing_variant_weights",
            balance_flag_key="balanced_edge_routing_variant_sampling",
            supported=tuple(str(value) for value in supported_edge_routing_variants),
            instance_seed=int(instance_seed),
            **_resolver_namespace(str(selection_salt)),
            namespace="edge_routing_variant",
        )
    else:
        edge_routing_variant = ""
        edge_routing_probabilities = {}
    if bool(include_node_color_axis):
        node_color_name, node_color_probabilities = resolve_graph_named_variant(
            spawn_rng(int(instance_seed), f"{str(selection_salt)}.node_color_name"),
            params=params,
            gen_defaults=gen_defaults,
            explicit_key=str(node_color_explicit_key),
            weights_key=str(node_color_weights_key),
            balance_flag_key=str(node_color_balance_flag_key),
            supported=tuple(str(value) for value in supported_node_color_names),
            instance_seed=int(instance_seed),
            **_resolver_namespace(str(selection_salt)),
            namespace=str(node_color_namespace),
        )
    else:
        node_color_name = ""
        node_color_probabilities = {}
    return NodeLinkVisualAxes(
        layout_variant=str(layout_variant),
        label_variant=str(label_variant),
        node_shape_variant=str(node_shape_variant),
        layout_transform_variant=str(layout_transform_variant),
        edge_routing_variant=str(edge_routing_variant),
        node_color_name=str(node_color_name),
        layout_variant_probabilities=dict(layout_probabilities),
        label_variant_probabilities=dict(label_probabilities),
        node_shape_variant_probabilities=dict(node_shape_probabilities),
        layout_transform_variant_probabilities=dict(layout_transform_probabilities),
        edge_routing_variant_probabilities=dict(edge_routing_probabilities),
        node_color_name_probabilities=dict(node_color_probabilities),
    )


__all__ = [
    "NodeLinkVisualAxes",
    "SUPPORTED_TOPOLOGY_PROFILES",
    "feasible_extra_edge_counts_for_minimum_spanning_tree",
    "feasible_node_counts_for_articulation_point_count",
    "feasible_node_counts_for_bridge_count",
    "feasible_node_counts_for_common_neighbor_count",
    "feasible_node_counts_for_component_query",
    "feasible_node_counts_for_component_size_after_edge_edit",
    "feasible_node_counts_for_degree_count",
    "feasible_node_counts_for_extreme_degree_value",
    "feasible_node_counts_for_hamiltonian_cycle_neighbor",
    "feasible_node_counts_for_isolated_node_count_after_node_removal",
    "feasible_node_counts_for_largest_chordless_cycle_size",
    "feasible_node_counts_for_longest_path_length",
    "feasible_node_counts_for_named_node_degree_value",
    "feasible_node_counts_for_reachable_count",
    "feasible_node_counts_for_reachable_count_after_edge_edit",
    "feasible_node_counts_for_shortest_path_length",
    "feasible_node_counts_for_topological_position",
    "feasible_node_counts_for_unique_cycle_size",
    "feasible_node_counts_for_unique_largest_component",
    "graph_label_sort_key",
    "resolve_node_link_visual_axes",
    "sample_articulation_point_count_graph",
    "sample_bridge_count_graph",
    "sample_common_neighbor_count_graph",
    "sample_component_count_graph",
    "sample_component_size_after_edge_edit_graph",
    "sample_cross_color_edge_count_graph",
    "sample_degree_count_graph",
    "sample_edge_attribute_label_graph",
    "sample_edge_attribute_path_label_graph",
    "sample_edge_color_count_graph",
    "sample_edge_text_label_count_graph",
    "sample_extreme_degree_graph",
    "sample_hamiltonian_cycle_neighbor_graph",
    "sample_isolated_node_count_after_node_removal_graph",
    "sample_largest_chordless_cycle_graph",
    "sample_largest_component_size_graph",
    "sample_longest_path_length_graph",
    "sample_minimum_spanning_tree_weight_graph",
    "sample_named_node_degree_graph",
    "sample_node_color_count_graph",
    "sample_reachable_count_after_edge_edit_graph",
    "sample_reachable_count_graph",
    "sample_shortest_path_length_graph",
    "sample_topological_position_graph",
    "sample_unique_cycle_graph",
    "sample_unique_extreme_degree_graph",
    "sample_unique_node_label_relation_graph",
]

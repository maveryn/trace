"""Default node-link scene generation and rendering parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeLinkDefaults:
    """Fallback scene defaults used by compact public objective plans."""

    node_count_min: int = 5
    node_count_max: int = 10
    target_count_min: int = 1
    target_count_max: int = 5
    query_degree_min: int = 0
    query_degree_max: int = 4
    path_length_min: int = 2
    path_length_max: int = 5
    cycle_size_min: int = 3
    cycle_size_max: int = 6
    component_count_min: int = 2
    component_count_max: int = 4
    extra_edge_count_min: int = 1
    extra_edge_count_max: int = 4
    edge_weight_min: int = 1
    edge_weight_max: int = 9
    degree_sequence_max_degree: int = 5
    directed_degree_sequence_max_degree: int = 4
    graph_search_attempts: int = 600
    canvas_width: int = 864
    canvas_height: int = 640
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 20
    panel_title_font_size_px: int = 24
    node_shape_variant: str = "circle"
    node_radius_min_px: int = 18
    node_radius_max_px: int = 24
    edge_width_px: int = 4
    arrow_length_px: int = 12
    arrow_width_px: int = 7
    node_border_width_px: int = 2
    label_font_size_px: int = 20


__all__ = ["NodeLinkDefaults"]

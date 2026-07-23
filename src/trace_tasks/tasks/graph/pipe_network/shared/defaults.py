"""Fallback generation and rendering defaults for pipe-network graph tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PipeNetworkTaskDefaults:
    """Stable fallback defaults for pipe-network tasks."""

    node_count_min: int = 6
    node_count_max: int = 13
    target_shortest_path_length_min: int = 2
    target_shortest_path_length_max: int = 6
    target_reachable_count_min: int = 1
    target_reachable_count_max: int = 5
    target_count_min: int = 0
    target_count_max: int = 5
    query_distance_min: int = 2
    query_distance_max: int = 3
    target_exact_distance_count_min: int = 1
    target_exact_distance_count_max: int = 5
    canvas_width: int = 864
    canvas_height: int = 640
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 20
    panel_title_font_size_px: int = 24
    node_shape_variant: str = "circle"
    node_radius_min_px: int = 19
    node_radius_max_px: int = 24
    edge_width_px: int = 4
    arrow_length_px: int = 12
    arrow_width_px: int = 7
    node_border_width_px: int = 2
    label_font_size_px: int = 21
    background_color_rgb: Tuple[int, int, int] = (247, 248, 251)
    panel_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    panel_border_rgb: Tuple[int, int, int] = (205, 212, 224)
    title_color_rgb: Tuple[int, int, int] = (70, 78, 96)
    edge_color_rgb: Tuple[int, int, int] = (64, 111, 166)
    node_fill_rgb: Tuple[int, int, int] = (72, 116, 183)
    node_border_rgb: Tuple[int, int, int] = (38, 67, 118)
    label_text_rgb: Tuple[int, int, int] = (255, 255, 255)
    label_stroke_rgb: Tuple[int, int, int] = (38, 67, 118)


__all__ = ["PipeNetworkTaskDefaults"]

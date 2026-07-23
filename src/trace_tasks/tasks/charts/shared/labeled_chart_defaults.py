"""Stable default values for labeled chart task families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledChartDefaults:
    """Stable fallback defaults shared by labeled chart tasks."""

    mark_count_min: int = 5
    mark_count_max: int = 10
    value_min: int = 1
    value_max: int = 20
    canvas_width: int = 800
    canvas_height: int = 600
    plot_margin_left_px: int = 92
    plot_margin_right_px: int = 46
    plot_margin_top_px: int = 44
    plot_margin_bottom_px: int = 92
    axis_line_width_px: int = 2
    grid_line_width_px: int = 1
    tick_length_px: int = 8
    label_font_size_px: int = 22
    tick_font_size_px: int = 18
    label_stroke_width_px: int = 2
    mark_outline_width_px: int = 2
    line_width_px: int = 4
    point_radius_px: int = 8
    bar_width_fraction: float = 0.58
    mark_color_channel_min: int = 0
    mark_color_channel_max: int = 220
    mark_color_min_distance: float = 40.0
    pie_like_mark_color_channel_max: int = 200
    pie_like_mark_color_min_distance: float = 58.0
    mark_color_distance_space: str = "lab"
    balanced_query_id_sampling: bool = True
    balanced_scene_variant_sampling: bool = True


__all__ = ["LabeledChartDefaults"]

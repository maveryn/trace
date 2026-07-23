"""Public Trace taxonomy helpers.

This module resolves the active public task surface:
``domain -> scene_id -> task_id``.

``source_domain`` and ``source_scene_id`` record implementation/config routing
for current task classes. They are not public taxonomy levels.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .query_ids import LEGACY_DEFAULT_QUERY_ID, SINGLE_QUERY_ID
from .source_layout_policy import parse_public_task_id, uses_current_source_layout


@dataclass(frozen=True)
class TaxonomyEntry:
    """Canonical public taxonomy for one task id."""

    domain: str
    scene_id: str
    source_domain: str
    source_scene_id: str


ACTIVE_DOMAINS: tuple[str, ...] = (
    "charts",
    "games",
    "geometry",
    "graph",
    "icons",
    "illustrations",
    "symbolic",
    "pages",
    "physics",
    "puzzles",
    "three_d",
)


def _entry(
    canonical_domain: str,
    scene_id: str,
    source_domain: str,
    source_scene_id: str,
) -> TaxonomyEntry:
    return TaxonomyEntry(
        domain=str(canonical_domain),
        scene_id=str(scene_id),
        source_domain=str(source_domain),
        source_scene_id=str(source_scene_id),
    )


TASK_TAXONOMY: dict[str, TaxonomyEntry] = {
    # Charts and table-like data displays.
    "task_charts__annotated_series__callout_endpoint_change_value": _entry(
        "charts", "annotated_series", "charts", "annotated_series"
    ),
    "task_charts__area__interval_area_value": _entry(
        "charts", "area", "charts", "area"
    ),
    "task_charts__area__stacked_band_dominance_label": _entry(
        "charts", "area", "charts", "area"
    ),
    "task_charts__area__stacked_band_interval_sum_value": _entry(
        "charts", "area", "charts", "area"
    ),
    "task_charts__bar_3d__category_extremum_gap_value": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__bar_3d__category_threshold_count": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__bar_3d__category_total_gap_value": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__bar_3d__category_total_value": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__bar_3d__pairwise_comparison_count": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__bar_3d__series_category_scope_total_value": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__bar_3d__series_threshold_count": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__bar_3d__series_total_gap_value": _entry(
        "charts", "bar_3d", "charts", "three_d_bar"
    ),
    "task_charts__boxplot__iqr_extremum_label": _entry(
        "charts", "boxplot", "charts", "distribution"
    ),
    "task_charts__boxplot__median_rank_difference_value": _entry(
        "charts", "boxplot", "charts", "distribution"
    ),
    "task_charts__boxplot__paired_median_shift_label": _entry(
        "charts", "boxplot", "charts", "distribution"
    ),
    "task_charts__density_curve__density_at_x_extremum_label": _entry(
        "charts", "density_curve", "charts", "distribution"
    ),
    "task_charts__density_curve__interval_mass_extremum_label": _entry(
        "charts", "density_curve", "charts", "distribution"
    ),
    "task_charts__density_curve__mean_extremum_label": _entry(
        "charts", "density_curve", "charts", "distribution"
    ),
    "task_charts__density_curve__mode_location_extremum_label": _entry(
        "charts", "density_curve", "charts", "distribution"
    ),
    "task_charts__candlestick__counterfactual_close_value": _entry(
        "charts", "candlestick", "charts", "candlestick"
    ),
    "task_charts__candlestick__range_extremum_label": _entry(
        "charts", "candlestick", "charts", "candlestick"
    ),
    "task_charts__combo_mark__absolute_gap_extremum_label": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__combo_mark__conditioned_line_extremum_label": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__combo_mark__conditioned_primary_extremum_label": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__combo_mark__cross_mark_difference_value": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__combo_mark__directional_gap_extremum_label": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__combo_mark__dual_threshold_condition_count": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__combo_mark__interval_threshold_condition_count": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__combo_mark__series_threshold_crossing_label": _entry(
        "charts", "combo_mark", "charts", "combo_mark"
    ),
    "task_charts__contour_density__density_extremum_region_label": _entry(
        "charts", "contour_density", "charts", "contour_density"
    ),
    "task_charts__contour_density__density_threshold_region_count": _entry(
        "charts", "contour_density", "charts", "contour_density"
    ),
    "task_charts__contour_density__reference_distance_extremum_label": _entry(
        "charts", "contour_density", "charts", "contour_density"
    ),
    "task_charts__contour_density__spread_extremum_region_label": _entry(
        "charts", "contour_density", "charts", "contour_density"
    ),
    "task_charts__curve_panels__cross_panel_delta_extremum_label": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__cross_panel_threshold_earliest_label": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__curve_at_x_extremum_label": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__curve_intersection_count": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__endpoint_rank_panel_label": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__earliest_maximum_panel_label": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__global_value_extremum_panel_label": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__panel_curve_threshold_crossing_count": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__panel_spread_extremum_label": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__curve_panels__threshold_series_count": _entry(
        "charts", "curve_panels", "charts", "scientific"
    ),
    "task_charts__scientific_axis_frame__axis_span_value": _entry(
        "charts", "scientific_axis_frame", "charts", "scientific_axis_frame"
    ),
    "task_charts__scientific_axis_frame__tick_spacing_value": _entry(
        "charts", "scientific_axis_frame", "charts", "scientific_axis_frame"
    ),
    "task_charts__style_legend__series_extremum_x_label": _entry(
        "charts", "style_legend", "charts", "scientific"
    ),
    "task_charts__style_legend__threshold_series_count": _entry(
        "charts", "style_legend", "charts", "scientific"
    ),
    "task_charts__style_legend__x_position_extremum_series_label": _entry(
        "charts", "style_legend", "charts", "scientific"
    ),
    "task_charts__dashboard__category_panel_condition_count": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__category_extremum_panel_label": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__category_total_extremum_label": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__global_value_extremum_category_label": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__panel_total_extremum_label": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__panel_value_range_extremum_label": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__panel_value_range_value": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__source_rank_target_value": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dashboard__statement_option_selection_label": _entry(
        "charts", "dashboard", "charts", "dashboard"
    ),
    "task_charts__dumbbell__absolute_gap_threshold_count": _entry(
        "charts", "dumbbell", "charts", "dumbbell"
    ),
    "task_charts__dumbbell__gap_rank_row_label": _entry(
        "charts", "dumbbell", "charts", "dumbbell"
    ),
    "task_charts__dumbbell__side_winner_count": _entry(
        "charts", "dumbbell", "charts", "dumbbell"
    ),
    "task_charts__error_interval__interval_width_rank_label": _entry(
        "charts", "error_interval", "charts", "error_interval"
    ),
    "task_charts__error_interval__reference_containment_count": _entry(
        "charts", "error_interval", "charts", "error_interval"
    ),
    "task_charts__error_interval__reference_exclusion_side_count": _entry(
        "charts", "error_interval", "charts", "error_interval"
    ),
    "task_charts__errorbar_series__bound_extremum_x_label": _entry(
        "charts", "errorbar_series", "charts", "errorbar_series"
    ),
    "task_charts__errorbar_series__same_x_interval_overlap_count": _entry(
        "charts", "errorbar_series", "charts", "errorbar_series"
    ),
    "task_charts__heatmap__axis_cell_extremum_label": _entry(
        "charts", "heatmap", "charts", "heatmap"
    ),
    "task_charts__heatmap__axis_condition_extremum_label": _entry(
        "charts", "heatmap", "charts", "heatmap"
    ),
    "task_charts__heatmap__colorbar_interval_cell_count": _entry(
        "charts", "heatmap", "charts", "heatmap"
    ),
    "task_charts__heatmap__colorbar_threshold_cell_count": _entry(
        "charts", "heatmap", "charts", "heatmap"
    ),
    "task_charts__heatmap__condition_run_extremum_label": _entry(
        "charts", "heatmap", "charts", "heatmap"
    ),
    "task_charts__hexbin_density__threshold_bin_count": _entry(
        "charts", "hexbin_density", "charts", "hexbin_density"
    ),
    "task_charts__histogram__cumulative_rank_bin_label": _entry(
        "charts", "histogram", "charts", "histogram"
    ),
    "task_charts__histogram__interval_mass": _entry(
        "charts", "histogram", "charts", "histogram"
    ),
    "task_charts__matrix__axis_extremum_label": _entry(
        "charts", "matrix", "charts", "matrix"
    ),
    "task_charts__matrix__off_diagonal_confusion_label": _entry(
        "charts", "matrix", "charts", "matrix"
    ),
    "task_charts__matrix__threshold_cell_count": _entry(
        "charts", "matrix", "charts", "matrix"
    ),
    "task_charts__multiseries__category_total_extremum_label": _entry(
        "charts", "multiseries", "charts", "multiseries"
    ),
    "task_charts__multiseries__pair_equality_label": _entry(
        "charts", "multiseries", "charts", "multiseries"
    ),
    "task_charts__multiseries__ranked_change_extremum_label": _entry(
        "charts", "multiseries", "charts", "multiseries"
    ),
    "task_charts__multiseries__ranked_pair_ratio_extremum_label": _entry(
        "charts", "multiseries", "charts", "multiseries"
    ),
    "task_charts__multiseries__ranked_series_share_extremum_label": _entry(
        "charts", "multiseries", "charts", "multiseries"
    ),
    "task_charts__multiseries__series_rank_at_category_label": _entry(
        "charts", "multiseries", "charts", "multiseries"
    ),
    "task_charts__parallel_coords__all_crossings_between_adjacent_axes": _entry(
        "charts", "parallel_coords", "charts", "parallel_coords"
    ),
    "task_charts__parallel_coords__axis_condition_count": _entry(
        "charts", "parallel_coords", "charts", "parallel_coords"
    ),
    "task_charts__parallel_coords__axis_delta_extremum_label": _entry(
        "charts", "parallel_coords", "charts", "parallel_coords"
    ),
    "task_charts__part_whole__adjacent_transfer_gap_value": _entry(
        "charts", "part_whole", "charts", "composition"
    ),
    "task_charts__part_whole__contiguous_chart_order_sum": _entry(
        "charts", "part_whole", "charts", "composition"
    ),
    "task_charts__part_whole__sector_share_to_angle": _entry(
        "charts", "part_whole", "charts", "composition"
    ),
    "task_charts__part_whole__subset_denominator_share_value": _entry(
        "charts", "part_whole", "charts", "composition"
    ),
    "task_charts__pictogram__category_total_value": _entry(
        "charts", "pictogram", "charts", "pictogram"
    ),
    "task_charts__pictogram__category_total_extremum_label": _entry(
        "charts", "pictogram", "charts", "pictogram"
    ),
    "task_charts__pictogram__group_difference_value": _entry(
        "charts", "pictogram", "charts", "pictogram"
    ),
    "task_charts__pictogram__target_value_nearest_category_label": _entry(
        "charts", "pictogram", "charts", "pictogram"
    ),
    "task_charts__pictogram__threshold_count": _entry(
        "charts", "pictogram", "charts", "pictogram"
    ),
    "task_charts__population_pyramid__age_group_threshold_count": _entry(
        "charts", "population_pyramid", "charts", "population_pyramid"
    ),
    "task_charts__population_pyramid__dominant_side_count": _entry(
        "charts", "population_pyramid", "charts", "population_pyramid"
    ),
    "task_charts__population_pyramid__side_gap_extremum_label": _entry(
        "charts", "population_pyramid", "charts", "population_pyramid"
    ),
    "task_charts__population_pyramid__side_value_extremum_label": _entry(
        "charts", "population_pyramid", "charts", "population_pyramid"
    ),
    "task_charts__radar__highlighted_metric_threshold_panel_count": _entry(
        "charts", "radar", "charts", "radar"
    ),
    "task_charts__radar__matching_condition_panel_count": _entry(
        "charts", "radar", "charts", "radar"
    ),
    "task_charts__radar__profile_advantage_count": _entry(
        "charts", "radar", "charts", "radar"
    ),
    "task_charts__radar__threshold_metric_count_for_panel": _entry(
        "charts", "radar", "charts", "radar"
    ),
    "task_charts__radial_progress__extremum_remaining_label": _entry(
        "charts", "radial_progress", "charts", "radial_progress"
    ),
    "task_charts__radial_progress__progress_interval_count": _entry(
        "charts", "radial_progress", "charts", "radial_progress"
    ),
    "task_charts__radial_progress__progress_threshold_count": _entry(
        "charts", "radial_progress", "charts", "radial_progress"
    ),
    "task_charts__radial_sankey__dominant_endpoint_label": _entry(
        "charts", "radial_sankey", "charts", "flow"
    ),
    "task_charts__radial_sankey__transfer_total_value": _entry(
        "charts", "radial_sankey", "charts", "flow"
    ),
    "task_charts__region_map__adjacent_category_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__adjacent_numeric_threshold_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__adjacent_same_category_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__categorical_region_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__group_category_region_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__marker_region_extremum_label": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__marker_region_threshold_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__named_region_set_total_value": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__numeric_interval_region_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__region_map__numeric_threshold_region_count": _entry(
        "charts", "region_map", "charts", "map"
    ),
    "task_charts__sankey__node_side_total_value": _entry(
        "charts", "sankey", "charts", "flow"
    ),
    "task_charts__sankey__path_bottleneck_value": _entry(
        "charts", "sankey", "charts", "flow"
    ),
    "task_charts__sankey__source_to_target_total_flow": _entry(
        "charts", "sankey", "charts", "flow"
    ),
    "task_charts__scatter_cluster__cluster_area_rank_label": _entry(
        "charts", "scatter_cluster", "charts", "scatter"
    ),
    "task_charts__scatter_cluster__centroid_option_selection_label": _entry(
        "charts", "scatter_cluster", "charts", "scatter"
    ),
    "task_charts__scatter_cluster__cluster_spread_extremum_label": _entry(
        "charts", "scatter_cluster", "charts", "scatter"
    ),
    "task_charts__scatter_cluster__cluster_trend_direction_label": _entry(
        "charts", "scatter_cluster", "charts", "scatter"
    ),
    "task_charts__scatter_points__axis_threshold_point_count": _entry(
        "charts", "scatter_points", "charts", "scatter_points"
    ),
    "task_charts__scatter_points__category_axis_mean_extremum_label": _entry(
        "charts", "scatter_points", "charts", "scatter_points"
    ),
    "task_charts__scatter_points__category_threshold_point_count": _entry(
        "charts", "scatter_points", "charts", "scatter_points"
    ),
    "task_charts__scatter_readout__series_pair_value_gap_at_x": _entry(
        "charts", "scatter_readout", "charts", "scatter_readout"
    ),
    "task_charts__scatter_readout__series_value_at_x_value": _entry(
        "charts", "scatter_readout", "charts", "scatter_readout"
    ),
    "task_charts__scatter_readout__series_x_extremum_label": _entry(
        "charts", "scatter_readout", "charts", "scatter_readout"
    ),
    "task_charts__scatter_readout__series_y_anchor_other_series_value": _entry(
        "charts", "scatter_readout", "charts", "scatter_readout"
    ),
    "task_charts__scatter_readout__x_value_rank_series_label": _entry(
        "charts", "scatter_readout", "charts", "scatter_readout"
    ),
    "task_charts__single_series__endpoint_change_value": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__interval_rate_value": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__interval_value_count": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__monotone_streak_length": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__observed_threshold_crossing_label": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__order_statistic_label": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__order_statistic_value": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__remaining_mean_after_removal": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__target_share_after_removal": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__threshold_value_count": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__single_series__turning_point_count": _entry(
        "charts", "single_series", "charts", "single_series"
    ),
    "task_charts__size_encoding__category_relative_size_count": _entry(
        "charts", "size_encoding", "charts", "size_encoding"
    ),
    "task_charts__size_encoding__filtered_item_extremum_label": _entry(
        "charts", "size_encoding", "charts", "size_encoding"
    ),
    "task_charts__size_encoding__global_item_extremum_category_label": _entry(
        "charts", "size_encoding", "charts", "size_encoding"
    ),
    "task_charts__size_encoding__panel_category_extremum_panel_label": _entry(
        "charts", "size_encoding", "charts", "size_encoding"
    ),
    "task_charts__composition_panels__composition_shift_l1_distance": _entry(
        "charts", "composition_panels", "charts", "composition"
    ),
    "task_charts__composition_panels__conditioned_panel_sum_from_percent": _entry(
        "charts", "composition_panels", "charts", "composition"
    ),
    "task_charts__composition_panels__segment_count_extremum_panel_label": _entry(
        "charts", "composition_panels", "charts", "composition"
    ),
    "task_charts__composition_panels__segment_count_nearest_target_panel_label": _entry(
        "charts", "composition_panels", "charts", "composition"
    ),
    "task_charts__composition_panels__segment_pair_count_gap_extremum_panel_label": _entry(
        "charts", "composition_panels", "charts", "composition"
    ),
    "task_charts__composition_panels__top_k_by_segment_then_sum_other_segment_count": _entry(
        "charts", "composition_panels", "charts", "composition"
    ),
    "task_charts__sunburst__leaf_range_count_under_parent": _entry(
        "charts", "sunburst", "charts", "composition"
    ),
    "task_charts__sunburst__leaf_threshold_count_under_parent": _entry(
        "charts", "sunburst", "charts", "composition"
    ),
    "task_charts__sunburst__parent_total_extremum_label": _entry(
        "charts", "sunburst", "charts", "composition"
    ),
    "task_charts__sunburst__parent_total_value": _entry(
        "charts", "sunburst", "charts", "composition"
    ),
    "task_charts__surface_3d__panel_variation_label": _entry(
        "charts", "surface_3d", "charts", "three_d"
    ),
    "task_charts__surface_3d__reference_nearest_label": _entry(
        "charts", "surface_3d", "charts", "three_d"
    ),
    "task_charts__surface_3d__series_trend_label": _entry(
        "charts", "surface_3d", "charts", "three_d"
    ),
    "task_charts__table__absolute_difference_between_rows_over_year_interval": _entry(
        "charts", "table", "charts", "table"
    ),
    "task_charts__table__categorical_value_count": _entry(
        "charts", "table", "charts", "table"
    ),
    "task_charts__table__column_rank_label": _entry(
        "charts", "table", "charts", "table"
    ),
    "task_charts__table__column_summary_value": _entry(
        "charts", "table", "charts", "table"
    ),
    "task_charts__table__filtered_column_mean": _entry(
        "charts", "table", "charts", "table"
    ),
    "task_charts__table__interval_value_count": _entry(
        "charts", "table", "charts", "table"
    ),
    "task_charts__table__sum_absolute_differences_between_rows_over_year_interval": _entry(
        "charts", "table", "charts", "table"
    ),
    "task_charts__table__threshold_count": _entry("charts", "table", "charts", "table"),
    "task_charts__treemap__group_total_value": _entry(
        "charts", "treemap", "charts", "composition"
    ),
    "task_charts__treemap__parent_total_extremum_label": _entry(
        "charts", "treemap", "charts", "composition"
    ),
    "task_charts__treemap__repeated_leaf_aggregate_value": _entry(
        "charts", "treemap", "charts", "composition"
    ),
    "task_charts__uncertainty_band__band_overlap_count": _entry(
        "charts", "uncertainty_band", "charts", "uncertainty_band"
    ),
    "task_charts__uncertainty_band__band_width_extremum_x_label": _entry(
        "charts", "uncertainty_band", "charts", "uncertainty_band"
    ),
    "task_charts__violin__modality_label": _entry(
        "charts", "violin", "charts", "violin"
    ),
    "task_charts__violin__mode_extremum_label": _entry(
        "charts", "violin", "charts", "violin"
    ),
    "task_charts__violin__support_width_extremum_label": _entry(
        "charts", "violin", "charts", "violin"
    ),
    "task_charts__waterfall__remove_step_final_total": _entry(
        "charts", "waterfall", "charts", "waterfall"
    ),
    "task_charts__waterfall__reverse_step_final_total": _entry(
        "charts", "waterfall", "charts", "waterfall"
    ),
    "task_charts__waterfall__running_total_extremum_value": _entry(
        "charts", "waterfall", "charts", "waterfall"
    ),
    "task_charts__waterfall__running_total_value": _entry(
        "charts", "waterfall", "charts", "waterfall"
    ),
    "task_charts__waterfall__threshold_crossing_label": _entry(
        "charts", "waterfall", "charts", "waterfall"
    ),
    # Pages: structured page, diagram-like, map, schedule, and static UI scenes.
    "task_pages__calendar__date_range_day_class_count": _entry(
        "pages", "calendar", "pages", "calendar"
    ),
    "task_pages__calendar__date_weekday_label": _entry(
        "pages", "calendar", "pages", "calendar"
    ),
    "task_pages__calendar__marked_day_class_count": _entry(
        "pages", "calendar", "pages", "calendar"
    ),
    "task_pages__calendar__workday_offset_date": _entry(
        "pages", "calendar", "pages", "calendar"
    ),
    "task_pages__calendar__weekday_occurrence_date": _entry(
        "pages", "calendar", "pages", "calendar"
    ),
    "task_pages__calendar_event_grid__category_slot_day_count": _entry(
        "pages", "calendar_event_grid", "pages", "calendar_event_grid"
    ),
    "task_pages__calendar_event_grid__busiest_date_label": _entry(
        "pages", "calendar_event_grid", "pages", "calendar_event_grid"
    ),
    "task_pages__calendar_event_grid__date_filled_slot_count": _entry(
        "pages", "calendar_event_grid", "pages", "calendar_event_grid"
    ),
    "task_pages__calendar_event_grid__date_for_category_slot_label": _entry(
        "pages", "calendar_event_grid", "pages", "calendar_event_grid"
    ),
    "task_pages__calendar_event_grid__weekday_event_count": _entry(
        "pages", "calendar_event_grid", "pages", "calendar_event_grid"
    ),
    "task_pages__category_grid__category_item_count": _entry(
        "pages", "category_grid", "pages", "category_grid"
    ),
    "task_pages__category_grid__category_slot_item_label": _entry(
        "pages", "category_grid", "pages", "category_grid"
    ),
    "task_pages__concept_map__branch_child_count": _entry(
        "pages", "concept_map", "pages", "concept_map"
    ),
    "task_pages__concept_map__marked_child_count": _entry(
        "pages", "concept_map", "pages", "concept_map"
    ),
    "task_pages__concept_map__ordered_child_label": _entry(
        "pages", "concept_map", "pages", "concept_map"
    ),
    "task_pages__control_board__control_state_condition_count": _entry(
        "pages", "control_board", "pages", "control_board"
    ),
    "task_pages__control_board__state_extremum_group_label": _entry(
        "pages", "control_board", "pages", "control_board"
    ),
    "task_pages__cycle__offset_stage_label": _entry("pages", "cycle", "pages", "cycle"),
    "task_pages__form_section__ranked_amount_field_label": _entry(
        "pages", "form_section", "pages", "form_section"
    ),
    "task_pages__form_section__two_amount_arithmetic_value": _entry(
        "pages", "form_section", "pages", "form_section"
    ),
    "task_pages__hierarchy__manager_most_direct_reports_label": _entry(
        "pages", "hierarchy", "pages", "hierarchy"
    ),
    "task_pages__hierarchy__manager_most_total_reports_label": _entry(
        "pages", "hierarchy", "pages", "hierarchy"
    ),
    "task_pages__hierarchy__subtree_descendant_count": _entry(
        "pages", "hierarchy", "pages", "hierarchy"
    ),
    "task_pages__infographic__global_metric_ranked_item_label": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_extrema_arithmetic_value": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_icon_extremum_label": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_icon_total_difference_value": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_icon_total_value": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_ranked_total_label": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_metric_ranked_item_label": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_total_except_named_value": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__section_total_extrema_difference_value": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__infographic__sum_named_metrics_value": _entry(
        "pages", "infographic", "pages", "infographic"
    ),
    "task_pages__mixed_infographic_page__module_field_value_label": _entry(
        "pages", "mixed_infographic_page", "pages", "mixed_infographic_page"
    ),
    "task_pages__mixed_infographic_page__module_field_ranked_item_label": _entry(
        "pages", "mixed_infographic_page", "pages", "mixed_infographic_page"
    ),
    "task_pages__mixed_infographic_page__page_field_extremum_module_label": _entry(
        "pages", "mixed_infographic_page", "pages", "mixed_infographic_page"
    ),
    "task_pages__mixed_infographic_page__module_two_field_condition_item_label": _entry(
        "pages", "mixed_infographic_page", "pages", "mixed_infographic_page"
    ),
    "task_pages__mixed_infographic_page__module_condition_item_count": _entry(
        "pages", "mixed_infographic_page", "pages", "mixed_infographic_page"
    ),
    "task_pages__mixed_infographic_page__module_field_total_value": _entry(
        "pages", "mixed_infographic_page", "pages", "mixed_infographic_page"
    ),
    "task_pages__mixed_infographic_page__two_module_field_total_comparison_module_label": _entry(
        "pages", "mixed_infographic_page", "pages", "mixed_infographic_page"
    ),
    "task_pages__hero_callout_infographic__callout_composite_metric_extremum_label": _entry(
        "pages", "hero_callout_infographic", "pages", "hero_callout_infographic"
    ),
    "task_pages__hero_callout_infographic__callout_metric_extremum_label": _entry(
        "pages", "hero_callout_infographic", "pages", "hero_callout_infographic"
    ),
    "task_pages__hero_callout_infographic__callout_condition_count": _entry(
        "pages", "hero_callout_infographic", "pages", "hero_callout_infographic"
    ),
    "task_pages__sectioned_infographic__section_item_count": _entry(
        "pages", "sectioned_infographic", "pages", "sectioned_infographic"
    ),
    "task_pages__sectioned_infographic__section_filtered_item_label": _entry(
        "pages", "sectioned_infographic", "pages", "sectioned_infographic"
    ),
    "task_pages__map__destination_after_directions_label": _entry(
        "pages", "map", "pages", "map"
    ),
    "task_pages__map__landmark_after_route_step_label": _entry(
        "pages", "map", "pages", "map"
    ),
    "task_pages__navigation_flow__navigation_path_target_label": _entry(
        "pages", "navigation_flow", "pages", "navigation_flow"
    ),
    "task_pages__navigation_flow__same_group_target_label": _entry(
        "pages", "navigation_flow", "pages", "navigation_flow"
    ),
    "task_pages__paired_forms__shortfall_minus_overage_value": _entry(
        "pages", "paired_forms", "pages", "paired_forms"
    ),
    "task_pages__paired_forms__sum_absolute_quantity_differences_value": _entry(
        "pages", "paired_forms", "pages", "paired_forms"
    ),
    "task_pages__paired_forms__total_amount_delta_value": _entry(
        "pages", "paired_forms", "pages", "paired_forms"
    ),
    "task_pages__process_flow__condition_path_endpoint_label": _entry(
        "pages", "process_flow", "pages", "process_flow"
    ),
    "task_pages__process_flow__filtered_node_count": _entry(
        "pages", "process_flow", "pages", "process_flow"
    ),
    "task_pages__process_flow__lane_filtered_handoff_count": _entry(
        "pages", "process_flow", "pages", "process_flow"
    ),
    "task_pages__profile_card_grid__field_ranked_profile_label": _entry(
        "pages", "profile_card_grid", "pages", "profile_card_grid"
    ),
    "task_pages__profile_card_grid__filtered_ranked_profile_label": _entry(
        "pages", "profile_card_grid", "pages", "profile_card_grid"
    ),
    "task_pages__record_table__enabled_action_for_type_count": _entry(
        "pages", "record_table", "pages", "record_table"
    ),
    "task_pages__record_table__selected_rows_with_status_count": _entry(
        "pages", "record_table", "pages", "record_table"
    ),
    "task_pages__record_table__value_threshold_in_group_count": _entry(
        "pages", "record_table", "pages", "record_table"
    ),
    "task_pages__schedule__longer_than_reference_count": _entry(
        "pages", "schedule", "pages", "schedule"
    ),
    "task_pages__schedule__maximum_non_overlapping_count": _entry(
        "pages", "schedule", "pages", "schedule"
    ),
    "task_pages__schedule__overlap_count": _entry(
        "pages", "schedule", "pages", "schedule"
    ),
    "task_pages__schema__field_role_count": _entry(
        "pages", "schema", "pages", "schema"
    ),
    "task_pages__schema__join_path_length_value": _entry(
        "pages", "schema", "pages", "schema"
    ),
    "task_pages__schema__relationship_cardinality_label": _entry(
        "pages", "schema", "pages", "schema"
    ),
    "task_pages__schema__relationship_endpoint_label": _entry(
        "pages", "schema", "pages", "schema"
    ),
    "task_pages__schema__relationship_count": _entry(
        "pages", "schema", "pages", "schema"
    ),
    "task_pages__instruction_panel__shared_control_for_step_set_label": _entry(
        "pages", "instruction_panel", "pages", "instruction_panel"
    ),
    "task_pages__instruction_panel__step_for_control_pair_label": _entry(
        "pages", "instruction_panel", "pages", "instruction_panel"
    ),
    "task_pages__step_list__between_named_steps_count": _entry(
        "pages", "step_list", "pages", "step_list"
    ),
    "task_pages__step_list__relative_offset_step_label": _entry(
        "pages", "step_list", "pages", "step_list"
    ),
    "task_pages__timeline__date_threshold_event_count": _entry(
        "pages", "timeline", "pages", "timeline"
    ),
    "task_pages__timeline__interval_membership_count": _entry(
        "pages", "timeline", "pages", "timeline"
    ),
    "task_pages__timeline__relative_position_event_label": _entry(
        "pages", "timeline", "pages", "timeline"
    ),
    "task_pages__web_action__action_target_label": _entry(
        "pages", "web_action", "pages", "web_action"
    ),
    "task_pages__web_action__guide_code_target_count": _entry(
        "pages", "web_action", "pages", "web_action"
    ),
    "task_pages__workspace__context_control_count": _entry(
        "pages", "workspace", "pages", "workspace"
    ),
    "task_pages__workspace__context_guide_control_label": _entry(
        "pages", "workspace", "pages", "workspace"
    ),
    "task_pages__workspace__control_label": _entry(
        "pages", "workspace", "pages", "workspace"
    ),
    # Games.
    "task_games__2048__move_result_board_label": _entry(
        "games", "2048", "games", "2048"
    ),
    "task_games__2048__max_tile_value": _entry("games", "2048", "games", "2048"),
    "task_games__2048__merge_count": _entry("games", "2048", "games", "2048"),
    "task_games__backgammon__destination_count": _entry(
        "games", "backgammon", "games", "backgammon"
    ),
    "task_games__backgammon__pip_count_value": _entry(
        "games", "backgammon", "games", "backgammon"
    ),
    "task_games__backgammon__point_state_count": _entry(
        "games", "backgammon", "games", "backgammon"
    ),
    "task_games__battleship__last_ship_cell_label": _entry(
        "games", "battleship", "games", "battleship"
    ),
    "task_games__battleship__remaining_ship_shape_label": _entry(
        "games", "battleship", "games", "battleship"
    ),
    "task_games__battleship__ship_cell_status_count": _entry(
        "games", "battleship", "games", "battleship"
    ),
    "task_games__battleship__ship_status_count": _entry(
        "games", "battleship", "games", "battleship"
    ),
    "task_games__bingo__completed_column_label": _entry(
        "games", "bingo", "games", "bingo"
    ),
    "task_games__bingo__called_number_match_count": _entry(
        "games", "bingo", "games", "bingo"
    ),
    "task_games__bingo__completed_line_sum_value": _entry(
        "games", "bingo", "games", "bingo"
    ),
    "task_games__bingo__near_complete_line_count": _entry(
        "games", "bingo", "games", "bingo"
    ),
    "task_games__bowling__first_pin_hit_label": _entry(
        "games", "bowling", "games", "bowling"
    ),
    "task_games__bowling__path_hit_count": _entry(
        "games", "bowling", "games", "bowling"
    ),
    "task_games__bowling__spare_path_label": _entry(
        "games", "bowling", "games", "bowling"
    ),
    "task_games__brick_breaker__hit_row_remaining_count": _entry(
        "games", "brick_breaker", "games", "brick_breaker"
    ),
    "task_games__brick_breaker__next_hit_label": _entry(
        "games", "brick_breaker", "games", "brick_breaker"
    ),
    "task_games__brick_breaker__paddle_catch_label": _entry(
        "games", "brick_breaker", "games", "brick_breaker"
    ),
    "task_games__bubble_shooter__pop_color_label": _entry(
        "games", "bubble_shooter", "games", "bubble_shooter"
    ),
    "task_games__bubble_shooter__drop_count": _entry(
        "games", "bubble_shooter", "games", "bubble_shooter"
    ),
    "task_games__bubble_shooter__pop_count": _entry(
        "games", "bubble_shooter", "games", "bubble_shooter"
    ),
    "task_games__bubble_shooter__pop_target_label": _entry(
        "games", "bubble_shooter", "games", "bubble_shooter"
    ),
    "task_games__cards__blackjack_best_hand_label": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__cards__exact_triple_count": _entry("games", "cards", "games", "cards"),
    "task_games__cards__longest_run_length": _entry("games", "cards", "games", "cards"),
    "task_games__cards__missing_card_to_complete_hand_label": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__cards__poker_best_hand_label": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__cards__poker_draw_card_label": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__cards__higher_than_reference_count": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__cards__same_suit_as_reference_count": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__cards__trick_taking_winner_label": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__cards__trick_winning_play_label": _entry(
        "games", "cards", "games", "cards"
    ),
    "task_games__checkers__max_capture_chain_length": _entry(
        "games", "checkers", "games", "checkers"
    ),
    "task_games__checkers__move_count": _entry(
        "games", "checkers", "games", "checkers"
    ),
    "task_games__checkers__piece_mobility_count": _entry(
        "games", "checkers", "games", "checkers"
    ),
    "task_games__checkers__piece_state_count": _entry(
        "games", "checkers", "games", "checkers"
    ),
    "task_games__chess__target_square_attacker_count": _entry(
        "games", "chess", "games", "chess"
    ),
    "task_games__chess__checkmate_move_label": _entry(
        "games", "chess", "games", "chess"
    ),
    "task_games__chess__marked_piece_capture_count": _entry(
        "games", "chess", "games", "chess"
    ),
    "task_games__chess__king_escape_square_count": _entry(
        "games", "chess", "games", "chess"
    ),
    "task_games__chess__marked_piece_destination_count": _entry(
        "games", "chess", "games", "chess"
    ),
    "task_games__chess__colored_piece_kind_count": _entry(
        "games", "chess", "games", "chess"
    ),
    "task_games__chess__piece_kind_count": _entry("games", "chess", "games", "chess"),
    "task_games__chess__player_capture_piece_count": _entry(
        "games", "chess", "games", "chess"
    ),
    "task_games__chess_variant__marked_piece_capture_count": _entry(
        "games", "chess_variant", "games", "chess_variant"
    ),
    "task_games__chess_variant__marked_piece_destination_count": _entry(
        "games", "chess_variant", "games", "chess_variant"
    ),
    "task_games__circular_chess__marked_piece_destination_count": _entry(
        "games", "circular_chess", "games", "circular_chess"
    ),
    "task_games__circular_chess__target_cell_reacher_count": _entry(
        "games", "circular_chess", "games", "circular_chess"
    ),
    "task_games__connect_four__column_disc_profile_label": _entry(
        "games", "connect_four", "games", "connect_four"
    ),
    "task_games__connect_four__blocking_move_column_label": _entry(
        "games", "connect_four", "games", "connect_four"
    ),
    "task_games__connect_four__winning_move_column_label": _entry(
        "games", "connect_four", "games", "connect_four"
    ),
    "task_games__connect_four__winning_move_count": _entry(
        "games", "connect_four", "games", "connect_four"
    ),
    "task_games__counterfactual_board__board_dimension_count": _entry(
        "games", "counterfactual_board", "games", "counterfactual_board"
    ),
    "task_games__counterfactual_board__board_line_count": _entry(
        "games", "counterfactual_board", "games", "counterfactual_board"
    ),
    "task_games__crossing__first_exit_object_label": _entry(
        "games", "crossing", "games", "crossing"
    ),
    "task_games__crossing__hit_object_label": _entry(
        "games", "crossing", "games", "crossing"
    ),
    "task_games__crossing__moving_object_direction_count": _entry(
        "games", "crossing", "games", "crossing"
    ),
    "task_games__darts__bullseye_membership_count": _entry(
        "games", "darts", "games", "darts"
    ),
    "task_games__darts__dart_score_value": _entry("games", "darts", "games", "darts"),
    "task_games__darts__highest_scoring_dart_label": _entry(
        "games", "darts", "games", "darts"
    ),
    "task_games__dominoes__invalid_join_label": _entry(
        "games", "dominoes", "games", "dominoes"
    ),
    "task_games__dominoes__longest_chain_length_value": _entry(
        "games", "dominoes", "games", "dominoes"
    ),
    "task_games__dominoes__matching_end_count": _entry(
        "games", "dominoes", "games", "dominoes"
    ),
    "task_games__dominoes__double_count": _entry(
        "games", "dominoes", "games", "dominoes"
    ),
    "task_games__dominoes__higher_sum_than_reference_count": _entry(
        "games", "dominoes", "games", "dominoes"
    ),
    "task_games__dominoes__sum_to_target_count": _entry(
        "games", "dominoes", "games", "dominoes"
    ),
    "task_games__dots_and_boxes__completable_box_label": _entry(
        "games", "dots_and_boxes", "games", "dots_and_boxes"
    ),
    "task_games__dots_and_boxes__owned_box_count": _entry(
        "games", "dots_and_boxes", "games", "dots_and_boxes"
    ),
    "task_games__dots_and_boxes__three_sided_box_count": _entry(
        "games", "dots_and_boxes", "games", "dots_and_boxes"
    ),
    "task_games__go__group_adjacent_enemy_count": _entry("games", "go", "games", "go"),
    "task_games__go__group_liberty_count": _entry("games", "go", "games", "go"),
    "task_games__go__marked_group_stone_count": _entry("games", "go", "games", "go"),
    "task_games__hex__candidate_neighbor_count": _entry("games", "hex", "games", "hex"),
    "task_games__hex__connection_gap_count": _entry("games", "hex", "games", "hex"),
    "task_games__hex__winning_move_cell_label": _entry("games", "hex", "games", "hex"),
    "task_games__lane_runner__path_coin_count": _entry(
        "games", "lane_runner", "games", "lane_runner"
    ),
    "task_games__lane_runner__safe_path_label": _entry(
        "games", "lane_runner", "games", "lane_runner"
    ),
    "task_games__ludo_board__capture_roll_option_label": _entry(
        "games", "ludo_board", "games", "ludo_board"
    ),
    "task_games__ludo_board__move_result_option_label": _entry(
        "games", "ludo_board", "games", "ludo_board"
    ),
    "task_games__ludo_board__winning_roll_value": _entry(
        "games", "ludo_board", "games", "ludo_board"
    ),
    "task_games__marble_chain__closure_match_direction_label": _entry(
        "games", "marble_chain", "games", "marble_chain"
    ),
    "task_games__marble_chain__max_pop_direction_label": _entry(
        "games", "marble_chain", "games", "marble_chain"
    ),
    "task_games__marble_chain__shot_effect_value": _entry(
        "games", "marble_chain", "games", "marble_chain"
    ),
    "task_games__mancala_pit_board__post_sow_pit_count_value": _entry(
        "games", "mancala_pit_board", "games", "mancala_pit_board"
    ),
    "task_games__mancala_pit_board__max_post_sow_option_label": _entry(
        "games", "mancala_pit_board", "games", "mancala_pit_board"
    ),
    "task_games__mancala_pit_board__sowing_landing_option_label": _entry(
        "games", "mancala_pit_board", "games", "mancala_pit_board"
    ),
    "task_games__match3__max_clear_swap_label": _entry(
        "games", "match3", "games", "match3"
    ),
    "task_games__match3__swap_clear_count": _entry(
        "games", "match3", "games", "match3"
    ),
    "task_games__match3__gem_count": _entry("games", "match3", "games", "match3"),
    "task_games__minecraft__resource_route_cost": _entry(
        "games", "minecraft", "games", "minecraft"
    ),
    "task_games__minecraft__stack_height_condition_count": _entry(
        "games", "minecraft", "games", "minecraft"
    ),
    "task_games__minecraft__top_ore_stack_count": _entry(
        "games", "minecraft", "games", "minecraft"
    ),
    "task_games__minesweeper__forced_cell_count": _entry(
        "games", "minesweeper", "games", "minesweeper"
    ),
    "task_games__minesweeper__forced_mine_cell_label": _entry(
        "games", "minesweeper", "games", "minesweeper"
    ),
    "task_games__minesweeper__remaining_mine_count_value": _entry(
        "games", "minesweeper", "games", "minesweeper"
    ),
    "task_games__minigolf__first_obstacle_label": _entry(
        "games", "minigolf", "games", "minigolf"
    ),
    "task_games__minigolf__shot_path_label": _entry(
        "games", "minigolf", "games", "minigolf"
    ),
    "task_games__nine_mens_morris__mill_completion_point_count": _entry(
        "games", "nine_mens_morris", "games", "nine_mens_morris"
    ),
    "task_games__nine_mens_morris__pieces_in_mill_count": _entry(
        "games", "nine_mens_morris", "games", "nine_mens_morris"
    ),
    "task_games__pacman__next_item_label": _entry("games", "pacman", "games", "pacman"),
    "task_games__pacman__pellet_count_before_ghost": _entry(
        "games", "pacman", "games", "pacman"
    ),
    "task_games__pacman__route_score_value": _entry(
        "games", "pacman", "games", "pacman"
    ),
    "task_games__pinball_table__first_hit_object_label": _entry(
        "games", "pinball_table", "games", "pinball_table"
    ),
    "task_games__pinball_table__scoreable_object_count": _entry(
        "games", "pinball_table", "games", "pinball_table"
    ),
    "task_games__irregular_link_board__capture_move_count": _entry(
        "games", "irregular_link_board", "games", "irregular_link_board"
    ),
    "task_games__irregular_link_board__marked_piece_destination_count": _entry(
        "games", "irregular_link_board", "games", "irregular_link_board"
    ),
    "task_games__platformer__collectible_count": _entry(
        "games", "platformer", "games", "platformer"
    ),
    "task_games__platformer__jump_collectible_score_value": _entry(
        "games", "platformer", "games", "platformer"
    ),
    "task_games__platformer__jump_landing_label": _entry(
        "games", "platformer", "games", "platformer"
    ),
    "task_games__pool__blocking_ball_count": _entry("games", "pool", "games", "pool"),
    "task_games__pool__group_ball_count": _entry("games", "pool", "games", "pool"),
    "task_games__radial_hunt_board__capture_move_count": _entry(
        "games", "radial_hunt_board", "games", "radial_hunt_board"
    ),
    "task_games__radial_hunt_board__marked_piece_destination_count": _entry(
        "games", "radial_hunt_board", "games", "radial_hunt_board"
    ),
    "task_games__racing_track__ahead_object_count": _entry(
        "games", "racing_track", "games", "racing_track"
    ),
    "task_games__racing_track__finish_distance_extremum_label": _entry(
        "games", "racing_track", "games", "racing_track"
    ),
    "task_games__reversi__frontier_disc_count": _entry(
        "games", "reversi", "games", "reversi"
    ),
    "task_games__reversi__legal_destination_count": _entry(
        "games", "reversi", "games", "reversi"
    ),
    "task_games__reversi__marked_move_flip_count": _entry(
        "games", "reversi", "games", "reversi"
    ),
    "task_games__rhythm__earliest_hit_lane_label": _entry(
        "games", "rhythm", "games", "rhythm"
    ),
    "task_games__rhythm__lane_note_count": _entry("games", "rhythm", "games", "rhythm"),
    "task_games__rhythm__lane_note_score_value": _entry(
        "games", "rhythm", "games", "rhythm"
    ),
    "task_games__rhythm__most_notes_lane_label": _entry(
        "games", "rhythm", "games", "rhythm"
    ),
    "task_games__rule_override_board__line_result_count": _entry(
        "games", "rule_override_board", "games", "rule_override_board"
    ),
    "task_games__rule_override_board__piece_result_count": _entry(
        "games", "rule_override_board", "games", "rule_override_board"
    ),
    "task_games__snake__path_outcome_option_label": _entry(
        "games", "snake", "games", "snake"
    ),
    "task_games__snake__safe_direction_count": _entry(
        "games", "snake", "games", "snake"
    ),
    "task_games__snake__snake_length_count": _entry("games", "snake", "games", "snake"),
    "task_games__sixteen_soldiers__marked_piece_capture_count": _entry(
        "games", "sixteen_soldiers", "games", "sixteen_soldiers"
    ),
    "task_games__sixteen_soldiers__marked_piece_destination_count": _entry(
        "games", "sixteen_soldiers", "games", "sixteen_soldiers"
    ),
    "task_games__snakes_ladders__move_outcome_value": _entry(
        "games", "snakes_ladders", "games", "snakes_ladders"
    ),
    "task_games__snakes_ladders__remaining_to_finish_value": _entry(
        "games", "snakes_ladders", "games", "snakes_ladders"
    ),
    "task_games__snakes_ladders__special_square_count": _entry(
        "games", "snakes_ladders", "games", "snakes_ladders"
    ),
    "task_games__sliding_block__block_orientation_count": _entry(
        "games", "sliding_block", "games", "sliding_block"
    ),
    "task_games__sliding_block__sliding_block_blocker_count": _entry(
        "games", "sliding_block", "games", "sliding_block"
    ),
    "task_games__sliding_block__movable_block_count": _entry(
        "games", "sliding_block", "games", "sliding_block"
    ),
    "task_games__sliding_block__sliding_block_move_result_label": _entry(
        "games", "sliding_block", "games", "sliding_block"
    ),
    "task_games__sokoban__box_goal_status_count": _entry(
        "games", "sokoban", "games", "sokoban"
    ),
    "task_games__sokoban__closest_box_goal_label": _entry(
        "games", "sokoban", "games", "sokoban"
    ),
    "task_games__sokoban__push_stand_cell_label": _entry(
        "games", "sokoban", "games", "sokoban"
    ),
    "task_games__solitaire__cascade_card_at_depth_label": _entry(
        "games", "solitaire", "games", "solitaire"
    ),
    "task_games__solitaire__column_card_count_value": _entry(
        "games", "solitaire", "games", "solitaire"
    ),
    "task_games__solitaire__foundation_ready_count": _entry(
        "games", "solitaire", "games", "solitaire"
    ),
    "task_games__solitaire__move_legality_label": _entry(
        "games", "solitaire", "games", "solitaire"
    ),
    "task_games__solitaire__tableau_movable_card_count_value": _entry(
        "games", "solitaire", "games", "solitaire"
    ),
    "task_games__slot_machine__paytable_score_value": _entry(
        "games", "slot_machine", "games", "slot_machine"
    ),
    "task_games__slot_machine__reel_completion_label": _entry(
        "games", "slot_machine", "games", "slot_machine"
    ),
    "task_games__slot_machine__winning_payline_count": _entry(
        "games", "slot_machine", "games", "slot_machine"
    ),
    "task_games__space_shooter__enemy_ship_count": _entry(
        "games", "space_shooter", "games", "space_shooter"
    ),
    "task_games__space_shooter__enemy_ship_hit_count": _entry(
        "games", "space_shooter", "games", "space_shooter"
    ),
    "task_games__space_shooter__first_hit_enemy_ship_label": _entry(
        "games", "space_shooter", "games", "space_shooter"
    ),
    "task_games__space_shooter__hit_enemy_ship_label": _entry(
        "games", "space_shooter", "games", "space_shooter"
    ),
    "task_games__space_shooter__safe_lane_count": _entry(
        "games", "space_shooter", "games", "space_shooter"
    ),
    "task_games__tetris__active_piece_shape_label": _entry(
        "games", "tetris", "games", "tetris"
    ),
    "task_games__tetris__drop_collision_time_value": _entry(
        "games", "tetris", "games", "tetris"
    ),
    "task_games__tetris__drop_result_label": _entry(
        "games", "tetris", "games", "tetris"
    ),
    "task_games__tetris__line_clear_count": _entry(
        "games", "tetris", "games", "tetris"
    ),
    "task_games__tetris__row_occupancy_status_count": _entry(
        "games", "tetris", "games", "tetris"
    ),
    "task_games__tic_tac_toe_3d__layer_piece_count": _entry(
        "games", "tic_tac_toe_3d", "games", "tic_tac_toe_3d"
    ),
    "task_games__tic_tac_toe_3d__blocking_move_cell_label": _entry(
        "games", "tic_tac_toe_3d", "games", "tic_tac_toe_3d"
    ),
    "task_games__tic_tac_toe_3d__winning_move_cell_label": _entry(
        "games", "tic_tac_toe_3d", "games", "tic_tac_toe_3d"
    ),
    "task_games__tower_defense__best_tower_position_label": _entry(
        "games", "tower_defense", "games", "tower_defense"
    ),
    "task_games__tower_defense__covered_path_segment_count": _entry(
        "games", "tower_defense", "games", "tower_defense"
    ),
    "task_games__tower_defense__nearest_exit_enemy_label": _entry(
        "games", "tower_defense", "games", "tower_defense"
    ),
    "task_games__tower_draughts_board__controlled_stack_count": _entry(
        "games", "tower_draughts_board", "games", "tower_draughts_board"
    ),
    "task_games__tower_draughts_board__marked_stack_capture_count": _entry(
        "games", "tower_draughts_board", "games", "tower_draughts_board"
    ),
    "task_games__ultimate_tictactoe__line_completion_move_label": _entry(
        "games", "ultimate_tictactoe", "games", "ultimate_tictactoe"
    ),
    "task_games__ultimate_tictactoe__macro_threat_board_count": _entry(
        "games", "ultimate_tictactoe", "games", "ultimate_tictactoe"
    ),
    "task_games__ultimate_tictactoe__small_board_status_count": _entry(
        "games", "ultimate_tictactoe", "games", "ultimate_tictactoe"
    ),
    # Geometry.
    "task_geometry__angle_relations__algebraic_angle_value": _entry(
        "geometry", "angle_relations", "geometry", "measurement"
    ),
    "task_geometry__angle_relations__parallel_algebraic_angle_value": _entry(
        "geometry", "angle_relations", "geometry", "measurement"
    ),
    "task_geometry__angle_relations__parallel_supplement_angle": _entry(
        "geometry", "angle_relations", "geometry", "measurement"
    ),
    "task_geometry__angle_relations__parallel_transversal_triangle_angle_value": _entry(
        "geometry", "angle_relations", "geometry", "measurement"
    ),
    "task_geometry__angle_relations__triangle_exterior_angle": _entry(
        "geometry", "angle_relations", "geometry", "measurement"
    ),
    "task_geometry__area_partition__total_area_value": _entry(
        "geometry", "area_partition", "geometry", "measurement"
    ),
    "task_geometry__bearing_route__endpoint_position_label": _entry(
        "geometry", "bearing_route", "geometry", "measurement"
    ),
    "task_geometry__bearing_route__final_bearing_value": _entry(
        "geometry", "bearing_route", "geometry", "measurement"
    ),
    "task_geometry__survey_traverse__outgoing_bearing_from_turn_value": _entry(
        "geometry", "survey_traverse", "geometry", "survey_traverse"
    ),
    "task_geometry__survey_traverse__station_elevation_value": _entry(
        "geometry", "survey_traverse", "geometry", "survey_traverse"
    ),
    "task_geometry__survey_traverse__traverse_area_value": _entry(
        "geometry", "survey_traverse", "geometry", "survey_traverse"
    ),
    "task_geometry__circle_theorem__chord_length_from_radius_central_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__chord_length_from_radius_inscribed_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__diameter_perpendicular_chord_length_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__cyclic_quadrilateral_exterior_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__cyclic_quadrilateral_opposite_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__external_secant_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__inscribed_central_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__inscribed_angle_value_inscribed_angle_from_arc": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__intersecting_chords_arc_measure_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__multi_step_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__secant_secant_length_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__tangent_chord_angle_value_tangent_chord_angle_from_arc": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__tangent_chord_angle_value_tangent_chord_angle_from_inscribed": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__radius_from_external_distance_and_angle_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__tangent_length_from_radius_and_external_distance_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_theorem__tangent_secant_length_value": _entry(
        "geometry", "circle_theorem", "geometry", "circle"
    ),
    "task_geometry__circle_centerline_overlap__segment_length_value": _entry(
        "geometry", "circle_centerline_overlap", "geometry", "measurement"
    ),
    "task_geometry__circle_pair_tangents__external_tangent_segment_length_value": _entry(
        "geometry", "circle_pair_tangents", "geometry", "measurement"
    ),
    "task_geometry__circle_polygon_composite__tangent_angle_value": _entry(
        "geometry", "circle_polygon_composite", "geometry", "measurement"
    ),
    "task_geometry__circle_polygon_composite__tangential_quadrilateral_side_length_value": _entry(
        "geometry", "circle_polygon_composite", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__house_outline_perimeter": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__l_profile_area": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__missing_width_from_semicircle_area": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__rectangle_quarter_sector_cutout_area": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__rectangle_quarter_sector_cutout_perimeter": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__rectangle_semicircle_area": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__rectangle_semicircle_perimeter": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__rectangle_triangle_cutout_area": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__sector_angle_value": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__composite_shape__tabbed_rectilinear_perimeter": _entry(
        "geometry", "composite_shape", "geometry", "measurement"
    ),
    "task_geometry__concentric_chord__chord_length_from_radii": _entry(
        "geometry", "concentric_chord", "geometry", "measurement"
    ),
    "task_geometry__concentric_chord__inner_radius_from_chord": _entry(
        "geometry", "concentric_chord", "geometry", "measurement"
    ),
    "task_geometry__cone_net__base_radius_from_sector_angle": _entry(
        "geometry", "cone_net", "geometry", "measurement"
    ),
    "task_geometry__cone_net__height_from_sector_angle": _entry(
        "geometry", "cone_net", "geometry", "measurement"
    ),
    "task_geometry__container_volume_transfer__fill_count_value": _entry(
        "geometry", "container_volume_transfer", "geometry", "measurement"
    ),
    "task_geometry__container_volume_transfer__resulting_height_value": _entry(
        "geometry", "container_volume_transfer", "geometry", "measurement"
    ),
    "task_geometry__coordinate_composite__boundary_point_match_label": _entry(
        "geometry", "coordinate_composite", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_composite__intersection_point_count": _entry(
        "geometry", "coordinate_composite", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_composite__region_membership_label": _entry(
        "geometry", "coordinate_composite", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_panels__point_set_transform_match_label": _entry(
        "geometry", "coordinate_panels", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_panels__quadrilateral_shape_match_label": _entry(
        "geometry", "coordinate_panels", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_panels__segment_relation_match_label": _entry(
        "geometry", "coordinate_panels", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__collinear_point_count": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__locus_panel_match_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__locus_point_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__missing_endpoint_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__point_in_polygon_count": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__quadrilateral_completion_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__reflected_point_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__rotated_point_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__same_quadrant_point_count": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__section_point_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__segment_relation_count": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__coordinate_plane__translated_point_label": _entry(
        "geometry", "coordinate_plane", "geometry", "coordinate"
    ),
    "task_geometry__cuboid_views__cuboid_projection_surface_area_value": _entry(
        "geometry", "cuboid_views", "geometry", "measurement"
    ),
    "task_geometry__cylinder_wrap__surface_path_length_value": _entry(
        "geometry", "cylinder_wrap", "geometry", "measurement"
    ),
    "task_geometry__cylinder_wrap__wrapped_mark_position_label": _entry(
        "geometry", "cylinder_wrap", "geometry", "measurement"
    ),
    "task_geometry__function_graph__average_rate_value": _entry(
        "geometry", "function_graph", "geometry", "graphing"
    ),
    "task_geometry__function_graph__extremum_count_local_extremum_count": _entry(
        "geometry", "function_graph", "geometry", "graphing"
    ),
    "task_geometry__function_graph__extremum_count_turning_point_count": _entry(
        "geometry", "function_graph", "geometry", "graphing"
    ),
    "task_geometry__function_panels__function_status_label": _entry(
        "geometry", "function_panels", "geometry", "analytical"
    ),
    "task_geometry__function_panels__intersection_property_label": _entry(
        "geometry", "function_panels", "geometry", "analytical"
    ),
    "task_geometry__function_panels__one_to_one_status_label": _entry(
        "geometry", "function_panels", "geometry", "analytical"
    ),
    "task_geometry__function_panels__range_match_label": _entry(
        "geometry", "function_panels", "geometry", "analytical"
    ),
    "task_geometry__function_panels__sign_interval_label": _entry(
        "geometry", "function_panels", "geometry", "analytical"
    ),
    "task_geometry__function_panels__x_axis_symmetry_label": _entry(
        "geometry", "function_panels", "geometry", "analytical"
    ),
    "task_geometry__graph_paper__angle_extremum_label": _entry(
        "geometry", "graph_paper", "geometry", "comparison"
    ),
    "task_geometry__graph_paper__angle_type_count": _entry(
        "geometry", "graph_paper", "geometry", "counting"
    ),
    "task_geometry__graph_paper__area_extremum_label": _entry(
        "geometry", "graph_paper", "geometry", "comparison"
    ),
    "task_geometry__graph_paper__circle_circumference_value": _entry(
        "geometry", "graph_paper", "geometry", "measurement"
    ),
    "task_geometry__graph_paper__ellipse_area_value": _entry(
        "geometry", "graph_paper", "geometry", "measurement"
    ),
    "task_geometry__graph_paper__length_extremum_label": _entry(
        "geometry", "graph_paper", "geometry", "comparison"
    ),
    "task_geometry__graph_paper__line_slope_value": _entry(
        "geometry", "graph_paper", "geometry", "measurement"
    ),
    "task_geometry__graph_paper__perimeter_extremum_label": _entry(
        "geometry", "graph_paper", "geometry", "comparison"
    ),
    "task_geometry__graph_paper__polygon_area_value": _entry(
        "geometry", "graph_paper", "geometry", "measurement"
    ),
    "task_geometry__graph_paper__polygon_convexity_count": _entry(
        "geometry", "graph_paper", "geometry", "counting"
    ),
    "task_geometry__graph_paper__polygon_perimeter_value": _entry(
        "geometry", "graph_paper", "geometry", "measurement"
    ),
    "task_geometry__graph_paper__quadrilateral_type_count": _entry(
        "geometry", "graph_paper", "geometry", "counting"
    ),
    "task_geometry__graph_paper__right_angle_vertex_count": _entry(
        "geometry", "graph_paper", "geometry", "counting"
    ),
    "task_geometry__graph_paper__shape_type_count": _entry(
        "geometry", "graph_paper", "geometry", "counting"
    ),
    "task_geometry__graph_paper__triangle_type_count": _entry(
        "geometry", "graph_paper", "geometry", "counting"
    ),
    "task_geometry__incircle_tangents__incircle_radius_from_area_value": _entry(
        "geometry", "incircle_tangents", "geometry", "measurement"
    ),
    "task_geometry__incircle_tangents__incircle_tangent_perimeter_value": _entry(
        "geometry", "incircle_tangents", "geometry", "measurement"
    ),
    "task_geometry__measuring_tools__protractor_angle_value": _entry(
        "geometry", "measuring_tools", "geometry", "measurement"
    ),
    "task_geometry__measuring_tools__ruler_length_value": _entry(
        "geometry", "measuring_tools", "geometry", "measurement"
    ),
    "task_geometry__paper_fold__folded_segment_length_value": _entry(
        "geometry", "paper_fold", "geometry", "measurement"
    ),
    "task_geometry__paper_fold__paper_fold_angle_value": _entry(
        "geometry", "paper_fold", "geometry", "measurement"
    ),
    "task_geometry__polar_graph_paper__coordinate_difference_value": _entry(
        "geometry", "polar_graph_paper", "geometry", "coordinate"
    ),
    "task_geometry__polar_graph_paper__coordinate_value_point_count": _entry(
        "geometry", "polar_graph_paper", "geometry", "counting"
    ),
    "task_geometry__polar_graph_paper__readout_value": _entry(
        "geometry", "polar_graph_paper", "geometry", "coordinate"
    ),
    "task_geometry__polygon_equation_diagram__equal_angle_measure_value": _entry(
        "geometry", "polygon_equation_diagram", "geometry", "measurement"
    ),
    "task_geometry__polygon_equation_diagram__equal_angle_variable_value": _entry(
        "geometry", "polygon_equation_diagram", "geometry", "measurement"
    ),
    "task_geometry__polygon_equation_diagram__equal_side_length_value": _entry(
        "geometry", "polygon_equation_diagram", "geometry", "measurement"
    ),
    "task_geometry__polygon_equation_diagram__equal_side_variable_value": _entry(
        "geometry", "polygon_equation_diagram", "geometry", "measurement"
    ),
    "task_geometry__polygon_equation_diagram__interior_angle_sum_angle_value": _entry(
        "geometry", "polygon_equation_diagram", "geometry", "measurement"
    ),
    "task_geometry__polygon_equation_diagram__interior_angle_sum_variable_value": _entry(
        "geometry", "polygon_equation_diagram", "geometry", "measurement"
    ),
    "task_geometry__polygon_equation_diagram__side_expression_perimeter_value": _entry(
        "geometry", "polygon_equation_diagram", "geometry", "measurement"
    ),
    "task_geometry__pythagorean_dissection__pythagorean_square_area_value": _entry(
        "geometry", "pythagorean_dissection", "geometry", "measurement"
    ),
    "task_geometry__pythagorean_tree__missing_square_area_value": _entry(
        "geometry", "pythagorean_tree", "geometry", "measurement"
    ),
    "task_geometry__rectangular_solid__cube_edge_from_frame_length_value": _entry(
        "geometry", "rectangular_solid", "geometry", "measurement"
    ),
    "task_geometry__rectangular_solid__cuboid_surface_area_value": _entry(
        "geometry", "rectangular_solid", "geometry", "measurement"
    ),
    "task_geometry__rectangular_solid__cuboid_volume_missing_dimension_value": _entry(
        "geometry", "rectangular_solid", "geometry", "measurement"
    ),
    "task_geometry__rectangular_solid__open_box_net_dimension_value": _entry(
        "geometry", "rectangular_solid", "geometry", "measurement"
    ),
    "task_geometry__regular_polygon_decomposition__central_angle_value": _entry(
        "geometry", "regular_polygon_decomposition", "geometry", "measurement"
    ),
    "task_geometry__regular_polygon_decomposition__marked_piece_area_value": _entry(
        "geometry", "regular_polygon_decomposition", "geometry", "measurement"
    ),
    "task_geometry__regular_polygon_decomposition__perimeter_value": _entry(
        "geometry", "regular_polygon_decomposition", "geometry", "measurement"
    ),
    "task_geometry__regular_polygon_decomposition__piece_area_value": _entry(
        "geometry", "regular_polygon_decomposition", "geometry", "measurement"
    ),
    "task_geometry__regular_polygon_decomposition__side_length_value": _entry(
        "geometry", "regular_polygon_decomposition", "geometry", "measurement"
    ),
    "task_geometry__regular_polygon_decomposition__wedge_area_from_side_apothem_value": _entry(
        "geometry", "regular_polygon_decomposition", "geometry", "measurement"
    ),
    "task_geometry__sector__arc_length_from_sector_area_value": _entry(
        "geometry", "sector", "geometry", "measurement"
    ),
    "task_geometry__sector__arc_length_from_supplement_angle_value": _entry(
        "geometry", "sector", "geometry", "measurement"
    ),
    "task_geometry__sector__central_angle_from_sector_measure_value": _entry(
        "geometry", "sector", "geometry", "measurement"
    ),
    "task_geometry__sector__sector_area_from_complement_angle_value": _entry(
        "geometry", "sector", "geometry", "measurement"
    ),
    "task_geometry__sector__related_angle_from_sector_measure_value": _entry(
        "geometry", "sector", "geometry", "measurement"
    ),
    "task_geometry__shape_reference__congruent_match": _entry(
        "geometry", "shape_reference", "geometry", "similarity"
    ),
    "task_geometry__shape_reference__reflection_match": _entry(
        "geometry", "shape_reference", "geometry", "transformation"
    ),
    "task_geometry__shape_reference__rotation_match": _entry(
        "geometry", "shape_reference", "geometry", "transformation"
    ),
    "task_geometry__shape_reference__similar_match": _entry(
        "geometry", "shape_reference", "geometry", "similarity"
    ),
    "task_geometry__shape_reference__translation_match": _entry(
        "geometry", "shape_reference", "geometry", "transformation"
    ),
    "task_geometry__solid_cross_section__cone_parallel_slice_area": _entry(
        "geometry", "solid_cross_section", "geometry", "measurement"
    ),
    "task_geometry__solid_cross_section__square_pyramid_parallel_slice_area": _entry(
        "geometry", "solid_cross_section", "geometry", "measurement"
    ),
    "task_geometry__solid_formula__cylinder_cone_height_from_volume_radius": _entry(
        "geometry", "solid_formula", "geometry", "solid_formula"
    ),
    "task_geometry__solid_formula__cylinder_cone_radius_from_volume_heights": _entry(
        "geometry", "solid_formula", "geometry", "solid_formula"
    ),
    "task_geometry__solid_formula__house_prism_length_from_volume": _entry(
        "geometry", "solid_formula", "geometry", "solid_formula"
    ),
    "task_geometry__solid_formula__prism_pyramid_height_from_volume": _entry(
        "geometry", "solid_formula", "geometry", "solid_formula"
    ),
    "task_geometry__similar_figure_measure_transfer__area_scale_side_length_value": _entry(
        "geometry", "similar_figure_measure_transfer", "geometry", "measurement"
    ),
    "task_geometry__similar_figure_measure_transfer__corresponding_side_value": _entry(
        "geometry", "similar_figure_measure_transfer", "geometry", "measurement"
    ),
    "task_geometry__similar_figure_measure_transfer__variable_value": _entry(
        "geometry", "similar_figure_measure_transfer", "geometry", "measurement"
    ),
    "task_geometry__solid_revolution__revolution_cone_volume_value": _entry(
        "geometry", "solid_revolution", "geometry", "solid_revolution"
    ),
    "task_geometry__solid_revolution__revolution_cylinder_volume_value": _entry(
        "geometry", "solid_revolution", "geometry", "solid_revolution"
    ),
    "task_geometry__solid_revolution__revolution_cylinder_volume_from_diagonal_value": _entry(
        "geometry", "solid_revolution", "geometry", "solid_revolution"
    ),
    "task_geometry__solid_revolution__revolution_double_cone_volume_value": _entry(
        "geometry", "solid_revolution", "geometry", "solid_revolution"
    ),
    "task_geometry__solid_revolution__revolution_frustum_volume_value": _entry(
        "geometry", "solid_revolution", "geometry", "solid_revolution"
    ),
    "task_geometry__special_quadrilateral__algebraic_angle_value": _entry(
        "geometry", "special_quadrilateral", "geometry", "special_quadrilateral"
    ),
    "task_geometry__special_quadrilateral__segment_length_value": _entry(
        "geometry", "special_quadrilateral", "geometry", "special_quadrilateral"
    ),
    "task_geometry__tangent_packing__circle_in_square_gap_area": _entry(
        "geometry", "tangent_packing", "geometry", "measurement"
    ),
    "task_geometry__tangent_packing__circle_in_square_radius_from_gap_area": _entry(
        "geometry", "tangent_packing", "geometry", "measurement"
    ),
    "task_geometry__tangent_packing__square_in_circle_gap_area": _entry(
        "geometry", "tangent_packing", "geometry", "measurement"
    ),
    "task_geometry__tangent_packing__square_in_circle_side_from_gap_area": _entry(
        "geometry", "tangent_packing", "geometry", "measurement"
    ),
    "task_geometry__tangent_packing__two_circles_in_rectangle_gap_area": _entry(
        "geometry", "tangent_packing", "geometry", "measurement"
    ),
    "task_geometry__tangent_packing__two_circles_in_rectangle_radius_from_gap_area": _entry(
        "geometry", "tangent_packing", "geometry", "measurement"
    ),
    "task_geometry__trapezoid_extension__extension_from_parallelogram_area": _entry(
        "geometry", "trapezoid_extension", "geometry", "measurement"
    ),
    "task_geometry__trapezoid_extension__extension_from_parallelogram_perimeter": _entry(
        "geometry", "trapezoid_extension", "geometry", "measurement"
    ),
    "task_geometry__trapezoid_extension__trapezoid_area_from_extension_and_height": _entry(
        "geometry", "trapezoid_extension", "geometry", "measurement"
    ),
    "task_geometry__trapezoid_extension__trapezoid_area_from_parallelogram_area": _entry(
        "geometry", "trapezoid_extension", "geometry", "measurement"
    ),
    "task_geometry__trapezoid_extension__trapezoid_area_from_parallelogram_perimeter": _entry(
        "geometry", "trapezoid_extension", "geometry", "measurement"
    ),
    "task_geometry__volume_equivalence_conversion__equal_volume_option_label": _entry(
        "geometry", "volume_equivalence_conversion", "geometry", "measurement"
    ),
    "task_geometry__volume_equivalence_conversion__missing_dimension_value": _entry(
        "geometry", "volume_equivalence_conversion", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__angle_bisector_segment_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__angle_bisector_variable_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__altitude_to_hypotenuse_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__centroid_median_segment_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__parallel_section_segment_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__pythagorean_length_value_chained_rectangle_diagonal_length": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__pythagorean_length_value_rectangle_triangle_shared_height_length": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__leg_projection_length_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__right_triangle_missing_side_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__similar_triangles_side_length": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__split_triangle_angle_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    "task_geometry__triangle_relations__split_triangle_trig_side_length_value": _entry(
        "geometry", "triangle_relations", "geometry", "measurement"
    ),
    # Graph.
    "task_graph__node_link__degree_extremum_value": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__largest_component_size": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__adjacency__directed_strong_component_count": _entry(
        "graph", "adjacency", "graph", "adjacency"
    ),
    "task_graph__adjacency__directed_pair_reciprocity_count": _entry(
        "graph", "adjacency", "graph", "adjacency"
    ),
    "task_graph__adjacency__undirected_component_count": _entry(
        "graph", "adjacency", "graph", "adjacency"
    ),
    "task_graph__node_link__articulation_point_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__binary_tree__child_structure_node_count": _entry(
        "graph", "binary_tree", "graph", "binary_tree"
    ),
    "task_graph__binary_tree__depth_level_node_count": _entry(
        "graph", "binary_tree", "graph", "binary_tree"
    ),
    "task_graph__node_link__bridge_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__cross_color_edge_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__degree_after_removal_filter_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__degree_value_filter_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__edge_color_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__edge_text_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__isolated_after_removal_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__named_node_degree_value": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__node_color_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__phylogeny_tree__clade_leaf_count": _entry(
        "graph", "phylogeny_tree", "graph", "phylogeny_tree"
    ),
    "task_graph__pipe_network__bridge_count": _entry(
        "graph", "pipe_network", "graph", "pipe_network"
    ),
    "task_graph__metro__station_membership_count": _entry(
        "graph", "metro", "graph", "metro"
    ),
    "task_graph__metro__route_condition_station_count": _entry(
        "graph", "metro", "graph", "metro"
    ),
    "task_graph__flow_network__max_flow_value": _entry(
        "graph", "flow_network", "graph", "flow_network"
    ),
    "task_graph__flow_network__min_cut_edge_count": _entry(
        "graph", "flow_network", "graph", "flow_network"
    ),
    "task_graph__adjacency__mst_weight": _entry(
        "graph", "adjacency", "graph", "adjacency"
    ),
    "task_graph__node_link__mst_weight": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__adjacency__traversal_kth_label": _entry(
        "graph", "adjacency", "graph", "adjacency"
    ),
    "task_graph__binary_tree__traversal_kth_label": _entry(
        "graph", "binary_tree", "graph", "binary_tree"
    ),
    "task_graph__node_link__topological_endpoint_node_label": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__longest_path_length": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__metro__shortest_path_length": _entry(
        "graph", "metro", "graph", "metro"
    ),
    "task_graph__pipe_network__shortest_path_length": _entry(
        "graph", "pipe_network", "graph", "pipe_network"
    ),
    "task_graph__node_link__shortest_path_length": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__automaton__state_after_input_label": _entry(
        "graph", "automaton", "graph", "automaton"
    ),
    "task_graph__automaton__dfa_accepted_string_label": _entry(
        "graph", "automaton", "graph", "automaton"
    ),
    "task_graph__automaton__nfa_accepted_string_label": _entry(
        "graph", "automaton", "graph", "automaton"
    ),
    "task_graph__automaton__nondeterministic_state_count": _entry(
        "graph", "automaton", "graph", "automaton"
    ),
    "task_graph__binary_tree__local_relative_node_label": _entry(
        "graph", "binary_tree", "graph", "binary_tree"
    ),
    "task_graph__binary_tree__lowest_common_ancestor_label": _entry(
        "graph", "binary_tree", "graph", "binary_tree"
    ),
    "task_graph__phylogeny_tree__mrca_clade_membership_count": _entry(
        "graph", "phylogeny_tree", "graph", "phylogeny_tree"
    ),
    "task_graph__phylogeny_tree__sister_leaf_label": _entry(
        "graph", "phylogeny_tree", "graph", "phylogeny_tree"
    ),
    "task_graph__phylogeny_tree__topology_outlier_label": _entry(
        "graph", "phylogeny_tree", "graph", "phylogeny_tree"
    ),
    "task_graph__pedigree_chart__relatedness_coefficient_label": _entry(
        "graph", "pedigree_chart", "graph", "pedigree_chart"
    ),
    "task_graph__pedigree_chart__relationship_label": _entry(
        "graph", "pedigree_chart", "graph", "pedigree_chart"
    ),
    "task_graph__node_link__common_related_node_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__component_size_after_edge_edit": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__edge_between_nodes_label": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__reachable_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__reachable_count_after_edge_edit": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__same_component_count": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__pipe_network__pipe_exact_distance_count": _entry(
        "graph", "pipe_network", "graph", "pipe_network"
    ),
    "task_graph__pipe_network__pipe_reachable_junction_count": _entry(
        "graph", "pipe_network", "graph", "pipe_network"
    ),
    "task_graph__metro__exact_distance_station_count": _entry(
        "graph", "metro", "graph", "metro"
    ),
    "task_graph__binary_tree__bst_path_operation_label": _entry(
        "graph", "binary_tree", "graph", "binary_tree"
    ),
    "task_graph__binary_tree__heap_property_violation_label": _entry(
        "graph", "binary_tree", "graph", "binary_tree"
    ),
    "task_graph__graph_options__contained_subgraph_label": _entry(
        "graph", "graph_options", "graph", "graph_options"
    ),
    "task_graph__graph_options__same_structure_label": _entry(
        "graph", "graph_options", "graph", "graph_options"
    ),
    "task_graph__node_link__unique_related_node_label": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__hamiltonian_cycle_neighbor_label": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__largest_chordless_cycle_size": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    "task_graph__node_link__unique_cycle_size": _entry(
        "graph", "node_link", "graph", "node_link"
    ),
    # Icons.
    "task_icons__icon_field__singleton_type_count": _entry(
        "icons", "icon_field", "icons", "icon_field"
    ),
    "task_icons__icon_field__most_frequent_type_count": _entry(
        "icons", "icon_field", "icons", "icon_field"
    ),
    "task_icons__icon_field__frequency_extreme_type_label": _entry(
        "icons", "icon_field", "icons", "icon_field"
    ),
    "task_icons__icon_grid__distinct_color_count": _entry(
        "icons", "icon_grid", "icons", "icon_grid"
    ),
    "task_icons__icon_grid__distinct_type_count": _entry(
        "icons", "icon_grid", "icons", "icon_grid"
    ),
    "task_icons__icon_cutout__partial_match_label": _entry(
        "icons", "icon_cutout", "icons", "icon_cutout"
    ),
    "task_icons__reference_canvas__reference_type_match_count": _entry(
        "icons", "reference_canvas", "icons", "reference_canvas"
    ),
    "task_icons__reference_canvas__reference_color_match_count": _entry(
        "icons", "reference_canvas", "icons", "reference_canvas"
    ),
    "task_icons__reference_canvas__reference_rotation_match_count": _entry(
        "icons", "reference_canvas", "icons", "reference_canvas"
    ),
    "task_icons__reference_canvas__reference_type_color_rotation_match_count": _entry(
        "icons", "reference_canvas", "icons", "reference_canvas"
    ),
    "task_icons__reference_canvas__reference_metric_relation_count": _entry(
        "icons", "reference_canvas", "icons", "reference_canvas"
    ),
    "task_icons__reference_canvas__anchor_position_count": _entry(
        "icons", "reference_canvas", "icons", "reference_canvas"
    ),
    "task_icons__named_field__single_attribute_membership_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__multi_attribute_and_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__multi_attribute_complement_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__multi_attribute_exclusion_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__multi_attribute_or_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__multi_attribute_xor_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__closer_to_reference_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__counterfactual_attribute_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__counterfactual_total_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__scoped_attribute_count": _entry(
        "icons", "named_field", "icons", "counting"
    ),
    "task_icons__named_field__reference_distance_rank_label": _entry(
        "icons", "named_field", "icons", "relation"
    ),
    "task_icons__named_grid__group_predicate_count": _entry(
        "icons", "named_grid", "icons", "named_grid"
    ),
    "task_icons__named_grid__line_adjacency_pair_count": _entry(
        "icons", "named_grid", "icons", "named_grid"
    ),
    "task_icons__named_grid__scoped_attribute_count": _entry(
        "icons", "named_grid", "icons", "named_grid"
    ),
    "task_icons__named_grid__row_column_shape_extreme_number": _entry(
        "icons", "named_grid", "icons", "named_grid"
    ),
    "task_icons__named_ring__scoped_attribute_count": _entry(
        "icons", "named_ring", "icons", "named_ring"
    ),
    "task_icons__named_ring__nearest_marker_target_count": _entry(
        "icons", "named_ring", "icons", "named_ring"
    ),
    "task_icons__named_path__path_neighbor_label": _entry(
        "icons", "named_path", "icons", "named_path"
    ),
    "task_icons__named_path__path_distance_value": _entry(
        "icons", "named_path", "icons", "named_path"
    ),
    "task_icons__venn_field__scoped_attribute_count": _entry(
        "icons", "venn_field", "icons", "venn_field"
    ),
    "task_icons__venn_field__same_region_as_reference_count": _entry(
        "icons", "venn_field", "icons", "venn_field"
    ),
    "task_icons__paired_canvas__panel_set_relation_count": _entry(
        "icons", "paired_canvas", "icons", "paired_canvas"
    ),
    "task_icons__paired_canvas__color_change_count": _entry(
        "icons", "paired_canvas", "icons", "paired_canvas"
    ),
    "task_icons__paired_canvas__rotation_change_count": _entry(
        "icons", "paired_canvas", "icons", "paired_canvas"
    ),
    "task_icons__pair_grid__reference_color_pair_match_label": _entry(
        "icons", "pair_grid", "icons", "pair_grid"
    ),
    "task_icons__pair_grid__reference_transform_match_label": _entry(
        "icons", "pair_grid", "icons", "pair_grid"
    ),
    "task_icons__single_transform_options__geometric_transform_result_label": _entry(
        "icons", "single_transform_options", "icons", "single_transform_options"
    ),
    "task_icons__single_transform_options__inverse_geometric_transform_source_label": _entry(
        "icons", "single_transform_options", "icons", "single_transform_options"
    ),
    "task_icons__mirror_grid__mirror_symmetry_match_label": _entry(
        "icons", "mirror_grid", "icons", "mirror_grid"
    ),
    "task_icons__mirror_grid__missing_mirror_cell_label": _entry(
        "icons", "mirror_grid", "icons", "mirror_grid"
    ),
    "task_icons__overlap_grid__occlusion_order_count": _entry(
        "icons", "overlap_grid", "icons", "overlap_grid"
    ),
    "task_icons__sequence_strip__count_progression_completion_label": _entry(
        "icons", "sequence_strip", "icons", "sequence_strip"
    ),
    "task_icons__sequence_strip__rotation_progression_completion_label": _entry(
        "icons", "sequence_strip", "icons", "sequence_strip"
    ),
    "task_icons__sequence_strip__size_progression_completion_label": _entry(
        "icons", "sequence_strip", "icons", "sequence_strip"
    ),
    "task_icons__wallpaper_panels__motif_violation_label": _entry(
        "icons", "wallpaper_panels", "icons", "wallpaper_panels"
    ),
    "task_icons__wallpaper_panels__same_pattern_as_reference_label": _entry(
        "icons", "wallpaper_panels", "icons", "wallpaper_panels"
    ),
    "task_icons__named_strip__shape_run_length": _entry(
        "icons", "named_strip", "icons", "named_strip"
    ),
    "task_icons__named_strip__shape_run_count": _entry(
        "icons", "named_strip", "icons", "named_strip"
    ),
    # Illustrations.
    "task_illustrations__environment__lit_window_count": _entry(
        "illustrations", "environment", "illustrations", "environment"
    ),
    "task_illustrations__construction_site__equipment_zone_count": _entry(
        "illustrations", "construction_site", "illustrations", "construction_site"
    ),
    "task_illustrations__construction_site__missing_patch_label": _entry(
        "illustrations", "construction_site", "illustrations", "construction_site"
    ),
    "task_illustrations__construction_site__rotated_tile_label": _entry(
        "illustrations", "construction_site", "illustrations", "construction_site"
    ),
    "task_illustrations__environment__feature_relation_object_count": _entry(
        "illustrations", "environment", "illustrations", "environment"
    ),
    "task_illustrations__environment__missing_patch_label": _entry(
        "illustrations", "environment", "illustrations", "environment"
    ),
    "task_illustrations__environment__rotated_tile_label": _entry(
        "illustrations", "environment", "illustrations", "environment"
    ),
    "task_illustrations__environment__crossing_feature_count": _entry(
        "illustrations", "environment", "illustrations", "environment"
    ),
    "task_illustrations__library__books_in_section_count": _entry(
        "illustrations", "library", "illustrations", "library"
    ),
    "task_illustrations__library__filtered_book_in_section_count": _entry(
        "illustrations", "library", "illustrations", "library"
    ),
    "task_illustrations__library__missing_patch_label": _entry(
        "illustrations", "library", "illustrations", "library"
    ),
    "task_illustrations__library__rotated_tile_label": _entry(
        "illustrations", "library", "illustrations", "library"
    ),
    "task_illustrations__library__swapped_tile_pair_label": _entry(
        "illustrations", "library", "illustrations", "library"
    ),
    "task_illustrations__indoor_room__surface_object_count": _entry(
        "illustrations", "indoor_room", "illustrations", "indoor_room"
    ),
    "task_illustrations__indoor_room__missing_patch_label": _entry(
        "illustrations", "indoor_room", "illustrations", "indoor_room"
    ),
    "task_illustrations__indoor_room__rotated_tile_label": _entry(
        "illustrations", "indoor_room", "illustrations", "indoor_room"
    ),
    "task_illustrations__indoor_room__swapped_tile_pair_label": _entry(
        "illustrations", "indoor_room", "illustrations", "indoor_room"
    ),
    "task_illustrations__park_playground__person_count": _entry(
        "illustrations", "park_playground", "illustrations", "park_playground"
    ),
    "task_illustrations__park_playground__missing_patch_label": _entry(
        "illustrations", "park_playground", "illustrations", "park_playground"
    ),
    "task_illustrations__park_playground__rotated_tile_label": _entry(
        "illustrations", "park_playground", "illustrations", "park_playground"
    ),
    "task_illustrations__park_playground__jigsaw_arrangement_label": _entry(
        "illustrations", "park_playground", "illustrations", "park_playground"
    ),
    "task_illustrations__park_playground__playground_equipment_count": _entry(
        "illustrations", "park_playground", "illustrations", "park_playground"
    ),
    "task_illustrations__park_playground__swapped_tile_pair_label": _entry(
        "illustrations", "park_playground", "illustrations", "park_playground"
    ),
    "task_illustrations__pixel_village__missing_patch_label": _entry(
        "illustrations", "pixel_village", "illustrations", "pixel_village"
    ),
    "task_illustrations__pixel_village__object_type_count": _entry(
        "illustrations", "pixel_village", "illustrations", "pixel_village"
    ),
    "task_illustrations__pixel_village__person_path_count": _entry(
        "illustrations", "pixel_village", "illustrations", "pixel_village"
    ),
    "task_illustrations__pixel_village__territory_object_count": _entry(
        "illustrations", "pixel_village", "illustrations", "pixel_village"
    ),
    "task_illustrations__pixel_village__river_side_object_count": _entry(
        "illustrations", "pixel_village", "illustrations", "pixel_village"
    ),
    "task_illustrations__pixel_village__rotated_tile_label": _entry(
        "illustrations", "pixel_village", "illustrations", "pixel_village"
    ),
    "task_illustrations__pixel_village__swapped_tile_pair_label": _entry(
        "illustrations", "pixel_village", "illustrations", "pixel_village"
    ),
    "task_illustrations__rpg_house__adjacent_room_count": _entry(
        "illustrations", "rpg_house", "illustrations", "rpg_house"
    ),
    "task_illustrations__rpg_house__missing_patch_label": _entry(
        "illustrations", "rpg_house", "illustrations", "rpg_house"
    ),
    "task_illustrations__rpg_house__reachable_room_count": _entry(
        "illustrations", "rpg_house", "illustrations", "rpg_house"
    ),
    "task_illustrations__rpg_house__room_count": _entry(
        "illustrations", "rpg_house", "illustrations", "rpg_house"
    ),
    "task_illustrations__rpg_house__swapped_tile_pair_label": _entry(
        "illustrations", "rpg_house", "illustrations", "rpg_house"
    ),
    "task_illustrations__rpg_dungeon__reachable_chest_count": _entry(
        "illustrations", "rpg_dungeon", "illustrations", "rpg_dungeon"
    ),
    "task_illustrations__rpg_dungeon__monster_chamber_count": _entry(
        "illustrations", "rpg_dungeon", "illustrations", "rpg_dungeon"
    ),
    "task_illustrations__rpg_dungeon__safe_reachable_chest_count": _entry(
        "illustrations", "rpg_dungeon", "illustrations", "rpg_dungeon"
    ),
    "task_illustrations__rpg_dungeon__missing_patch_label": _entry(
        "illustrations", "rpg_dungeon", "illustrations", "rpg_dungeon"
    ),
    "task_illustrations__rpg_tactical_map__counterfactual_terrain_conversion_cost_value": _entry(
        "illustrations", "rpg_tactical_map", "illustrations", "rpg_tactical_map"
    ),
    "task_illustrations__rpg_tactical_map__movement_cost_value": _entry(
        "illustrations", "rpg_tactical_map", "illustrations", "rpg_tactical_map"
    ),
    "task_illustrations__rpg_tactical_map__movement_reachable_tile_count": _entry(
        "illustrations", "rpg_tactical_map", "illustrations", "rpg_tactical_map"
    ),
    "task_illustrations__rpg_tactical_map__movement_reachable_tile_label": _entry(
        "illustrations", "rpg_tactical_map", "illustrations", "rpg_tactical_map"
    ),
    "task_illustrations__rpg_tactical_map__movement_sequence_endpoint_label": _entry(
        "illustrations", "rpg_tactical_map", "illustrations", "rpg_tactical_map"
    ),
    "task_illustrations__rpg_tactical_map__terrain_type_tile_count": _entry(
        "illustrations", "rpg_tactical_map", "illustrations", "rpg_tactical_map"
    ),
    "task_illustrations__rpg_tactical_map__water_barrier_unreachable_tile_label": _entry(
        "illustrations", "rpg_tactical_map", "illustrations", "rpg_tactical_map"
    ),
    "task_illustrations__isometric_farmstead__farmer_same_level_tile_label": _entry(
        "illustrations", "isometric_farmstead", "illustrations", "isometric_farmstead"
    ),
    "task_illustrations__isometric_farmstead__highest_terrain_tile_count": _entry(
        "illustrations", "isometric_farmstead", "illustrations", "isometric_farmstead"
    ),
    "task_illustrations__isometric_farmstead__terrain_elevation_extremum_label": _entry(
        "illustrations", "isometric_farmstead", "illustrations", "isometric_farmstead"
    ),
    "task_illustrations__isometric_farmstead__terrain_level_object_count": _entry(
        "illustrations", "isometric_farmstead", "illustrations", "isometric_farmstead"
    ),
    "task_illustrations__isometric_harbor__boat_side_count": _entry(
        "illustrations", "isometric_harbor", "illustrations", "isometric_harbor"
    ),
    "task_illustrations__isometric_harbor__boat_mooring_status_count": _entry(
        "illustrations", "isometric_harbor", "illustrations", "isometric_harbor"
    ),
    "task_illustrations__isometric_harbor__boat_heading_status_count": _entry(
        "illustrations", "isometric_harbor", "illustrations", "isometric_harbor"
    ),
    "task_illustrations__isometric_harbor__shoreline_nearest_boat_label": _entry(
        "illustrations", "isometric_harbor", "illustrations", "isometric_harbor"
    ),
    "task_illustrations__isometric_quarry__highest_terrain_tile_count": _entry(
        "illustrations", "isometric_quarry", "illustrations", "isometric_quarry"
    ),
    "task_illustrations__isometric_quarry__terrain_elevation_extremum_label": _entry(
        "illustrations", "isometric_quarry", "illustrations", "isometric_quarry"
    ),
    "task_illustrations__isometric_quarry__terrain_level_object_count": _entry(
        "illustrations", "isometric_quarry", "illustrations", "isometric_quarry"
    ),
    "task_illustrations__isometric_quarry__worker_same_level_tile_label": _entry(
        "illustrations", "isometric_quarry", "illustrations", "isometric_quarry"
    ),
    "task_illustrations__construction_site__worker_attribute_count": _entry(
        "illustrations", "construction_site", "illustrations", "construction_site"
    ),
    "task_illustrations__indoor_room__furniture_side_count": _entry(
        "illustrations", "indoor_room", "illustrations", "indoor_room"
    ),
    # Physics.
    "task_physics__analog_meter__meter_readout_value": _entry(
        "physics", "analog_meter", "physics", "analog_meter"
    ),
    "task_physics__bridge_circuit__bridge_missing_resistance_value": _entry(
        "physics", "bridge_circuit", "physics", "bridge_circuit"
    ),
    "task_physics__bulb_circuit__brightness_extremum_label": _entry(
        "physics", "bulb_circuit", "physics", "bulb_circuit"
    ),
    "task_physics__circuit_equivalent__total_capacitance_value": _entry(
        "physics", "circuit_equivalent", "physics", "circuit_equivalent"
    ),
    "task_physics__circuit_equivalent__total_resistance_value": _entry(
        "physics", "circuit_equivalent", "physics", "circuit_equivalent"
    ),
    "task_physics__circuit_state_change__bulb_brightness_change_label": _entry(
        "physics", "circuit_state_change", "physics", "circuit_state_change"
    ),
    "task_physics__switch_circuit__lit_bulb_count": _entry(
        "physics", "switch_circuit", "physics", "switch_circuit"
    ),
    "task_physics__electrostatic_field__field_direction_choice": _entry(
        "physics", "electrostatic_field", "physics", "electrostatic_field"
    ),
    "task_physics__electrostatic_field__potential_value": _entry(
        "physics", "electrostatic_field", "physics", "electrostatic_field"
    ),
    "task_physics__electrostatic_field__zero_field_point_label": _entry(
        "physics", "electrostatic_field", "physics", "electrostatic_field"
    ),
    "task_physics__buoyancy_density__object_density_value": _entry(
        "physics", "buoyancy_density", "physics", "buoyancy_density"
    ),
    "task_physics__fluid_flow__continuity_speed_value": _entry(
        "physics", "fluid_flow", "physics", "fluid_flow"
    ),
    "task_physics__graduated_cylinder__displacement_volume_value": _entry(
        "physics", "graduated_cylinder", "physics", "graduated_cylinder"
    ),
    "task_physics__graduated_cylinder__volume_readout_value": _entry(
        "physics", "graduated_cylinder", "physics", "graduated_cylinder"
    ),
    "task_physics__hydraulic__hydraulic_missing_value": _entry(
        "physics", "hydraulic", "physics", "hydraulic"
    ),
    "task_physics__manometer__pressure_difference_value": _entry(
        "physics", "manometer", "physics", "manometer"
    ),
    "task_physics__electromagnetic_induction__induced_current_direction_count": _entry(
        "physics", "electromagnetic_induction", "physics", "electromagnetic_induction"
    ),
    "task_physics__magnetic_force__force_direction_choice": _entry(
        "physics", "magnetic_force", "physics", "magnetic_force"
    ),
    "task_physics__wire_magnetism__wire_field_direction_choice": _entry(
        "physics", "wire_magnetism", "physics", "wire_magnetism"
    ),
    "task_physics__free_body_forces__net_force_direction_choice": _entry(
        "physics", "free_body_forces", "physics", "free_body_forces"
    ),
    "task_physics__gear_train__output_direction_label": _entry(
        "physics", "gear_train", "physics", "gear_train"
    ),
    "task_physics__gear_train__output_speed_value": _entry(
        "physics", "gear_train", "physics", "gear_train"
    ),
    "task_physics__lever__missing_weight_balance_value": _entry(
        "physics", "lever", "physics", "lever"
    ),
    "task_physics__motion_graph__average_speed_value": _entry(
        "physics", "motion_graph", "physics", "motion_graph"
    ),
    "task_physics__motion_graph__interval_displacement_value": _entry(
        "physics", "motion_graph", "physics", "motion_graph"
    ),
    "task_physics__motion_graph__speed_change_state_choice": _entry(
        "physics", "motion_graph", "physics", "motion_graph"
    ),
    "task_physics__orbital_motion__focus_location_label": _entry(
        "physics", "orbital_motion", "physics", "orbital_motion"
    ),
    "task_physics__orbital_motion__orbital_speed_extremum_label": _entry(
        "physics", "orbital_motion", "physics", "orbital_motion"
    ),
    "task_physics__pulley__pulley_mechanical_advantage": _entry(
        "physics", "pulley", "physics", "pulley"
    ),
    "task_physics__stack_stability__stability_status_label": _entry(
        "physics", "stack_stability", "physics", "stack_stability"
    ),
    "task_physics__lever__side_torque_value": _entry(
        "physics", "lever", "physics", "lever"
    ),
    "task_physics__collision__sticky_collision_direction_choice": _entry(
        "physics", "collision", "physics", "collision"
    ),
    "task_physics__collision__sticky_collision_speed_value": _entry(
        "physics", "collision", "physics", "collision"
    ),
    "task_physics__spring__spring_extension_difference": _entry(
        "physics", "spring", "physics", "spring"
    ),
    "task_physics__spring__spring_missing_value": _entry(
        "physics", "spring", "physics", "spring"
    ),
    "task_physics__ray_optics__ray_bounce_count": _entry(
        "physics", "ray_optics", "physics", "ray_optics"
    ),
    "task_physics__ray_optics__ray_target_hit_count": _entry(
        "physics", "ray_optics", "physics", "ray_optics"
    ),
    "task_physics__refraction_layers__medium_speed_order_label": _entry(
        "physics", "refraction_layers", "physics", "refraction_layers"
    ),
    "task_physics__shadow_cause__light_source_label": _entry(
        "physics", "shadow_cause", "physics", "shadow_cause"
    ),
    "task_physics__lens_optics__image_property_choice": _entry(
        "physics", "lens_optics", "physics", "lens_optics"
    ),
    "task_physics__piston_cylinder__boundary_work_value": _entry(
        "physics", "piston_cylinder", "physics", "piston_cylinder"
    ),
    "task_physics__thermal_mixing__final_temperature_value": _entry(
        "physics", "thermal_mixing", "physics", "thermal_mixing"
    ),
    "task_physics__thermometer__temperature_conversion_value": _entry(
        "physics", "thermometer", "physics", "thermometer"
    ),
    "task_physics__vernier_caliper__length_readout_value": _entry(
        "physics", "vernier_caliper", "physics", "vernier_caliper"
    ),
    "task_physics__pv_diagram__pv_process_sign_choice": _entry(
        "physics", "pv_diagram", "physics", "pv_diagram"
    ),
    "task_physics__pv_diagram__pv_work_value": _entry(
        "physics", "pv_diagram", "physics", "pv_diagram"
    ),
    "task_physics__wave_interference__interference_point_choice": _entry(
        "physics", "wave_interference", "physics", "wave_interference"
    ),
    "task_physics__wave_interference__path_difference_value": _entry(
        "physics", "wave_interference", "physics", "wave_interference"
    ),
    "task_physics__waveform_panel__wave_property_extremum_label": _entry(
        "physics", "waveform_panel", "physics", "waveform_panel"
    ),
    "task_physics__signal_transform__periodic_harmonic_spectrum_match_label": _entry(
        "physics", "signal_transform", "physics", "signal_transform"
    ),
    # Synthetic 3D scenes.
    "task_three_d__object_scene__between_references_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__camera_distance_extremum_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__height_extremum_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__line_side_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__marked_point_depth_extremum_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__marked_point_vertical_relation_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__multiview_object_match_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__point_camera_distance_order_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__point_on_object_line_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__reference_triangle_inside_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_cluster__multi_attribute_and_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__color_membership_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__counterfactual_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__color_count_arithmetic": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__object_type_count_arithmetic": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__multi_attribute_exclusion_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__multi_attribute_or_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__multi_attribute_xor_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__object_type_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__type_frequency_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__object_cluster__total_object_count": _entry(
        "three_d", "object_cluster", "three_d", "object_cluster"
    ),
    "task_three_d__carousel__between_object_type_anchors_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__belt_object_type_count_arithmetic_value": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__belt_total_object_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__color_ordered_adjacent_pair_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__color_transfer_total_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__object_type_ordered_adjacent_pair_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__object_type_transfer_total_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__scoped_belt_color_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__scoped_belt_object_type_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__carousel__scoped_color_type_count": _entry(
        "three_d", "carousel", "three_d", "carousel"
    ),
    "task_three_d__conveyor__between_object_type_anchors_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__belt_total_object_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__color_ordered_adjacent_pair_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__color_transfer_total_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__lane_object_type_count_arithmetic_value": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__object_type_ordered_adjacent_pair_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__object_type_transfer_total_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__scoped_belt_color_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__scoped_belt_object_type_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__conveyor__scoped_color_type_count": _entry(
        "three_d", "conveyor", "three_d", "conveyor"
    ),
    "task_three_d__surface_fixture__color_count_after_operations_value": _entry(
        "three_d", "surface_fixture", "three_d", "surface_fixture"
    ),
    "task_three_d__surface_fixture__color_frequency_option_label": _entry(
        "three_d", "surface_fixture", "three_d", "surface_fixture"
    ),
    "task_three_d__surface_fixture__colored_element_count": _entry(
        "three_d", "surface_fixture", "three_d", "surface_fixture"
    ),
    "task_three_d__surface_fixture__element_count_extremum_label": _entry(
        "three_d", "surface_fixture", "three_d", "surface_fixture"
    ),
    "task_three_d__surface_fixture__recolor_board_match_label": _entry(
        "three_d", "surface_fixture", "three_d", "surface_fixture"
    ),
    "task_three_d__surface_fixture__repeated_element_count": _entry(
        "three_d", "surface_fixture", "three_d", "surface_fixture"
    ),
    "task_three_d__surface_fixture__scoped_colored_element_count": _entry(
        "three_d", "surface_fixture", "three_d", "surface_fixture"
    ),
    "task_three_d__object_scene__occlusion_order_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__object_relation_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__reference_nearest_label": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__image_plane_lateral_relation_count": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__object_scene__camera_depth_relation_count": _entry(
        "three_d", "object_scene", "three_d", "object_scene"
    ),
    "task_three_d__room__wall_object_camera_distance_label": _entry(
        "three_d", "room", "three_d", "room"
    ),
    "task_three_d__room__wall_object_side_relation_label": _entry(
        "three_d", "room", "three_d", "room"
    ),
    "task_three_d__room__wall_object_same_wall_reference_label": _entry(
        "three_d", "room", "three_d", "room"
    ),
    "task_three_d__street__intersection_nearest_label": _entry(
        "three_d", "street", "three_d", "street"
    ),
    "task_three_d__street__lane_ahead_object_label": _entry(
        "three_d", "street", "three_d", "street"
    ),
    "task_three_d__street__same_road_arm_reference_label": _entry(
        "three_d", "street", "three_d", "street"
    ),
    "task_three_d__warehouse__robot_forward_path_label": _entry(
        "three_d", "warehouse", "three_d", "warehouse"
    ),
    "task_three_d__warehouse__nearest_candidate_to_reference_label": _entry(
        "three_d", "warehouse", "three_d", "warehouse"
    ),
    # Symbolic and puzzle scenes split from the former combined puzzle/notation surface.
    "task_symbolic__abacus__displayed_value_readout": _entry(
        "symbolic", "abacus", "symbolic", "abacus"
    ),
    "task_symbolic__abacus__place_digit_readout": _entry(
        "symbolic", "abacus", "symbolic", "abacus"
    ),
    "task_symbolic__abacus__target_value_match_label": _entry(
        "symbolic", "abacus", "symbolic", "abacus"
    ),
    "task_symbolic__agent_automaton__agent_final_pose_label": _entry(
        "symbolic", "agent_automaton", "symbolic", "agent_automaton"
    ),
    "task_symbolic__agent_automaton__future_grid_label": _entry(
        "symbolic", "agent_automaton", "symbolic", "agent_automaton"
    ),
    "task_symbolic__clock__offset_readout": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__clock__elapsed_time_value": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__clock__hand_angle_value": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__clock__full_time_readout": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__clock__alarm_wait_time_value": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__braille_cell__braille_word_read_label": _entry(
        "symbolic", "braille_cell", "symbolic", "braille_cell"
    ),
    "task_symbolic__braille_cell__matching_pattern_label": _entry(
        "symbolic", "braille_cell", "symbolic", "braille_cell"
    ),
    "task_symbolic__braille_cell__word_braille_match_label": _entry(
        "symbolic", "braille_cell", "symbolic", "braille_cell"
    ),
    "task_symbolic__chemical_equation__balanced_option_label": _entry(
        "symbolic", "chemical_equation", "symbolic", "chemical_equation"
    ),
    "task_symbolic__chemical_equation__missing_coefficient_value": _entry(
        "symbolic", "chemical_equation", "symbolic", "chemical_equation"
    ),
    "task_symbolic__logic_gate_circuit__gate_type_count": _entry(
        "symbolic", "logic_gate_circuit", "symbolic", "logic_gate_circuit"
    ),
    "task_symbolic__logic_gate_circuit__internal_output_count": _entry(
        "symbolic", "logic_gate_circuit", "symbolic", "logic_gate_circuit"
    ),
    "task_symbolic__logic_gate_circuit__output_value_label": _entry(
        "symbolic", "logic_gate_circuit", "symbolic", "logic_gate_circuit"
    ),
    "task_symbolic__logic_gate_circuit__satisfying_assignment_label": _entry(
        "symbolic", "logic_gate_circuit", "symbolic", "logic_gate_circuit"
    ),
    "task_symbolic__morse_code__morse_word_read_label": _entry(
        "symbolic", "morse_code", "symbolic", "morse_code"
    ),
    "task_symbolic__morse_code__word_morse_match_label": _entry(
        "symbolic", "morse_code", "symbolic", "morse_code"
    ),
    "task_symbolic__radial_code_wheel__code_output_label": _entry(
        "symbolic", "radial_code_wheel", "symbolic", "radial_code_wheel"
    ),
    "task_symbolic__radial_code_wheel__missing_code_symbol_label": _entry(
        "symbolic", "radial_code_wheel", "symbolic", "radial_code_wheel"
    ),
    "task_symbolic__radial_code_wheel__output_code_match_label": _entry(
        "symbolic", "radial_code_wheel", "symbolic", "radial_code_wheel"
    ),
    "task_symbolic__truth_table__satisfying_row_count": _entry(
        "symbolic", "truth_table", "symbolic", "truth_table"
    ),
    "task_symbolic__truth_table__truth_pattern_label": _entry(
        "symbolic", "truth_table", "symbolic", "truth_table"
    ),
    "task_symbolic__truth_table__expression_from_rows_label": _entry(
        "symbolic", "truth_table", "symbolic", "truth_table"
    ),
    "task_puzzles__arithmetic_panel__equal_sum_line_constraint_value": _entry(
        "puzzles", "arithmetic_panel", "puzzles", "arithmetic_panel"
    ),
    "task_puzzles__arithmetic_panel__number_wall_value": _entry(
        "puzzles", "arithmetic_panel", "puzzles", "arithmetic_panel"
    ),
    "task_puzzles__arithmetic_panel__operation_table_cell_value": _entry(
        "puzzles", "arithmetic_panel", "puzzles", "arithmetic_panel"
    ),
    "task_puzzles__arithmetic_panel__row_column_total_missing_value": _entry(
        "puzzles", "arithmetic_panel", "puzzles", "arithmetic_panel"
    ),
    "task_puzzles__arithmetic_panel__vertical_arithmetic_hidden_digit_value": _entry(
        "puzzles", "arithmetic_panel", "puzzles", "arithmetic_panel"
    ),
    "task_puzzles__balance_scale__equivalent_object_count_value": _entry(
        "puzzles", "balance_scale", "puzzles", "balance_scale"
    ),
    "task_puzzles__balance_scale__missing_object_weight_value": _entry(
        "puzzles", "balance_scale", "puzzles", "balance_scale"
    ),
    "task_puzzles__balance_scale__query_side_relation_label": _entry(
        "puzzles", "balance_scale", "puzzles", "balance_scale"
    ),
    "task_puzzles__balance_scale__weight_order_label": _entry(
        "puzzles", "balance_scale", "puzzles", "balance_scale"
    ),
    "task_puzzles__cell_board__largest_component_size": _entry(
        "puzzles", "cell_board", "puzzles", "cell_board"
    ),
    "task_puzzles__cell_board__reachable_region_size": _entry(
        "puzzles", "cell_board", "puzzles", "cell_board"
    ),
    "task_puzzles__cell_board__shortest_path_length_value": _entry(
        "puzzles", "cell_board", "puzzles", "cell_board"
    ),
    "task_puzzles__cell_board__symmetry_violation_count": _entry(
        "puzzles", "cell_board", "puzzles", "cell_board"
    ),
    "task_symbolic__clock__equivalent_time_label": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__clock__sequence_completion_label": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__clock__time_order_label": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_symbolic__clock__time_extremum_label": _entry(
        "symbolic", "clock", "symbolic", "clock"
    ),
    "task_puzzles__color_gradient__color_gradient_completion_label": _entry(
        "puzzles", "color_gradient", "puzzles", "color_gradient"
    ),
    "task_puzzles__color_gradient__color_gradient_violation_cell_label": _entry(
        "puzzles", "color_gradient", "puzzles", "color_gradient"
    ),
    "task_puzzles__cube_net__marked_edge_neighbor_face_label": _entry(
        "puzzles", "cube_net", "puzzles", "cube_net"
    ),
    "task_puzzles__cube_net__opposite_face_label": _entry(
        "puzzles", "cube_net", "puzzles", "cube_net"
    ),
    "task_puzzles__cube_net__equivalent_net_label": _entry(
        "puzzles", "cube_net", "puzzles", "cube_net"
    ),
    "task_puzzles__cyclic_order__cyclic_order_equivalent_label": _entry(
        "puzzles", "cyclic_order", "puzzles", "cyclic_order"
    ),
    "task_puzzles__cyclic_order__insertion_position_label": _entry(
        "puzzles", "cyclic_order", "puzzles", "cyclic_order"
    ),
    "task_puzzles__cyclic_order__swap_repair_label": _entry(
        "puzzles", "cyclic_order", "puzzles", "cyclic_order"
    ),
    "task_symbolic__dice__dice_conditional_event_value": _entry(
        "symbolic", "dice", "symbolic", "dice"
    ),
    "task_symbolic__dice__pair_attribute_combo_probability": _entry(
        "symbolic", "dice", "symbolic", "dice"
    ),
    "task_symbolic__dice__pair_difference_probability": _entry(
        "symbolic", "dice", "symbolic", "dice"
    ),
    "task_symbolic__dice__pair_sum_probability": _entry(
        "symbolic", "dice", "symbolic", "dice"
    ),
    "task_symbolic__dice__pair_sum_threshold_probability": _entry(
        "symbolic", "dice", "symbolic", "dice"
    ),
    "task_symbolic__dice__single_attribute_probability": _entry(
        "symbolic", "dice", "symbolic", "dice"
    ),
    "task_symbolic__dice__single_threshold_probability": _entry(
        "symbolic", "dice", "symbolic", "dice"
    ),
    "task_symbolic__life_automaton__life_future_grid_label": _entry(
        "symbolic", "life_automaton", "symbolic", "life_automaton"
    ),
    "task_symbolic__life_automaton__one_step_cell_state_count": _entry(
        "symbolic", "life_automaton", "symbolic", "life_automaton"
    ),
    "task_puzzles__matchstick__equation_repair_stick_label": _entry(
        "puzzles", "matchstick", "puzzles", "matchstick"
    ),
    "task_puzzles__matchstick__matchstick_number_transform_label": _entry(
        "puzzles", "matchstick", "puzzles", "matchstick"
    ),
    "task_puzzles__matchstick__max_square_count_after_additions_value": _entry(
        "puzzles", "matchstick", "puzzles", "matchstick"
    ),
    "task_puzzles__maze__exit_reachability_label": _entry(
        "puzzles", "maze", "puzzles", "maze"
    ),
    "task_puzzles__maze__nearest_exit_label": _entry(
        "puzzles", "maze", "puzzles", "maze"
    ),
    "task_symbolic__music_staff__articulation_symbol_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__chord_inversion_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__chord_quality_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__duration_equivalence_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__interval_name_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__key_signature_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__meter_type_count": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__note_name_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__roman_numeral_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__scale_degree_function_label": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__scale_validation_count": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__music_staff__transposed_pitch_pair_count": _entry(
        "symbolic", "music_staff", "symbolic", ""
    ),
    "task_symbolic__organic_structure__bond_order_count": _entry(
        "symbolic", "organic_structure", "symbolic", ""
    ),
    "task_symbolic__organic_structure__ring_size_count": _entry(
        "symbolic", "organic_structure", "symbolic", ""
    ),
    "task_puzzles__nonogram__candidate_solution_label": _entry(
        "puzzles", "nonogram", "puzzles", "nonogram"
    ),
    "task_puzzles__nonogram__line_completion_label": _entry(
        "puzzles", "nonogram", "puzzles", "nonogram"
    ),
    "task_puzzles__pipe_flow__pipe_flow_repair_tile_label": _entry(
        "puzzles", "pipe_flow", "puzzles", "pipe_flow"
    ),
    "task_puzzles__pipe_flow__misrotated_tile_label": _entry(
        "puzzles", "pipe_flow", "puzzles", "pipe_flow"
    ),
    "task_puzzles__polyomino_assembly__composition_result_label": _entry(
        "puzzles", "polyomino_assembly", "puzzles", "polyomino_assembly"
    ),
    "task_puzzles__polyomino_assembly__decomposition_pair_label": _entry(
        "puzzles", "polyomino_assembly", "puzzles", "polyomino_assembly"
    ),
    "task_puzzles__polyomino_assembly__hole_fill_piece_label": _entry(
        "puzzles", "polyomino_assembly", "puzzles", "polyomino_assembly"
    ),
    "task_puzzles__raven_matrix__raven_analogical_transform_label": _entry(
        "puzzles", "raven_matrix", "puzzles", "raven_matrix"
    ),
    "task_puzzles__raven_matrix__raven_count_progression_label": _entry(
        "puzzles", "raven_matrix", "puzzles", "raven_matrix"
    ),
    "task_puzzles__raven_matrix__raven_feature_binding_label": _entry(
        "puzzles", "raven_matrix", "puzzles", "raven_matrix"
    ),
    "task_puzzles__raven_matrix__raven_position_progression_label": _entry(
        "puzzles", "raven_matrix", "puzzles", "raven_matrix"
    ),
    "task_puzzles__raven_matrix__raven_set_operation_label": _entry(
        "puzzles", "raven_matrix", "puzzles", "raven_matrix"
    ),
    "task_puzzles__raven_matrix__raven_spatial_transform_label": _entry(
        "puzzles", "raven_matrix", "puzzles", "raven_matrix"
    ),
    "task_puzzles__rubiks_net__post_move_face_color_count_label": _entry(
        "puzzles", "rubiks_net", "puzzles", "rubiks_net"
    ),
    "task_puzzles__rubiks_net__post_move_sticker_color_label": _entry(
        "puzzles", "rubiks_net", "puzzles", "rubiks_net"
    ),
    "task_puzzles__rubiks_net__rubiks_move_result_label": _entry(
        "puzzles", "rubiks_net", "puzzles", "rubiks_net"
    ),
    "task_puzzles__sheet_transform__fold_cut_result_label": _entry(
        "puzzles", "sheet_transform", "puzzles", "sheet_transform"
    ),
    "task_puzzles__sheet_transform__fold_projection_result_label": _entry(
        "puzzles", "sheet_transform", "puzzles", "sheet_transform"
    ),
    "task_puzzles__sheet_transform__overlay_union_result_label": _entry(
        "puzzles", "sheet_transform", "puzzles", "sheet_transform"
    ),
    "task_symbolic__spinner__multi_attribute_and_probability": _entry(
        "symbolic", "spinner", "symbolic", "spinner"
    ),
    "task_symbolic__spinner__multi_attribute_or_probability": _entry(
        "symbolic", "spinner", "symbolic", "spinner"
    ),
    "task_symbolic__spinner__single_attribute_probability": _entry(
        "symbolic", "spinner", "symbolic", "spinner"
    ),
    "task_symbolic__spinner__pair_color_event_probability": _entry(
        "symbolic", "spinner", "symbolic", "spinner"
    ),
    "task_puzzles__star_battle__remaining_valid_cell_count": _entry(
        "puzzles", "star_battle", "puzzles", "star_battle"
    ),
    "task_puzzles__star_battle__valid_cell_anywhere_label": _entry(
        "puzzles", "star_battle", "puzzles", "star_battle"
    ),
    "task_puzzles__sudoku__marked_cell_candidate_count": _entry(
        "puzzles", "sudoku", "puzzles", "sudoku"
    ),
    "task_puzzles__sudoku__marked_cell_value": _entry(
        "puzzles", "sudoku", "puzzles", "sudoku"
    ),
    "task_puzzles__sudoku__mistake_cell_label": _entry(
        "puzzles", "sudoku", "puzzles", "sudoku"
    ),
    "task_puzzles__tents__missing_tent_cell_label": _entry(
        "puzzles", "tents", "puzzles", "tents"
    ),
    "task_puzzles__tents__violating_tent_label": _entry(
        "puzzles", "tents", "puzzles", "tents"
    ),
    "task_puzzles__toggle_grid__toggle_repair_switch_label": _entry(
        "puzzles", "toggle_grid", "puzzles", "toggle_grid"
    ),
    "task_puzzles__toggle_grid__toggle_result_label": _entry(
        "puzzles", "toggle_grid", "puzzles", "toggle_grid"
    ),
    "task_symbolic__turing_tape__final_head_position_value": _entry(
        "symbolic", "turing_tape", "symbolic", "turing_tape"
    ),
    "task_symbolic__turing_tape__turing_written_symbol_count": _entry(
        "symbolic", "turing_tape", "symbolic", "turing_tape"
    ),
    "task_puzzles__voxel_cube__cube_count": _entry(
        "puzzles", "voxel_cube", "puzzles", "voxel_cube"
    ),
    "task_puzzles__voxel_cube__cube_projection_match_label": _entry(
        "puzzles", "voxel_cube", "puzzles", "voxel_cube"
    ),
    "task_puzzles__voxel_cube__cube_structure_change_count": _entry(
        "puzzles", "voxel_cube", "puzzles", "voxel_cube"
    ),
    "task_puzzles__voxel_cube__cube_visible_projection_count": _entry(
        "puzzles", "voxel_cube", "puzzles", "voxel_cube"
    ),
    "task_puzzles__word_search__search_location_label": _entry(
        "puzzles", "word_search", "puzzles", "word_search"
    ),
    "task_puzzles__word_search__present_word_option_label": _entry(
        "puzzles", "word_search", "puzzles", "word_search"
    ),
}


def canonical_domain(domain: str) -> str:
    """Return a normalized public domain label."""

    return str(domain or "").strip()


def lookup_task_taxonomy(task_id: str) -> TaxonomyEntry | None:
    """Return explicit taxonomy metadata for a known task id."""

    return TASK_TAXONOMY.get(str(task_id))


def resolve_task_taxonomy(
    task_id: str,
    *,
    source_domain: str = "",
    source_scene_id: str = "",
) -> TaxonomyEntry:
    """Resolve taxonomy for known tasks, with a permissive fallback for tests/tools."""

    entry = lookup_task_taxonomy(str(task_id))
    if entry is not None:
        if uses_current_source_layout(str(task_id), domain=str(entry.domain)):
            return TaxonomyEntry(
                domain=str(entry.domain),
                scene_id=str(entry.scene_id),
                source_domain=str(source_domain or entry.source_domain or entry.domain),
                source_scene_id="",
            )
        return entry

    parsed_domain = ""
    parsed_scene = ""
    try:
        parsed = parse_public_task_id(str(task_id))
        parsed_domain = str(parsed.domain)
        parsed_scene = str(parsed.scene_id)
    except ValueError:
        pass

    fallback_domain = canonical_domain(str(source_domain or parsed_domain or "unknown"))
    fallback_scene = (
        str(source_scene_id or parsed_scene or "unknown").strip() or "unknown"
    )
    if uses_current_source_layout(str(task_id), domain=str(fallback_domain)):
        fallback_scene = str(parsed_scene or fallback_scene)
        return TaxonomyEntry(
            domain=fallback_domain,
            scene_id=fallback_scene,
            source_domain=str(source_domain or fallback_domain),
            source_scene_id="",
        )
    return TaxonomyEntry(
        domain=fallback_domain,
        scene_id=fallback_scene,
        source_domain=str(source_domain or fallback_domain),
        source_scene_id=str(source_scene_id or fallback_scene),
    )


def resolve_task_query_id(
    *,
    query_id: str | None = None,
    trace_payload: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the diagnostic query id for one generated instance.

    ``query_id`` is the canonical branch identifier and is used here as a final
    fallback for wrappers that have not populated trace metadata yet.
    """

    trace_payload = trace_payload if isinstance(trace_payload, Mapping) else {}
    query_spec = (
        trace_payload.get("query_spec")
        if isinstance(trace_payload.get("query_spec"), Mapping)
        else {}
    )
    execution_trace = (
        trace_payload.get("execution_trace")
        if isinstance(trace_payload.get("execution_trace"), Mapping)
        else {}
    )
    for source in (query_spec, execution_trace):
        query_id = source.get("query_id") if isinstance(source, Mapping) else None
        if query_id is not None and str(query_id).strip():
            return str(query_id)

    query_id_text = str(query_id or "").strip()
    if query_id_text and query_id_text != LEGACY_DEFAULT_QUERY_ID:
        return query_id_text
    return SINGLE_QUERY_ID


def _string_or_empty(value: Any) -> str:
    """Return a stripped string value, or ``""`` for empty/None values."""

    text = str(value or "").strip()
    return text


def _mapping_values(trace_payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield metadata-bearing mappings from one trace payload."""

    for key in ("scene_ir", "query_spec", "render_spec", "execution_trace"):
        value = trace_payload.get(key)
        if not isinstance(value, Mapping):
            continue
        yield value
        relations = value.get("relations")
        if isinstance(relations, Mapping):
            yield relations
        params = value.get("params")
        if isinstance(params, Mapping):
            yield params
        prompt_variant = value.get("prompt_variant")
        if isinstance(prompt_variant, Mapping):
            yield prompt_variant
        prompt_variants = value.get("prompt_variants")
        if isinstance(prompt_variants, Mapping):
            for prompt_record in prompt_variants.values():
                if not isinstance(prompt_record, Mapping):
                    continue
                metadata = prompt_record.get("metadata")
                if isinstance(metadata, Mapping):
                    yield metadata


def _first_payload_value(trace_payload: Mapping[str, Any], *keys: str) -> str:
    """Return the first non-empty string stored under any key in trace metadata."""

    for mapping in _mapping_values(trace_payload):
        for key in keys:
            candidate = _string_or_empty(mapping.get(str(key)))
            if candidate:
                return candidate
    return ""


def inject_taxonomy_metadata(
    trace_payload: Mapping[str, Any],
    *,
    task_id: str,
    taxonomy: TaxonomyEntry,
    query_id: str = "",
    registered_domain: str = "",
    registered_scene_id: str | None = "",
) -> dict[str, Any]:
    """Return a trace payload copy with explicit taxonomy metadata injected.

    Consumers should prefer the nested blocks:

    - ``public``: public dataset taxonomy and query id.
    - ``registered``: the registered wrapper task object that produced output.
    - ``source``: source implementation/config/prompt surfaces used internally.
    """

    payload = deepcopy(dict(trace_payload))
    public_task_id = str(task_id)
    public_query_id = str(query_id).strip()
    source_layout_domain = uses_current_source_layout(public_task_id, domain=str(taxonomy.domain))
    registered_domain_text = _string_or_empty(registered_domain) or taxonomy.domain
    registered_scene_id_text = (
        _string_or_empty(registered_scene_id) or taxonomy.source_scene_id
    )
    source_task_id = _first_payload_value(
        payload, "source_task_id", "implementation_task_id"
    )
    if not source_task_id:
        candidate_task_id = _first_payload_value(payload, "task_id")
        if candidate_task_id and candidate_task_id != public_task_id:
            source_task_id = candidate_task_id
    source_task_id = source_task_id or public_task_id
    source_domain = (
        _first_payload_value(payload, "source_domain", "implementation_domain")
        or registered_domain_text
    )
    source_scene_id = (
        _first_payload_value(payload, "source_scene_id", "implementation_scene_id")
        or registered_scene_id_text
    )
    prompt_domain = _first_payload_value(payload, "prompt_domain") or source_domain
    prompt_scene_id = (
        _first_payload_value(payload, "prompt_scene_id") or source_scene_id
    )

    public_metadata = {
        "domain": taxonomy.domain,
        "scene_id": taxonomy.scene_id,
        "task_id": public_task_id,
    }
    if public_query_id:
        public_metadata["query_id"] = public_query_id
    registered_metadata = {
        "task_id": public_task_id,
        "domain": registered_domain_text,
    }
    if not source_layout_domain:
        registered_metadata["scene_id"] = registered_scene_id_text
    source_metadata = {
        "implementation_task_id": source_task_id,
        "implementation_domain": source_domain,
        "config_domain": registered_domain_text,
        "prompt_domain": prompt_domain,
    }
    if not source_layout_domain:
        source_metadata.update(
            {
                "implementation_scene_id": source_scene_id,
                "config_scene_id": registered_scene_id_text,
                "prompt_scene_id": prompt_scene_id,
            }
        )
    metadata = {
        "metadata_schema_version": "v0",
        "domain": taxonomy.domain,
        "scene_id": taxonomy.scene_id,
        "task_id": public_task_id,
        "public": public_metadata,
        "registered": registered_metadata,
        "source": source_metadata,
    }
    if public_query_id:
        metadata["query_id"] = public_query_id
    payload["taxonomy"] = metadata

    for key in ("scene_ir", "query_spec", "render_spec", "execution_trace"):
        value = payload.get(key)
        if not isinstance(value, dict):
            continue
        value.setdefault("domain", taxonomy.domain)
        value.setdefault("scene_id", taxonomy.scene_id)
        value.setdefault("task_id", public_task_id)
        if public_query_id:
            value.setdefault("query_id", public_query_id)

    return payload


def missing_taxonomy_task_ids(task_ids: Iterable[str]) -> list[str]:
    """Return task ids without explicit public taxonomy metadata."""

    return sorted(
        str(task_id) for task_id in task_ids if str(task_id) not in TASK_TAXONOMY
    )

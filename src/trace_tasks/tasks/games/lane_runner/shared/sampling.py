"""Scene-axis and render-parameter resolution for lane-runner tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.layout import resolve_games_layout_jitter, resolve_games_unit_size_scale, scale_games_px
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .rendering import LaneRunnerRenderParams
from .state import (
    DEFAULTS,
    SUPPORTED_LANE_RUNNER_SCENE_VARIANTS,
    SUPPORTED_LANE_RUNNER_STYLE_VARIANTS,
    LaneRunnerIntegerAxis,
    LaneRunnerSceneAxes,
)


def resolve_lane_runner_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> LaneRunnerSceneAxes:
    """Resolve shared row, lane, scene, and style axes."""

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=[str(value) for value in SUPPORTED_LANE_RUNNER_SCENE_VARIANTS],
    )
    style_variant, style_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=[str(value) for value in SUPPORTED_LANE_RUNNER_STYLE_VARIANTS],
    )
    row_count, row_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="row_count_support",
        explicit_key="row_count",
        fallback_support=DEFAULTS.row_count_support,
        namespace=f"{namespace}.row_count",
        balanced_flag_key="balanced_row_count_sampling",
        namespace_support_permutation=True,
    )
    start_lane, start_lane_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="start_lane_support",
        explicit_key="start_lane",
        fallback_support=DEFAULTS.start_lane_support,
        namespace=f"{namespace}.start_lane",
        balanced_flag_key="balanced_start_lane_sampling",
        namespace_support_permutation=True,
    )
    return LaneRunnerSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        row_count=int(row_count),
        lane_count=int(DEFAULTS.lane_count),
        start_lane=int(start_lane),
        row_count_support=resolve_integer_support(
            params,
            gen_defaults=gen_defaults,
            key="row_count_support",
            fallback=DEFAULTS.row_count_support,
        ),
        start_lane_support=resolve_integer_support(
            params,
            gen_defaults=gen_defaults,
            key="start_lane_support",
            fallback=DEFAULTS.start_lane_support,
        ),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        row_count_probabilities=dict(row_count_probabilities),
        start_lane_probabilities=dict(start_lane_probabilities),
    )


def resolve_lane_runner_integer_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
    upper_bound: int | None = None,
) -> LaneRunnerIntegerAxis:
    """Resolve one bounded integer target axis for a public objective."""

    raw_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    if upper_bound is None:
        feasible_support = tuple(int(value) for value in raw_support)
    else:
        feasible_support = tuple(int(value) for value in raw_support if int(value) <= int(upper_bound))
    explicit_value = params.get(str(explicit_key))
    if explicit_value is not None and upper_bound is not None and int(explicit_value) > int(upper_bound):
        raise ValueError(f"{explicit_key} must be no greater than {upper_bound}")
    if not feasible_support:
        raise ValueError(f"{support_key} has no feasible values")
    axis_params = dict(params)
    axis_params[str(support_key)] = list(feasible_support)
    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=axis_params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=feasible_support,
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return LaneRunnerIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in feasible_support),
        probabilities=dict(probabilities),
    )


def _option_card_column_count(option_count: int) -> int:
    """Return balanced option columns for four- or six-card panels."""

    if int(option_count) == 4:
        return 2
    if int(option_count) == 6:
        return 3
    return max(1, min(3, int(option_count)))


def resolve_lane_runner_render_params(
    params: Mapping[str, Any],
    *,
    axes: LaneRunnerSceneAxes,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    font_family: str,
    namespace: str,
    option_count: int | None = None,
) -> LaneRunnerRenderParams:
    """Resolve render dimensions while preserving board and option-card scale."""

    unit_scale, unit_size_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.unit_size",
        fallback_min=0.5,
        fallback_max=1.0,
    )
    cell_size = scale_games_px(
        params.get("cell_size_px", group_default(render_defaults, "cell_size_px", DEFAULTS.cell_size_px)),
        unit_scale,
        min_px=28,
    )
    cell_gap = scale_games_px(
        params.get("cell_gap_px", group_default(render_defaults, "cell_gap_px", DEFAULTS.cell_gap_px)),
        unit_scale,
        min_px=3,
    )
    panel_margin = scale_games_px(
        params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px)),
        unit_scale,
        min_px=14,
    )
    start_height = scale_games_px(
        params.get("start_band_height_px", group_default(render_defaults, "start_band_height_px", DEFAULTS.start_band_height_px)),
        unit_scale,
        min_px=24,
    )
    finish_height = scale_games_px(
        params.get("finish_band_height_px", group_default(render_defaults, "finish_band_height_px", DEFAULTS.finish_band_height_px)),
        unit_scale,
        min_px=24,
    )
    board_w = (int(axes.lane_count) * int(cell_size)) + ((int(axes.lane_count) - 1) * int(cell_gap))
    board_h = (int(axes.row_count) * int(cell_size)) + ((int(axes.row_count) - 1) * int(cell_gap))

    option_area_w = 0
    option_area_h = 0
    option_card_cell = scale_games_px(
        params.get("option_card_cell_size_px", group_default(render_defaults, "option_card_cell_size_px", DEFAULTS.option_card_cell_size_px)),
        unit_scale,
        min_px=28,
    )
    option_card_gap = scale_games_px(
        params.get("option_card_gap_px", group_default(render_defaults, "option_card_gap_px", DEFAULTS.option_card_gap_px)),
        unit_scale,
        min_px=3,
    )
    option_card_margin = scale_games_px(
        params.get("option_card_margin_px", group_default(render_defaults, "option_card_margin_px", DEFAULTS.option_card_margin_px)),
        unit_scale,
        min_px=6,
    )
    option_area_gap = scale_games_px(
        params.get("option_card_area_gap_px", group_default(render_defaults, "option_card_area_gap_px", DEFAULTS.option_card_area_gap_px)),
        unit_scale,
        min_px=14,
    )
    path_label_font_size = scale_games_px(
        params.get("path_label_font_size_px", group_default(render_defaults, "path_label_font_size_px", DEFAULTS.path_label_font_size_px)),
        unit_scale,
        min_px=11,
    )
    if option_count is not None and int(option_count) > 0:
        option_cols = _option_card_column_count(int(option_count))
        option_rows = (int(option_count) + int(option_cols) - 1) // int(option_cols)
        option_card_w = (
            (int(axes.lane_count) * int(option_card_cell))
            + ((int(axes.lane_count) - 1) * int(option_card_gap))
            + (2 * int(option_card_margin))
        )
        option_card_h = (
            (int(axes.row_count) * int(option_card_cell))
            + ((int(axes.row_count) - 1) * int(option_card_gap))
            + (4 * int(option_card_margin))
            + (2 * max(10, int(round(float(path_label_font_size) * 0.72))))
        )
        option_area_w = (int(option_cols) * int(option_card_w)) + ((int(option_cols) - 1) * 10)
        option_area_h = (int(option_rows) * int(option_card_h)) + ((int(option_rows) - 1) * 10)
        content_w = int(option_area_w) + (2 * int(panel_margin))
        content_h = int(option_area_h) + (2 * int(panel_margin))
    else:
        content_w = int(board_w) + (2 * int(panel_margin))
        content_h = int(finish_height) + int(cell_gap) + int(board_h) + int(cell_gap) + int(start_height) + (2 * int(panel_margin))

    outer_margin = int(params.get("canvas_outer_margin_px", group_default(render_defaults, "canvas_outer_margin_px", DEFAULTS.canvas_outer_margin_px)))
    dynamic_canvas = bool(params.get("dynamic_canvas_size_enabled", group_default(render_defaults, "dynamic_canvas_size_enabled", True)))
    if dynamic_canvas:
        canvas_width = max(
            int(params.get("canvas_min_width", group_default(render_defaults, "canvas_min_width", DEFAULTS.canvas_min_width))),
            int(content_w + (2 * outer_margin)),
        )
        canvas_height = max(
            int(params.get("canvas_min_height", group_default(render_defaults, "canvas_min_height", DEFAULTS.canvas_min_height))),
            int(content_h + (2 * outer_margin)),
        )
    else:
        canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 520)))
        canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 780)))

    return LaneRunnerRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        row_count=int(axes.row_count),
        lane_count=int(axes.lane_count),
        cell_size_px=int(cell_size),
        cell_gap_px=int(cell_gap),
        panel_margin_px=int(panel_margin),
        start_band_height_px=int(start_height),
        finish_band_height_px=int(finish_height),
        coin_radius_px=scale_games_px(
            params.get("coin_radius_px", group_default(render_defaults, "coin_radius_px", DEFAULTS.coin_radius_px)),
            unit_scale,
            min_px=8,
        ),
        runner_radius_px=scale_games_px(
            params.get("runner_radius_px", group_default(render_defaults, "runner_radius_px", DEFAULTS.runner_radius_px)),
            unit_scale,
            min_px=10,
        ),
        hazard_radius_px=scale_games_px(
            params.get("hazard_radius_px", group_default(render_defaults, "hazard_radius_px", DEFAULTS.hazard_radius_px)),
            unit_scale,
            min_px=10,
        ),
        path_line_width_px=scale_games_px(
            params.get("path_line_width_px", group_default(render_defaults, "path_line_width_px", DEFAULTS.path_line_width_px)),
            unit_scale,
            min_px=3,
        ),
        path_label_font_size_px=int(path_label_font_size),
        option_card_cell_size_px=int(option_card_cell),
        option_card_gap_px=int(option_card_gap),
        option_card_margin_px=int(option_card_margin),
        option_card_area_gap_px=int(option_area_gap),
        grid_line_width_px=scale_games_px(
            params.get("grid_line_width_px", group_default(render_defaults, "grid_line_width_px", DEFAULTS.grid_line_width_px)),
            unit_scale,
            min_px=1,
        ),
        label_font_size_px=scale_games_px(
            params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px)),
            unit_scale,
            min_px=12,
        ),
        font_family=str(font_family),
        instance_seed=int(instance_seed),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.layout_jitter",
        ),
        unit_size_meta=dict(unit_size_meta),
    )


__all__ = [
    "resolve_lane_runner_integer_axis",
    "resolve_lane_runner_render_params",
    "resolve_lane_runner_scene_axes",
]

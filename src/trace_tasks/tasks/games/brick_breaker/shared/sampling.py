"""Shared axis and render-parameter sampling helpers for Brick-breaker scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.layout import (
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import SUPPORTED_BRICK_BREAKER_SCENE_VARIANTS, SUPPORTED_BRICK_BREAKER_STYLE_VARIANTS
from .defaults import RENDER_FALLBACKS, SCENE_ID
from .rendering import BrickBreakerRenderParams


_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


@dataclass(frozen=True)
class ResolvedBrickBreakerSceneAxes:
    """Resolved scene/style axes shared by Brick-breaker tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedBrickBreakerIntegerAxis:
    """Resolved integer axis and support metadata."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedBrickBreakerPlayfieldAxes:
    """Resolved common brick-grid and catch-lane dimensions."""

    brick_rows: ResolvedBrickBreakerIntegerAxis
    brick_cols: ResolvedBrickBreakerIntegerAxis
    lane_count: ResolvedBrickBreakerIntegerAxis


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"games.brick_breaker.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=tuple(str(value) for value in supported),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"games.brick_breaker.{namespace}",
    )
    return str(selected), dict(probabilities)


def resolve_brick_breaker_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
) -> ResolvedBrickBreakerSceneAxes:
    """Resolve scene/style axes for one Brick-breaker scene."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_BRICK_BREAKER_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_BRICK_BREAKER_STYLE_VARIANTS,
    )
    return ResolvedBrickBreakerSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_brick_breaker_integer_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> ResolvedBrickBreakerIntegerAxis:
    """Resolve one task-owned Brick-breaker integer axis."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(item) for item in fallback_support),
    )
    return ResolvedBrickBreakerIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_brick_breaker_playfield_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    brick_row_count_support: Sequence[int],
    brick_col_count_support: Sequence[int],
    catch_lane_count_support: Sequence[int],
) -> ResolvedBrickBreakerPlayfieldAxes:
    """Resolve common Brick-breaker playfield dimension axes."""

    return ResolvedBrickBreakerPlayfieldAxes(
        brick_rows=resolve_brick_breaker_integer_axis(
            int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            support_key="brick_row_count_support",
            explicit_key="brick_rows",
            fallback_support=brick_row_count_support,
            namespace="brick_rows",
            balanced_flag_key="balanced_brick_row_sampling",
        ),
        brick_cols=resolve_brick_breaker_integer_axis(
            int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            support_key="brick_col_count_support",
            explicit_key="brick_cols",
            fallback_support=brick_col_count_support,
            namespace="brick_cols",
            balanced_flag_key="balanced_brick_col_sampling",
        ),
        lane_count=resolve_brick_breaker_integer_axis(
            int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            support_key="catch_lane_count_support",
            explicit_key="lane_count",
            fallback_support=catch_lane_count_support,
            namespace="lane_count",
            balanced_flag_key="balanced_lane_count_sampling",
        ),
    )


def resolve_brick_breaker_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> BrickBreakerRenderParams:
    """Resolve Brick-breaker rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.brick_breaker.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.brick_breaker.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.brick_breaker.layout",
        ),
        unit_scale_meta,
    )
    base_canvas_width = int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", RENDER_FALLBACKS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", RENDER_FALLBACKS.canvas_height)))
    playfield_width_px = scale_games_px(
        params.get("playfield_width_px", group_default(_RENDER_DEFAULTS, "playfield_width_px", RENDER_FALLBACKS.playfield_width_px)),
        unit_scale,
        min_px=430,
    )
    playfield_height_px = scale_games_px(
        params.get("playfield_height_px", group_default(_RENDER_DEFAULTS, "playfield_height_px", RENDER_FALLBACKS.playfield_height_px)),
        unit_scale,
        min_px=320,
    )
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(_RENDER_DEFAULTS, "dynamic_canvas_size_enabled", RENDER_FALLBACKS.dynamic_canvas_size_enabled),
        )
    )
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(_RENDER_DEFAULTS, "canvas_min_width_px", RENDER_FALLBACKS.canvas_min_width_px))),
                int(
                    round(
                        float(playfield_width_px)
                        + (2.0 * float(params.get("canvas_side_padding_px", group_default(_RENDER_DEFAULTS, "canvas_side_padding_px", RENDER_FALLBACKS.canvas_side_padding_px))))
                    )
                ),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(_RENDER_DEFAULTS, "canvas_min_height_px", RENDER_FALLBACKS.canvas_min_height_px))),
                int(
                    round(
                        float(playfield_height_px)
                        + (2.0 * float(params.get("canvas_vertical_padding_px", group_default(_RENDER_DEFAULTS, "canvas_vertical_padding_px", RENDER_FALLBACKS.canvas_vertical_padding_px))))
                    )
                ),
            ),
        )

    return BrickBreakerRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", RENDER_FALLBACKS.panel_margin_px))),
        playfield_width_px=int(playfield_width_px),
        playfield_height_px=int(playfield_height_px),
        playfield_border_width_px=scale_games_px(params.get("playfield_border_width_px", group_default(_RENDER_DEFAULTS, "playfield_border_width_px", RENDER_FALLBACKS.playfield_border_width_px)), unit_scale, min_px=2),
        brick_wall_top_px=scale_games_px(params.get("brick_wall_top_px", group_default(_RENDER_DEFAULTS, "brick_wall_top_px", RENDER_FALLBACKS.brick_wall_top_px)), unit_scale, min_px=23),
        brick_wall_height_px=scale_games_px(params.get("brick_wall_height_px", group_default(_RENDER_DEFAULTS, "brick_wall_height_px", RENDER_FALLBACKS.brick_wall_height_px)), unit_scale, min_px=135),
        brick_gap_px=scale_games_px(params.get("brick_gap_px", group_default(_RENDER_DEFAULTS, "brick_gap_px", RENDER_FALLBACKS.brick_gap_px)), unit_scale, min_px=4),
        lane_pad_height_px=scale_games_px(params.get("lane_pad_height_px", group_default(_RENDER_DEFAULTS, "lane_pad_height_px", RENDER_FALLBACKS.lane_pad_height_px)), unit_scale, min_px=21),
        lane_pad_gap_px=scale_games_px(params.get("lane_pad_gap_px", group_default(_RENDER_DEFAULTS, "lane_pad_gap_px", RENDER_FALLBACKS.lane_pad_gap_px)), unit_scale, min_px=4),
        ball_radius_px=scale_games_px(params.get("ball_radius_px", group_default(_RENDER_DEFAULTS, "ball_radius_px", RENDER_FALLBACKS.ball_radius_px)), unit_scale, min_px=8),
        path_width_px=scale_games_px(params.get("path_width_px", group_default(_RENDER_DEFAULTS, "path_width_px", RENDER_FALLBACKS.path_width_px)), unit_scale, min_px=2),
        label_font_size_px=scale_games_px(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", RENDER_FALLBACKS.label_font_size_px)), unit_scale, min_px=12),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


__all__ = [
    "ResolvedBrickBreakerIntegerAxis",
    "ResolvedBrickBreakerPlayfieldAxes",
    "ResolvedBrickBreakerSceneAxes",
    "resolve_brick_breaker_integer_axis",
    "resolve_brick_breaker_playfield_axes",
    "resolve_brick_breaker_render_params",
    "resolve_brick_breaker_scene_axes",
]

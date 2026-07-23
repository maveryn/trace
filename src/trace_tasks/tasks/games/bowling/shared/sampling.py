"""Shared axis and render-parameter sampling helpers for Bowling scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.layout import resolve_games_layout_jitter
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from trace_tasks.core.seed import spawn_rng

from .state import SUPPORTED_BOWLING_SCENE_VARIANTS, SUPPORTED_BOWLING_STYLE_VARIANTS
from .defaults import RENDER_FALLBACKS, SCENE_ID
from .rendering import BowlingRenderParams


_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


@dataclass(frozen=True)
class ResolvedBowlingSceneAxes:
    """Resolved scene/style axes shared by Bowling tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedBowlingIntegerAxis:
    """Resolved integer axis and support metadata."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


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
    rng = spawn_rng(int(instance_seed), f"games.bowling.{namespace}")
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
        sampling_namespace=f"games.bowling.{namespace}",
    )
    return str(selected), dict(probabilities)


def resolve_bowling_scene_axes(instance_seed: int, *, params: Mapping[str, Any]) -> ResolvedBowlingSceneAxes:
    """Resolve scene/style axes for one Bowling scene."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_BOWLING_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_BOWLING_STYLE_VARIANTS,
    )
    return ResolvedBowlingSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_bowling_integer_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> ResolvedBowlingIntegerAxis:
    """Resolve one task-owned Bowling integer axis."""

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
    return ResolvedBowlingIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_bowling_render_params(params: Mapping[str, Any], *, instance_seed: int) -> BowlingRenderParams:
    """Resolve Bowling rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.bowling.text_font",
        params=params,
    )
    return BowlingRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", RENDER_FALLBACKS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", RENDER_FALLBACKS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", RENDER_FALLBACKS.panel_margin_px))),
        lane_width_px=int(params.get("lane_width_px", group_default(_RENDER_DEFAULTS, "lane_width_px", RENDER_FALLBACKS.lane_width_px))),
        lane_height_px=int(params.get("lane_height_px", group_default(_RENDER_DEFAULTS, "lane_height_px", RENDER_FALLBACKS.lane_height_px))),
        lane_border_width_px=int(params.get("lane_border_width_px", group_default(_RENDER_DEFAULTS, "lane_border_width_px", RENDER_FALLBACKS.lane_border_width_px))),
        pin_radius_px=int(params.get("pin_radius_px", group_default(_RENDER_DEFAULTS, "pin_radius_px", RENDER_FALLBACKS.pin_radius_px))),
        ball_radius_px=int(params.get("ball_radius_px", group_default(_RENDER_DEFAULTS, "ball_radius_px", RENDER_FALLBACKS.ball_radius_px))),
        path_width_px=int(params.get("path_width_px", group_default(_RENDER_DEFAULTS, "path_width_px", RENDER_FALLBACKS.path_width_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", RENDER_FALLBACKS.label_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.bowling.layout",
        ),
    )


__all__ = [
    "ResolvedBowlingIntegerAxis",
    "ResolvedBowlingSceneAxes",
    "resolve_bowling_integer_axis",
    "resolve_bowling_render_params",
    "resolve_bowling_scene_axes",
]

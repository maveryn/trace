"""Style and render-parameter helpers for symbolic agent automata."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import hash64
from ....shared.config_defaults import group_default
from ....shared.font_assets import font_asset_version, sample_font_family
from ...shared.common import resolve_symbolic_axis_variant
from ...shared.scene_style import (
    SYMBOLIC_SCENE_TREATMENTS,
    SymbolicSceneStyle,
    resolve_panel_chrome_mode,
    resolve_symbolic_scene_style,
)
from ...shared.unit_size_jitter import resolve_symbolic_unit_size_scale, scale_symbolic_px

from .rules import BOARD_STYLES, RULE_VARIANTS, SCENE_VARIANTS
from .state import AgentRenderParams


def resolve_axis(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_scope: str,
    supported_variants: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve a non-query scene/task axis using the standard symbolic sampler."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=[str(item) for item in supported_variants],
        task_id=str(sampling_scope),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        axis_namespace=str(axis_namespace),
    )


def resolve_scene_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_scope: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the non-semantic scene variant."""

    return resolve_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
        supported_variants=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_board_style(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_scope: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the non-semantic board-rendering style."""

    return resolve_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
        supported_variants=BOARD_STYLES,
        explicit_key="agent_board_style",
        weights_key="agent_board_style_weights",
        balance_flag_key="balanced_agent_board_style_sampling",
        axis_namespace="agent_board_style",
    )


def resolve_rule_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_scope: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the rule family as a task-owned semantic generation axis."""

    return resolve_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
        supported_variants=RULE_VARIANTS,
        explicit_key="rule_variant",
        weights_key="rule_variant_weights",
        balance_flag_key="balanced_rule_variant_sampling",
        axis_namespace="rule_variant",
    )


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> AgentRenderParams:
    """Resolve render params from scene config defaults."""

    unit_scale, unit_meta = resolve_symbolic_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.automaton.unit_size",
    )
    font_params = {**dict(render_defaults), **dict(params)}
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="puzzles.automaton.font",
        params=font_params,
    )
    return AgentRenderParams(
        canvas_width=int(group_default(render_defaults, "canvas_width", 1040)),
        canvas_height=int(group_default(render_defaults, "canvas_height", 880)),
        cell_size_px=scale_symbolic_px(group_default(render_defaults, "cell_size_px", 56), unit_scale, min_px=28),
        grid_gap_px=scale_symbolic_px(group_default(render_defaults, "grid_gap_px", 2), unit_scale, min_px=1),
        panel_padding_px=scale_symbolic_px(group_default(render_defaults, "panel_padding_px", 28), unit_scale, min_px=12),
        panel_corner_radius_px=scale_symbolic_px(group_default(render_defaults, "panel_corner_radius_px", 22), unit_scale, min_px=8),
        panel_border_width_px=scale_symbolic_px(group_default(render_defaults, "panel_border_width_px", 3), unit_scale, min_px=1),
        grid_line_width_px=scale_symbolic_px(group_default(render_defaults, "grid_line_width_px", 2), unit_scale, min_px=1),
        option_card_width_px=scale_symbolic_px(group_default(render_defaults, "option_card_width_px", 150), unit_scale, min_px=80),
        option_card_height_px=scale_symbolic_px(group_default(render_defaults, "option_card_height_px", 116), unit_scale, min_px=70),
        option_gap_px=scale_symbolic_px(group_default(render_defaults, "option_gap_px", 18), unit_scale, min_px=8),
        option_grid_cell_px=scale_symbolic_px(group_default(render_defaults, "option_grid_cell_px", 24), unit_scale, min_px=9),
        label_font_size_px=scale_symbolic_px(group_default(render_defaults, "label_font_size_px", 22), unit_scale, min_px=12),
        small_font_size_px=scale_symbolic_px(group_default(render_defaults, "small_font_size_px", 16), unit_scale, min_px=10),
        arrow_width_px=scale_symbolic_px(group_default(render_defaults, "arrow_width_px", 6), unit_scale, min_px=2),
        unit_size_jitter=dict(unit_meta),
        layout_seed=int(hash64(int(instance_seed), "puzzles.automaton.layout", 0)),
        font_family=str(font_family),
    )


def style_meta_with_font(style_meta: Mapping[str, Any], render_params: AgentRenderParams) -> Dict[str, Any]:
    """Attach sampled font metadata used by rendered scene text."""

    return {
        **dict(style_meta),
        "font_family": str(render_params.font_family),
        "font": {
            "source": "global_font_pool",
            "font_family": str(render_params.font_family),
            "font_asset_version": font_asset_version(),
            "scope": "single_automaton_panel",
        },
    }


def resolve_agent_style(*, scene_variant: str, render_params: AgentRenderParams) -> Tuple[SymbolicSceneStyle, Dict[str, Any]]:
    """Resolve one non-semantic symbolic style pack."""

    style, metadata = resolve_symbolic_scene_style(
        instance_seed=int(render_params.layout_seed),
        namespace=f"agent_automaton.{scene_variant}",
        treatments=tuple(SYMBOLIC_SCENE_TREATMENTS),
    )
    chrome_mode, chrome_metadata = resolve_panel_chrome_mode(
        instance_seed=int(render_params.layout_seed),
        namespace=f"agent_automaton.{scene_variant}",
    )
    return style, {
        **dict(metadata),
        "scene_variant": str(scene_variant),
        "panel_chrome": dict(chrome_metadata),
        "panel_chrome_mode": str(chrome_mode),
    }


def style_meta_with_board(style_meta: Mapping[str, Any], *, board_style: str, board_style_probabilities: Mapping[str, float]) -> Dict[str, Any]:
    """Attach board-style metadata for trace inspection."""

    return {
        **dict(style_meta),
        "agent_board": {
            "board_style": str(board_style),
            "board_style_probabilities": {str(key): float(value) for key, value in board_style_probabilities.items()},
            "semantic_color_policy": {
                "state_colors_preserved_from_scene_style": True,
                "same_board_style_for_source_and_options": True,
            },
        },
    }

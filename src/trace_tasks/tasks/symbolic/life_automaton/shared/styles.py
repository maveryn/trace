"""Style and render-parameter helpers for symbolic Life automata."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import hash64
from ....shared.color_distance import color_distance
from ....shared.config_defaults import group_default
from ....shared.font_assets import font_asset_version, sample_font_family
from ....shared.text_legibility import contrast_ratio
from ...shared.common import resolve_symbolic_axis_variant
from ...shared.scene_style import (
    SYMBOLIC_SCENE_TREATMENTS,
    SymbolicSceneStyle,
    resolve_panel_chrome_mode,
    resolve_symbolic_scene_style,
)
from ...shared.unit_size_jitter import resolve_symbolic_unit_size_scale, scale_symbolic_px

from .rules import BOARD_STYLES, CELL_PALETTES, SCENE_VARIANTS
from .state import LifeBoardVisual, LifeRenderParams


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
    """Resolve the non-semantic Life scene variant."""

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


def resolve_life_board_visual(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_scope: str,
) -> LifeBoardVisual:
    """Resolve Life-specific board style and alive/dead palette."""

    board_style, board_probs = resolve_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
        supported_variants=BOARD_STYLES,
        explicit_key="life_board_style",
        weights_key="life_board_style_weights",
        balance_flag_key="balanced_life_board_style_sampling",
        axis_namespace="life_board_style",
    )
    palette_id, palette_probs = resolve_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_scope=str(sampling_scope),
        supported_variants=tuple(CELL_PALETTES),
        explicit_key="life_cell_palette",
        weights_key="life_cell_palette_weights",
        balance_flag_key="balanced_life_cell_palette_sampling",
        axis_namespace="life_cell_palette",
    )
    palette = CELL_PALETTES[str(palette_id)]
    return LifeBoardVisual(
        board_style=str(board_style),
        cell_palette_id=str(palette_id),
        dead_rgb=tuple(int(value) for value in palette["dead"]),
        alive_rgb=tuple(int(value) for value in palette["alive"]),
        grid_rgb=tuple(int(value) for value in palette["grid"]),
        edge_rgb=tuple(int(value) for value in palette["edge"]),
        mark_rgb=tuple(int(value) for value in palette["mark"]),
        accent_rgb=tuple(int(value) for value in palette["accent"]),
        board_style_probabilities={str(key): float(value) for key, value in board_probs.items()},
        cell_palette_probabilities={str(key): float(value) for key, value in palette_probs.items()},
    )


def life_board_visual_metadata(life_visual: LifeBoardVisual) -> Dict[str, Any]:
    """Return Life board metadata for trace inspection."""

    alive_dead_lab = float(color_distance(life_visual.alive_rgb, life_visual.dead_rgb, distance_space="lab"))
    alive_dead_contrast = float(contrast_ratio(life_visual.alive_rgb, life_visual.dead_rgb))
    mark_contrast = min(
        float(contrast_ratio(life_visual.mark_rgb, life_visual.dead_rgb)),
        float(contrast_ratio(life_visual.mark_rgb, life_visual.alive_rgb)),
    )
    mark_lab = min(
        float(color_distance(life_visual.mark_rgb, life_visual.dead_rgb, distance_space="lab")),
        float(color_distance(life_visual.mark_rgb, life_visual.alive_rgb, distance_space="lab")),
    )
    return {
        "board_style": str(life_visual.board_style),
        "board_style_probabilities": dict(life_visual.board_style_probabilities),
        "cell_palette_id": str(life_visual.cell_palette_id),
        "cell_palette_probabilities": dict(life_visual.cell_palette_probabilities),
        "resolved_rgb": {
            "dead": list(life_visual.dead_rgb),
            "alive": list(life_visual.alive_rgb),
            "grid": list(life_visual.grid_rgb),
            "edge": list(life_visual.edge_rgb),
            "mark": list(life_visual.mark_rgb),
            "accent": list(life_visual.accent_rgb),
        },
        "semantic_color_policy": {
            "alive_cells_remain_dark": True,
            "empty_cells_remain_light": True,
            "same_style_and_palette_for_source_and_options": True,
        },
        "contrast_checks": {
            "alive_dead_contrast_ratio": round(alive_dead_contrast, 3),
            "alive_dead_lab_distance": round(alive_dead_lab, 3),
            "alive_dead_pass": bool(alive_dead_contrast >= 4.5 and alive_dead_lab >= 45.0),
            "mark_min_cell_contrast_ratio": round(mark_contrast, 3),
            "mark_min_cell_lab_distance": round(mark_lab, 3),
            "mark_pass": bool(mark_contrast >= 2.0 and mark_lab >= 30.0),
        },
    }


def style_meta_with_font(style_meta: Mapping[str, Any], render_params: LifeRenderParams) -> Dict[str, Any]:
    """Attach the sampled font metadata used by rendered scene text."""

    return {
        **dict(style_meta),
        "font_family": str(render_params.font_family),
        "font": {
            "source": "global_font_pool",
            "font_family": str(render_params.font_family),
            "font_asset_version": font_asset_version(),
            "scope": "single_life_automaton_panel",
        },
    }


def style_meta_with_life_board(style_meta: Mapping[str, Any], *, life_visual: LifeBoardVisual) -> Dict[str, Any]:
    """Attach Life board style metadata."""

    return {
        **dict(style_meta),
        "life_board": life_board_visual_metadata(life_visual),
    }


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> LifeRenderParams:
    """Resolve render params from scene config defaults."""

    unit_scale, unit_meta = resolve_symbolic_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="symbolic.life_automaton.unit_size",
    )
    font_params = {**dict(render_defaults), **dict(params)}
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="symbolic.life_automaton.font",
        params=font_params,
    )
    return LifeRenderParams(
        canvas_width=int(group_default(render_defaults, "canvas_width", 1040)),
        canvas_height=int(group_default(render_defaults, "canvas_height", 880)),
        cell_size_px=scale_symbolic_px(group_default(render_defaults, "cell_size_px", 54), unit_scale, min_px=28),
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
        layout_seed=int(hash64(int(instance_seed), "symbolic.life_automaton.layout", 0)),
        font_family=str(font_family),
    )


def resolve_life_style(*, scene_variant: str, render_params: LifeRenderParams) -> Tuple[SymbolicSceneStyle, Dict[str, Any]]:
    """Resolve one non-semantic symbolic style pack."""

    style, metadata = resolve_symbolic_scene_style(
        instance_seed=int(render_params.layout_seed),
        namespace=f"life_automaton.{scene_variant}",
        treatments=tuple(SYMBOLIC_SCENE_TREATMENTS),
    )
    chrome_mode, chrome_metadata = resolve_panel_chrome_mode(
        instance_seed=int(render_params.layout_seed),
        namespace=f"life_automaton.{scene_variant}",
    )
    return style, {
        **dict(metadata),
        "scene_variant": str(scene_variant),
        "panel_chrome": dict(chrome_metadata),
        "panel_chrome_mode": str(chrome_mode),
    }

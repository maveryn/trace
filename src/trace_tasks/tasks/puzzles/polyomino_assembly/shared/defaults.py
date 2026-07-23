"""Config resolution for polyomino assembly scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import integer_range_choice, uniform_choice_with_probabilities
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import resolve_puzzle_unit_size_scale
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds
from trace_tasks.tasks.shared.named_colors import sample_named_color

from .state import (
    DEFAULTS,
    RENDER_DEFAULTS,
    SCENE_VARIANTS,
    AssemblyRenderParams,
)


def resolve_scene_variant(
    *,
    rng,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    """Resolve the scene visual variant as a uniform semantic axis."""

    del params
    configured = generation_defaults.get("scene_variants", SCENE_VARIANTS)
    variants = tuple(str(value) for value in configured) or SCENE_VARIANTS
    selected, probabilities = uniform_choice_with_probabilities(rng, variants)
    return str(selected), dict(probabilities)


def resolve_total_cell_count(
    *,
    rng,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> tuple[int, tuple[int, int], dict[str, float]]:
    """Sample the total target area from configured inclusive bounds."""

    lower, upper = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="total_cell_count_min",
        max_key="total_cell_count_max",
        fallback_min=DEFAULTS.total_cell_count_min,
        fallback_max=DEFAULTS.total_cell_count_max,
        context="polyomino assembly total cell count",
    )
    selected, probabilities = integer_range_choice(rng, int(lower), int(upper))
    return int(selected), (int(lower), int(upper)), dict(probabilities)


def generation_int(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer generation knob from params/defaults/fallback."""

    return int(params.get(str(key), group_default(generation_defaults, str(key), int(fallback))))


def resolve_render_params(
    params: Mapping[str, Any],
    *,
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
) -> AssemblyRenderParams:
    """Resolve render params before palette colors are injected."""

    def _int_value(key: str, fallback: int) -> int:
        return int(params.get(str(key), group_default(rendering_defaults, str(key), int(fallback))))

    unit_size_scale, unit_size_jitter = resolve_puzzle_unit_size_scale(
        params=params,
        defaults=rendering_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.polyomino_assembly.unit_size",
        fallback_min=0.5,
        fallback_max=1.05,
    )
    cell_size = int(
        round(
            float(_int_value("cell_size_px", RENDER_DEFAULTS.cell_size_px))
            * float(unit_size_scale)
        )
    )
    cell_gap = max(2, int(round(0.09 * float(cell_size))))
    color_rng = spawn_rng(
        int(instance_seed),
        "puzzles.polyomino_assembly.shape_color",
    )
    shape_color_name, shape_color_rgb = sample_named_color(color_rng)
    return AssemblyRenderParams(
        canvas_width=_int_value("canvas_width", RENDER_DEFAULTS.canvas_width),
        canvas_height=_int_value("canvas_height", RENDER_DEFAULTS.canvas_height),
        scene_margin_left_px=_int_value(
            "scene_margin_left_px",
            RENDER_DEFAULTS.scene_margin_left_px,
        ),
        scene_margin_right_px=_int_value(
            "scene_margin_right_px",
            RENDER_DEFAULTS.scene_margin_right_px,
        ),
        scene_margin_top_px=_int_value(
            "scene_margin_top_px",
            RENDER_DEFAULTS.scene_margin_top_px,
        ),
        scene_margin_bottom_px=_int_value(
            "scene_margin_bottom_px",
            RENDER_DEFAULTS.scene_margin_bottom_px,
        ),
        cell_size_px=int(cell_size),
        cell_gap_px=int(cell_gap),
        panel_padding_px=_int_value("panel_padding_px", RENDER_DEFAULTS.panel_padding_px),
        top_panel_height_px=_int_value(
            "top_panel_height_px",
            RENDER_DEFAULTS.top_panel_height_px,
        ),
        top_to_options_gap_px=_int_value(
            "top_to_options_gap_px",
            RENDER_DEFAULTS.top_to_options_gap_px,
        ),
        option_panel_width_px=_int_value(
            "option_panel_width_px",
            RENDER_DEFAULTS.option_panel_width_px,
        ),
        option_panel_height_px=_int_value(
            "option_panel_height_px",
            RENDER_DEFAULTS.option_panel_height_px,
        ),
        option_gap_px=_int_value("option_gap_px", RENDER_DEFAULTS.option_gap_px),
        option_row_gap_px=_int_value(
            "option_row_gap_px",
            RENDER_DEFAULTS.option_row_gap_px,
        ),
        panel_corner_radius_px=_int_value(
            "panel_corner_radius_px",
            RENDER_DEFAULTS.panel_corner_radius_px,
        ),
        cell_corner_radius_px=_int_value(
            "cell_corner_radius_px",
            RENDER_DEFAULTS.cell_corner_radius_px,
        ),
        border_width_px=_int_value("border_width_px", RENDER_DEFAULTS.border_width_px),
        option_label_font_size_px=_int_value(
            "option_label_font_size_px",
            RENDER_DEFAULTS.option_label_font_size_px,
        ),
        source_gap_px=_int_value("source_gap_px", RENDER_DEFAULTS.source_gap_px),
        panel_fill_rgb=(242, 245, 248),
        option_panel_fill_rgb=(255, 255, 255),
        shape_fill_rgb=tuple(int(value) for value in shape_color_rgb),
        source_shape_fill_rgb=tuple(int(value) for value in shape_color_rgb),
        shape_color_name=str(shape_color_name),
        border_color_rgb=(70, 80, 96),
        text_color_rgb=(31, 36, 46),
        text_stroke_rgb=(255, 255, 255),
        unit_size_jitter=dict(unit_size_jitter),
    )


def apply_scene_palette(render_params: AssemblyRenderParams, scene_style: Any) -> AssemblyRenderParams:
    """Inject the selected puzzle treatment colors into render params."""

    return replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        option_panel_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        border_color_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
    )


__all__ = [
    "apply_scene_palette",
    "generation_int",
    "resolve_render_params",
    "resolve_scene_variant",
    "resolve_total_cell_count",
]

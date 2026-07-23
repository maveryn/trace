"""Config fallback and axis-resolution helpers for Raven-matrix puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.sampling import integer_range_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
)
from trace_tasks.tasks.shared.render_variation import resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .state import RavenAxes, RavenRenderParams, SUPPORTED_SCENE_VARIANTS


@dataclass(frozen=True)
class RavenDefaults:
    """Stable fallback defaults for Raven-matrix scene construction."""

    option_count: int = 4
    count_min: int = 1
    count_max: int = 8


@dataclass(frozen=True)
class RavenRenderDefaults:
    """Stable fallback render defaults for Raven-matrix scenes."""

    canvas_width: int = 1200
    canvas_height: int = 920
    scene_margin_left_px: int = 64
    scene_margin_right_px: int = 64
    scene_margin_top_px: int = 56
    scene_margin_bottom_px: int = 56
    cell_size_px: int = 126
    cell_gap_px: int = 14
    board_panel_padding_px: int = 22
    board_to_options_gap_px: int = 76
    option_panel_width_px: int = 144
    option_panel_height_px: int = 180
    option_gap_px: int = 20
    option_symbol_box_size_px: int = 118
    option_label_gap_px: int = 16
    slot_corner_radius_px: int = 18
    border_width_px: int = 3
    panel_corner_radius_px: int = 28
    value_font_size_px: int = 52
    option_label_font_size_px: int = 30


RAVEN_DEFAULTS = RavenDefaults()
RENDER_DEFAULTS = RavenRenderDefaults()


def resolve_option_count(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> int:
    """Resolve the fixed four-option Raven answer panel count."""

    value = int(
        params.get(
            "option_count",
            group_default(
                generation_defaults,
                "option_count",
                int(RAVEN_DEFAULTS.option_count),
            ),
        )
    )
    if int(value) != 4:
        raise ValueError("Raven-matrix tasks require exactly 4 options")
    return int(value)


def resolve_count_bounds(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> tuple[int, int]:
    """Resolve count-panel value bounds for count-progression samples."""

    return resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="count_min",
        max_key="count_max",
        fallback_min=int(RAVEN_DEFAULTS.count_min),
        fallback_max=int(RAVEN_DEFAULTS.count_max),
        context="Raven count-progression bounds",
    )


def resolve_scene_variant(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one Raven visual scene variant from explicit support."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant.balance",
    )
    return str(balanced), {str(key): float(value) for key, value in probabilities.items()}


def resolve_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> RavenAxes:
    """Resolve scene variant, option count, and answer slot axes."""

    scene_variant, scene_probabilities = resolve_scene_variant(
        params,
        generation_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    option_count = resolve_option_count(params, generation_defaults)
    explicit_index = params.get("correct_option_index")
    if explicit_index is not None:
        answer_index = int(explicit_index)
        if not 0 <= int(answer_index) < int(option_count):
            raise ValueError("correct_option_index must be inside the option range")
        answer_probabilities = {
            str(index): (1.0 if index == int(answer_index) else 0.0)
            for index in range(int(option_count))
        }
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.answer_option")
        answer_index, answer_probabilities = integer_range_choice(
            rng,
            0,
            int(option_count) - 1,
        )
    return RavenAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probabilities),
        option_count=int(option_count),
        answer_option_index=int(answer_index),
        answer_option_probabilities={
            str(key): float(value)
            for key, value in answer_probabilities.items()
        },
    )


def resolve_render_params(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> RavenRenderParams:
    """Resolve render parameters for one Raven matrix sample."""

    def _int(key: str, fallback: int) -> int:
        return int(
            resolve_render_int(
                params,
                rendering_defaults,
                str(key),
                int(fallback),
                instance_seed=int(instance_seed),
                namespace="raven_matrix_render",
            )
        )

    def _rgb(key: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(
            int(value)
            for value in resolve_render_rgb(
                params,
                rendering_defaults,
                str(key),
                fallback,
                instance_seed=int(instance_seed),
                namespace="raven_matrix_render",
            )
        )

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.raven_matrix.unit_size",
    )

    return RavenRenderParams(
        canvas_width=_int("canvas_width", RENDER_DEFAULTS.canvas_width),
        canvas_height=_int("canvas_height", RENDER_DEFAULTS.canvas_height),
        scene_margin_left_px=_int(
            "scene_margin_left_px",
            RENDER_DEFAULTS.scene_margin_left_px,
        ),
        scene_margin_right_px=_int(
            "scene_margin_right_px",
            RENDER_DEFAULTS.scene_margin_right_px,
        ),
        scene_margin_top_px=_int(
            "scene_margin_top_px",
            RENDER_DEFAULTS.scene_margin_top_px,
        ),
        scene_margin_bottom_px=_int(
            "scene_margin_bottom_px",
            RENDER_DEFAULTS.scene_margin_bottom_px,
        ),
        cell_size_px=scale_puzzle_px(
            _int("cell_size_px", RENDER_DEFAULTS.cell_size_px),
            unit_scale,
            min_px=20,
        ),
        cell_gap_px=scale_puzzle_px(
            _int("cell_gap_px", RENDER_DEFAULTS.cell_gap_px),
            unit_scale,
            min_px=2,
        ),
        board_panel_padding_px=scale_puzzle_px(
            _int("board_panel_padding_px", RENDER_DEFAULTS.board_panel_padding_px),
            unit_scale,
            min_px=10,
        ),
        board_to_options_gap_px=scale_puzzle_px(
            _int("board_to_options_gap_px", RENDER_DEFAULTS.board_to_options_gap_px),
            unit_scale,
            min_px=22,
        ),
        option_panel_width_px=_int(
            "option_panel_width_px",
            RENDER_DEFAULTS.option_panel_width_px,
        ),
        option_panel_height_px=_int(
            "option_panel_height_px",
            RENDER_DEFAULTS.option_panel_height_px,
        ),
        option_gap_px=_int("option_gap_px", RENDER_DEFAULTS.option_gap_px),
        option_symbol_box_size_px=scale_puzzle_px(
            _int("option_symbol_box_size_px", RENDER_DEFAULTS.option_symbol_box_size_px),
            unit_scale,
            min_px=46,
        ),
        option_label_gap_px=_int(
            "option_label_gap_px",
            RENDER_DEFAULTS.option_label_gap_px,
        ),
        slot_corner_radius_px=scale_puzzle_px(
            _int("slot_corner_radius_px", RENDER_DEFAULTS.slot_corner_radius_px),
            unit_scale,
            min_px=6,
        ),
        border_width_px=_int("border_width_px", RENDER_DEFAULTS.border_width_px),
        panel_corner_radius_px=_int(
            "panel_corner_radius_px",
            RENDER_DEFAULTS.panel_corner_radius_px,
        ),
        value_font_size_px=scale_puzzle_px(
            _int("value_font_size_px", RENDER_DEFAULTS.value_font_size_px),
            unit_scale,
            min_px=18,
        ),
        option_label_font_size_px=_int(
            "option_label_font_size_px",
            RENDER_DEFAULTS.option_label_font_size_px,
        ),
        panel_fill_rgb=_rgb("panel_fill_rgb", (248, 249, 252)),
        cell_fill_rgb=_rgb("cell_fill_rgb", (252, 252, 255)),
        unknown_cell_fill_rgb=_rgb("unknown_cell_fill_rgb", (242, 246, 255)),
        option_panel_fill_rgb=_rgb("option_panel_fill_rgb", (251, 251, 255)),
        option_symbol_fill_rgb=_rgb("option_symbol_fill_rgb", (252, 252, 255)),
        border_color_rgb=_rgb("border_color_rgb", (86, 94, 108)),
        text_color_rgb=_rgb("text_color_rgb", (30, 34, 40)),
        text_stroke_rgb=_rgb("text_stroke_rgb", (255, 255, 255)),
        accent_color_rgb=_rgb("accent_color_rgb", (54, 102, 180)),
        unit_size_jitter=dict(unit_meta),
    )


__all__ = [
    "RAVEN_DEFAULTS",
    "RENDER_DEFAULTS",
    "resolve_axes",
    "resolve_count_bounds",
    "resolve_option_count",
    "resolve_render_params",
    "resolve_scene_variant",
]

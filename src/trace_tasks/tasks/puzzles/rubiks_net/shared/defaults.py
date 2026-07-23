"""Config fallback and axis-resolution helpers for Rubik cube-net scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from trace_tasks.core.sampling import integer_range_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .state import RubiksAxes, RubiksRenderParams, SUPPORTED_SCENE_VARIANTS


def _int_value(mapping: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(mapping.get(str(key), int(fallback)))


def _rgb_value(
    mapping: Mapping[str, Any], key: str, fallback: Sequence[int]
) -> tuple[int, int, int]:
    raw = mapping.get(str(key), fallback)
    if (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes))
        and len(raw) >= 3
    ):
        return (int(raw[0]), int(raw[1]), int(raw[2]))
    return (int(fallback[0]), int(fallback[1]), int(fallback[2]))


def resolve_option_count(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> int:
    """Resolve Rubik option count; this scene currently renders four choices."""

    value = int(
        params.get(
            "option_count",
            group_default(generation_defaults, "option_count", 4),
        )
    )
    if int(value) != 4:
        raise ValueError("Rubik cube-net tasks require exactly 4 options")
    return int(value)


def resolve_scene_variant(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one Rubik visual scene variant from explicit support."""

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
    return str(balanced), {
        str(key): float(value) for key, value in probabilities.items()
    }


def resolve_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> RubiksAxes:
    """Resolve scene variant, option count, and answer-option slot axes."""

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
    return RubiksAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probabilities),
        option_count=int(option_count),
        answer_option_index=int(answer_index),
        answer_option_probabilities={
            str(key): float(value) for key, value in answer_probabilities.items()
        },
    )


def resolve_render_params(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    instance_seed: int | None = None,
) -> RubiksRenderParams:
    """Resolve Rubik render params from task params over scene defaults."""

    merged: dict[str, Any] = dict(rendering_defaults)
    merged.update(dict(params))
    defaults = RubiksRenderParams()
    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        rendering_defaults,
        instance_seed=instance_seed,
        namespace="puzzles.rubiks_net.unit_size",
    )
    return replace(
        defaults,
        canvas_width=_int_value(merged, "canvas_width", defaults.canvas_width),
        canvas_height=_int_value(merged, "canvas_height", defaults.canvas_height),
        scene_margin_left_px=_int_value(
            merged,
            "scene_margin_left_px",
            defaults.scene_margin_left_px,
        ),
        scene_margin_top_px=_int_value(
            merged,
            "scene_margin_top_px",
            defaults.scene_margin_top_px,
        ),
        main_cell_size_px=scale_puzzle_px(
            _int_value(merged, "main_cell_size_px", defaults.main_cell_size_px),
            unit_scale,
            min_px=18,
        ),
        face_gap_px=_int_value(merged, "face_gap_px", defaults.face_gap_px),
        net_panel_padding_px=scale_puzzle_px(
            _int_value(
                merged,
                "net_panel_padding_px",
                defaults.net_panel_padding_px,
            ),
            unit_scale,
            min_px=10,
        ),
        panel_corner_radius_px=scale_puzzle_px(
            _int_value(
                merged,
                "panel_corner_radius_px",
                defaults.panel_corner_radius_px,
            ),
            unit_scale,
            min_px=9,
        ),
        option_panel_width_px=_int_value(
            merged,
            "option_panel_width_px",
            defaults.option_panel_width_px,
        ),
        option_panel_height_px=_int_value(
            merged,
            "option_panel_height_px",
            defaults.option_panel_height_px,
        ),
        option_gap_px=scale_puzzle_px(
            _int_value(merged, "option_gap_px", defaults.option_gap_px),
            unit_scale,
            min_px=8,
        ),
        option_row_gap_px=scale_puzzle_px(
            _int_value(merged, "option_row_gap_px", defaults.option_row_gap_px),
            unit_scale,
            min_px=8,
        ),
        result_option_panel_width_px=_int_value(
            merged,
            "result_option_panel_width_px",
            defaults.result_option_panel_width_px,
        ),
        result_option_panel_height_px=_int_value(
            merged,
            "result_option_panel_height_px",
            defaults.result_option_panel_height_px,
        ),
        result_option_gap_px=scale_puzzle_px(
            _int_value(merged, "result_option_gap_px", defaults.result_option_gap_px),
            unit_scale,
            min_px=8,
        ),
        result_option_row_gap_px=_int_value(
            merged,
            "result_option_row_gap_px",
            defaults.result_option_row_gap_px,
        ),
        swatch_size_px=scale_puzzle_px(
            _int_value(merged, "swatch_size_px", defaults.swatch_size_px),
            unit_scale,
            min_px=34,
        ),
        border_width_px=scale_puzzle_px(
            _int_value(merged, "border_width_px", defaults.border_width_px),
            unit_scale,
            min_px=1,
        ),
        sticker_gap_px=scale_puzzle_px(
            _int_value(merged, "sticker_gap_px", defaults.sticker_gap_px),
            unit_scale,
            min_px=1,
        ),
        option_label_font_size_px=_int_value(
            merged,
            "option_label_font_size_px",
            defaults.option_label_font_size_px,
        ),
        face_label_font_size_px=scale_puzzle_px(
            _int_value(
                merged,
                "face_label_font_size_px",
                defaults.face_label_font_size_px,
            ),
            unit_scale,
            min_px=12,
        ),
        small_label_font_size_px=scale_puzzle_px(
            _int_value(
                merged,
                "small_label_font_size_px",
                defaults.small_label_font_size_px,
            ),
            unit_scale,
            min_px=10,
        ),
        number_font_size_px=scale_puzzle_px(
            _int_value(merged, "number_font_size_px", defaults.number_font_size_px),
            unit_scale,
            min_px=18,
        ),
        panel_fill_rgb=_rgb_value(merged, "panel_fill_rgb", defaults.panel_fill_rgb),
        net_panel_fill_rgb=_rgb_value(
            merged,
            "net_panel_fill_rgb",
            defaults.net_panel_fill_rgb,
        ),
        option_panel_fill_rgb=_rgb_value(
            merged,
            "option_panel_fill_rgb",
            defaults.option_panel_fill_rgb,
        ),
        target_swatch_panel_fill_rgb=_rgb_value(
            merged,
            "target_swatch_panel_fill_rgb",
            defaults.target_swatch_panel_fill_rgb,
        ),
        sticker_outline_rgb=_rgb_value(
            merged,
            "sticker_outline_rgb",
            defaults.sticker_outline_rgb,
        ),
        border_color_rgb=_rgb_value(
            merged,
            "border_color_rgb",
            defaults.border_color_rgb,
        ),
        text_color_rgb=_rgb_value(merged, "text_color_rgb", defaults.text_color_rgb),
        text_stroke_rgb=_rgb_value(
            merged,
            "text_stroke_rgb",
            defaults.text_stroke_rgb,
        ),
        coordinate_fill_rgb=_rgb_value(
            merged,
            "coordinate_fill_rgb",
            defaults.coordinate_fill_rgb,
        ),
        coordinate_grid_rgb=_rgb_value(
            merged,
            "coordinate_grid_rgb",
            defaults.coordinate_grid_rgb,
        ),
        unit_size_jitter=dict(unit_meta),
    )


__all__ = [
    "resolve_axes",
    "resolve_option_count",
    "resolve_render_params",
    "resolve_scene_variant",
]
